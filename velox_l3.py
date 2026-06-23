"""


  Ce que L3 + L4 ensemble apporteront :
    - Le vrai forward batché (N séquences → 1 forward() par step)
    - Le gain GPU maximal

Usage :
  python velox_l3.py --model gpt2 --device cpu --dtype float32 --no-chat-template --demo
  python velox_l3.py --model gpt2 --device cpu --dtype float32 --no-chat-template --benchmark
"""
from __future__ import annotations

import argparse
import json
import logging
import queue
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from velox_l0 import (
        VeloxConfig, GenerationResult, NaiveModelRunner,
        DEFAULT_PROMPTS, PercentileStats,
    )
    from velox_l1 import KVModelRunner
except ImportError as e:
    print(f"ERREUR : {e}")
    print("Mets velox_l0.py, velox_l1.py et velox_l3.py dans le même dossier.")
    sys.exit(1)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST STATE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RequestState:
    req_id: str
    prompt: str
    max_new_tokens: int
    greedy: bool = True
    use_chat_template: bool = False


    result: Optional[GenerationResult] = None


    t_submitted: float = field(default_factory=time.perf_counter)
    t_started: Optional[float] = None  
    t_finished: Optional[float] = None  

    @property
    def queue_wait_ms(self) -> float:
        if self.t_started is None:
            return 0.0
        return (self.t_started - self.t_submitted) * 1000.0

    @property
    def e2e_ms(self) -> float:
        if self.t_finished is None or self.t_submitted is None:
            return 0.0
        return (self.t_finished - self.t_submitted) * 1000.0


# ══════════════════════════════════════════════════════════════════════════════
# SCHEDULER
#FCFS (First Come First Served).
#   Référence : ORCA (Yu et al., OSDI '22) — source académique du
#               continuous batching.
# ══════════════════════════════════════════════════════════════════════════════

class Scheduler:


    def __init__(self, max_batch_size: int = 4, max_queue_depth: int = 64) -> None:
        self.max_batch_size = max_batch_size
        self.max_queue_depth = max_queue_depth

        self._queue: queue.Queue[RequestState] = queue.Queue(maxsize=max_queue_depth)
        self._running: Dict[str, RequestState] = {} 
        self._lock = threading.Lock()

    def submit(self, req: RequestState) -> bool:

        try:
            self._queue.put_nowait(req)
            return True
        except queue.Full:
            logger.warning("File pleine — requête %s rejetée.", req.req_id[:8])
            return False

    def admit_pending(self) -> List[RequestState]:

        newly_admitted = []
        with self._lock:
            slots_free = self.max_batch_size - len(self._running)
            while slots_free > 0:
                try:
                    req = self._queue.get_nowait()
                    req.t_started = time.perf_counter()
                    self._running[req.req_id] = req
                    newly_admitted.append(req)
                    slots_free -= 1
                except queue.Empty:
                    break
        return newly_admitted

    def evict(self, req_id: str) -> Optional[RequestState]:
        with self._lock:
            return self._running.pop(req_id, None)

    @property
    def running(self) -> Dict[str, RequestState]:
        with self._lock:
            return dict(self._running)

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def batch_size(self) -> int:
        with self._lock:
            return len(self._running)


# ══════════════════════════════════════════════════════════════════════════════
# CONTINUOUS BATCH ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class ContinuousBatchEngine:


    def __init__(self, config: VeloxConfig, max_batch_size: int = 4) -> None:
        self.config = config
        self.scheduler = Scheduler(max_batch_size=max_batch_size)

        self._result_events: Dict[str, threading.Event] = {}
        self._results: Dict[str, RequestState] = {}
        self._lock = threading.Lock()


        logger.info("Chargement du modèle pour L3…")
        self._runner = KVModelRunner(config)


        self.stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "steps": 0,
        }


        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="velox-worker")
        self._worker.start()
        logger.info("Worker thread démarré.")


    def submit(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        greedy: bool = True,
        use_chat_template: bool = True,
    ) -> str:

        req_id = str(uuid.uuid4())[:8]
        req = RequestState(
            req_id=req_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            greedy=greedy,
            use_chat_template=use_chat_template,
        )
        event = threading.Event()
        with self._lock:
            self._result_events[req_id] = event
        self.scheduler.submit(req)
        return req_id

    def wait_result(self, req_id: str, timeout: float = 300.0) -> Optional[RequestState]:

        with self._lock:
            event = self._result_events.get(req_id)
        if event is None:
            return None
        event.wait(timeout=timeout)
        with self._lock:
            return self._results.get(req_id)


        logger.info("Arrêt du moteur…")
        self._stop_event.set()
        self._worker.join(timeout=30.0)

    @property
    def queue_depth(self) -> int:
        return self.scheduler.queue_depth

    @property
    def batch_size(self) -> int:
        return self.scheduler.batch_size

    def _worker_loop(self) -> None:
        """
        Boucle principale du worker.
        Tourne en continu, admet des requêtes, les traite, publie les résultats.

        Structure :
          while not stop:
            admit_pending()          ← remplir les slots libres
            for req in running:
              result = process(req)  ← L1 generate() (séquentiel sans L4)
              publish(result)        ← signaler au thread demandeur
        """
        logger.debug("Worker loop démarrée.")

        while not self._stop_event.is_set():

            admitted = self.scheduler.admit_pending()
            if admitted:
                logger.debug(
                    "Admis %d requête(s) | batch=%d | queue=%d",
                    len(admitted), self.scheduler.batch_size, self.scheduler.queue_depth,
                )

            running_snapshot = self.scheduler.running
            if not running_snapshot:
                time.sleep(0.001)
                continue

            self.stats["steps"] += 1


            for req_id, req in running_snapshot.items():
                result = self._runner.generate(
                    req.prompt,
                    max_new_tokens=req.max_new_tokens,
                    greedy=req.greedy,
                    use_chat_template=req.use_chat_template,
                )
                req.result = result
                req.t_finished = time.perf_counter()
                self.stats["total_tokens"] += result.output_tokens
                self.stats["total_requests"] += 1


                self.scheduler.evict(req_id)


                with self._lock:
                    self._results[req_id] = req
                    event = self._result_events.get(req_id)
                if event:
                    event.set()


                newly = self.scheduler.admit_pending()
                if newly:
                    logger.debug(
                        "Slot libéré → admis %s | queue=%d",
                        newly[0].req_id[:8], self.scheduler.queue_depth,
                    )

        logger.debug("Worker loop terminée.")


# ══════════════════════════════════════════════════════════════════════════════
# DÉMONSTRATION 
# ══════════════════════════════════════════════════════════════════════════════

def run_mixed_demo(config: VeloxConfig, max_batch_size: int = 2) -> dict:

    SHORT_TOKENS = 30
    LONG_TOKENS  = 150


    prompts_short = DEFAULT_PROMPTS[:4]
    prompts_long  = DEFAULT_PROMPTS[4:8]

    all_prompts  = [p for pair in zip(prompts_short, prompts_long) for p in pair]
    token_limits = [SHORT_TOKENS, LONG_TOKENS] * 4

    print(f"\n  Workload mixte : {len(prompts_short)} × {SHORT_TOKENS} tok + "
          f"{len(prompts_long)} × {LONG_TOKENS} tok")
    print(f"  max_batch_size = {max_batch_size}\n")


    print("  L2 (séquentiel lot par lot)…")
    runner_ref = KVModelRunner(config)
    t_l2_start = time.perf_counter()
    l2_results = []
    for p, n in zip(all_prompts, token_limits):
        r = runner_ref.generate(p, max_new_tokens=n, greedy=True, use_chat_template=False)
        l2_results.append(r)
    t_l2_total = time.perf_counter() - t_l2_start
    l2_tps = sum(r.output_tokens for r in l2_results) / t_l2_total


    print(f"  L3 (continuous, max_batch={max_batch_size})…")
    engine = ContinuousBatchEngine(config, max_batch_size=max_batch_size)
    time.sleep(0.1)  

    t_l3_start = time.perf_counter()

    req_ids = []
    submit_threads = []

    def _submit(prompt, n_tokens):
        req_id = engine.submit(prompt, max_new_tokens=n_tokens, greedy=True, use_chat_template=False)
        req_ids.append(req_id)

    for p, n in zip(all_prompts, token_limits):
        t = threading.Thread(target=_submit, args=(p, n))
        submit_threads.append(t)
        t.start()

    for t in submit_threads:
        t.join()


    req_ids_snapshot = list(req_ids)
    all_req_states = [engine.wait_result(rid) for rid in req_ids_snapshot]

    t_l3_total = time.perf_counter() - t_l3_start

    engine.shutdown()


    l3_total_tokens = sum(
        rs.result.output_tokens for rs in all_req_states if rs and rs.result
    )
    l3_tps = l3_total_tokens / t_l3_total


    queue_waits = [rs.queue_wait_ms for rs in all_req_states if rs]
    e2e_times   = [rs.e2e_ms for rs in all_req_states if rs]

    print()
    print("╔══ L2 vs L3 — Workload mixte ═══════════════════════════════════╗")
    print(f"  {'Métrique':<28} {'L2 Séquentiel':>14} {'L3 Continu':>12}")
    print("  " + "─" * 56)
    print(f"  {'Temps total (s)':<28} {t_l2_total:>14.1f} {t_l3_total:>12.1f}")
    print(f"  {'Débit (tok/s)':<28} {l2_tps:>14.1f} {l3_tps:>12.1f}")
    gain = l3_tps / max(l2_tps, 0.001)
    print(f"  {'Gain débit':<28} {'—':>14} {gain:>11.1f}×")

    if queue_waits:
        print(f"  {'Attente queue p50 (ms)':<28} {'—':>14} {float(np.median(queue_waits)):>12.1f}")
    if e2e_times:
        print(f"  {'E2E p50 (ms)':<28} {'—':>14} {float(np.median(e2e_times)):>12.1f}")

    print("╚" + "═" * 66 + "╝")

    print()
    print("  Détail par requête (L3) :")
    print(f"  {'Req':>4}  {'Longueur':>8}  {'Queue wait':>12}  {'E2E (ms)':>10}  {'Tokens':>8}")
    print("  " + "─" * 48)
    for i, rs in enumerate(all_req_states):
        if rs and rs.result:
            kind = "court" if token_limits[i] == SHORT_TOKENS else "long "
            print(f"  {i+1:>4}  {kind:>8}  {rs.queue_wait_ms:>10.0f}ms  "
                  f"{rs.e2e_ms:>8.0f}ms  {rs.result.output_tokens:>8}")
    print()

    return {
        "l2": {"total_s": round(t_l2_total, 2), "tps": round(l2_tps, 1)},
        "l3": {"total_s": round(t_l3_total, 2), "tps": round(l3_tps, 1), "gain": round(gain, 2)},
    }


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 
# ══════════════════════════════════════════════════════════════════════════════

def run_throughput_benchmark(
    config: VeloxConfig,
    max_new_tokens: int = 80,
    concurrency_levels: List[int] = [1, 2, 4, 8],
) -> dict:
    """
    Mesure le débit total pour différents niveaux de concurrence.
    Compare L1 (séquentiel) et L3 (continuous scheduling).
    """
    prompts = DEFAULT_PROMPTS  

    print(f"\n  Benchmark débit — max_tokens={max_new_tokens}")
    print(f"  {'Concurrence':>12}  {'L3 tok/s':>10}  {'L1 ref tok/s':>14}  {'Gain':>8}")
    print("  " + "─" * 50)

    
    runner_ref = KVModelRunner(config)
    t_ref = time.perf_counter()
    ref_results = [
        runner_ref.generate(p, max_new_tokens=max_new_tokens,
                            greedy=True, use_chat_template=False)
        for p in prompts
    ]
    t_ref_total = time.perf_counter() - t_ref
    ref_tps = sum(r.output_tokens for r in ref_results) / t_ref_total

    results = {}

    for n_concurrent in concurrency_levels:
        engine = ContinuousBatchEngine(config, max_batch_size=n_concurrent)
        time.sleep(0.1)

        t_start = time.perf_counter()
        req_ids = [
            engine.submit(p, max_new_tokens=max_new_tokens,
                          greedy=True, use_chat_template=False)
            for p in prompts
        ]
        req_states = [engine.wait_result(rid) for rid in req_ids]
        t_total = time.perf_counter() - t_start
        engine.shutdown()

        total_tokens = sum(rs.result.output_tokens for rs in req_states if rs and rs.result)
        tps = total_tokens / t_total
        gain = tps / max(ref_tps, 0.001)

        print(f"  {n_concurrent:>12}  {tps:>10.1f}  {ref_tps:>14.1f}  {gain:>7.1f}×")
        results[n_concurrent] = {"tps": round(tps, 1), "gain": round(gain, 2)}

    print()
    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(description="Velox L3 — continuous batching.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max-tokens", type=int, default=80)
    p.add_argument("--max-batch", type=int, default=2,
                   help="Taille max du batch courant (slots disponibles simultanément).")
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--demo", action="store_true",
                   help="Démonstration workload mixte court/long.")
    p.add_argument("--benchmark", action="store_true",
                   help="Sweep de concurrence L1 vs L3.")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config = VeloxConfig(model_name=args.model, device=args.device, dtype=args.dtype)
    results = {}

    if args.demo or (not args.benchmark):
        results["demo"] = run_mixed_demo(config, max_batch_size=args.max_batch)

    if args.benchmark:
        results["benchmark"] = run_throughput_benchmark(
            config, max_new_tokens=args.max_tokens,
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"  Rapport → {args.output}")


if __name__ == "__main__":
    main()
