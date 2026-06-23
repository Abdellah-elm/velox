# velox_l7_colab.py
from __future__ import annotations
"""
velox_l7_colab.py — L7 : Benchmark Velox vs vLLM sur Colab T4.

"""


INSTALL = """
!pip install -q torch transformers accelerate numpy
!pip install -q vllm          # état de l'art — Linux/GPU uniquement
"""


import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("velox.l7")



MODEL_NAME   = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE       = "cuda"          # T4 sur Colab
DTYPE        = "float16"       # float16 sur T4
MAX_TOKENS   = 128
WARMUP       = 2

BENCH_PROMPTS = [
    "What is quantum computing?",
    "Explain the difference between a qubit and a classical bit.",
    "How does the Qiskit SDK help developers build quantum circuits?",
    "What is superposition in quantum mechanics?",
    "Describe the concept of quantum entanglement.",
    "What are the main challenges in building fault-tolerant quantum computers?",
    "How does error mitigation differ from error correction in quantum computing?",
    "What is the purpose of transpilation in Qiskit?",
]

# Charges à tester (nombre de requêtes simultanées)
CONCURRENCY_LEVELS = [1, 4, 8, 16, 32]


@dataclass
class BenchResult:
    runner: str
    concurrency: int
    throughput_tps: float
    ttft_p50_ms: float
    tpot_p50_ms: float
    total_tokens: int
    wall_time_s: float




def run_l0_naive(prompts: List[str], max_tokens: int, warmup: int) -> BenchResult:
    """
    L0 : model.generate() séquentiel.
    C'est la référence "dumb" — une requête à la fois, pas de batching.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("L0 Naïf : chargement %s…", MODEL_NAME)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="cuda", trust_remote_code=True,
    ).eval()

    # Warmup
    for p in prompts[:warmup]:
        enc = tok(p, return_tensors="pt").to("cuda")
        model.generate(**enc, max_new_tokens=32, pad_token_id=tok.pad_token_id)

    ttfts, tpots, total_toks = [], [], []
    t_start = time.perf_counter()

    for prompt in prompts:
        enc = tok(prompt, return_tensors="pt").to("cuda")
        prompt_len = enc["input_ids"].shape[1]

        from transformers import StoppingCriteria, StoppingCriteriaList
        class Timer(StoppingCriteria):
            def __init__(self, t0):
                self.t0 = t0; self.ttft = None; self._done = False
            def __call__(self, input_ids, scores, **kw):
                if not self._done:
                    self.ttft = (time.perf_counter() - self.t0) * 1000; self._done = True
                return False

        t_req = time.perf_counter()
        timer = Timer(t_req)
        with torch.inference_mode():
            out = model.generate(
                **enc, max_new_tokens=max_tokens,
                pad_token_id=tok.pad_token_id,
                stopping_criteria=StoppingCriteriaList([timer]),
            )
        t_end = time.perf_counter()

        n_out = out.shape[1] - prompt_len
        total_ms = (t_end - t_req) * 1000
        ttft = timer.ttft or total_ms
        tpot = (total_ms - ttft) / max(n_out - 1, 1)
        ttfts.append(ttft); tpots.append(tpot); total_toks.append(n_out)

    wall = time.perf_counter() - t_start
    tps = sum(total_toks) / wall
    logger.info("L0 Naïf : %.1f tok/s | TTFT p50 %.0f ms | TPOT p50 %.1f ms",
                tps, np.percentile(ttfts, 50), np.percentile(tpots, 50))

    return BenchResult(
        runner="L0_Naive", concurrency=1,
        throughput_tps=tps,
        ttft_p50_ms=float(np.percentile(ttfts, 50)),
        tpot_p50_ms=float(np.percentile(tpots, 50)),
        total_tokens=sum(total_toks), wall_time_s=wall,
    )


def run_l1_kv(prompts: List[str], max_tokens: int, warmup: int) -> BenchResult:

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("L1 KV Cache : chargement…")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="cuda", trust_remote_code=True,
    ).eval()

    def generate_one(prompt, max_new_tokens):
        enc = tok(prompt, return_tensors="pt").to("cuda")
        prompt_len = enc["input_ids"].shape[1]
        attn = enc["attention_mask"]

        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(**enc, use_cache=True)
        past = out.past_key_values
        next_tok = int(out.logits[0, -1].argmax().item())
        ttft = (time.perf_counter() - t0) * 1000
        generated = [next_tok]

        for step in range(max_new_tokens - 1):
            if next_tok == tok.eos_token_id: break
            attn = torch.cat([attn, torch.ones(1,1,device="cuda",dtype=attn.dtype)], dim=1)
            with torch.inference_mode():
                out2 = model(
                    input_ids=torch.tensor([[next_tok]], device="cuda"),
                    attention_mask=attn,
                    position_ids=torch.tensor([[prompt_len + step]], device="cuda"),
                    past_key_values=past, use_cache=True,
                )
            past = out2.past_key_values
            next_tok = int(out2.logits[0, -1].argmax().item())
            generated.append(next_tok)

        total = (time.perf_counter() - t0) * 1000
        tpot = (total - ttft) / max(len(generated) - 1, 1)
        return len(generated), ttft, tpot

    for p in prompts[:warmup]:
        generate_one(p, 32)

    ttfts, tpots, toks = [], [], []
    t_start = time.perf_counter()
    for p in prompts:
        n, ttft, tpot = generate_one(p, max_tokens)
        ttfts.append(ttft); tpots.append(tpot); toks.append(n)

    wall = time.perf_counter() - t_start
    tps = sum(toks) / wall
    logger.info("L1 KV Cache : %.1f tok/s | TTFT p50 %.0f ms | TPOT p50 %.1f ms",
                tps, np.percentile(ttfts, 50), np.percentile(tpots, 50))

    return BenchResult(
        runner="L1_KVCache", concurrency=1,
        throughput_tps=tps,
        ttft_p50_ms=float(np.percentile(ttfts, 50)),
        tpot_p50_ms=float(np.percentile(tpots, 50)),
        total_tokens=sum(toks), wall_time_s=wall,
    )



def run_l2_static_batching(
    prompts: List[str], max_tokens: int, warmup: int,
    batch_sizes: List[int] = [1, 4, 8],
) -> List[BenchResult]:
    """
    L2 : batching statique, sweep de taille de batch.
    Montre le gain de débit avec le batch size (H2 partielle).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch.nn.functional as F

    logger.info("L2 Static Batching : chargement…")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="cuda", trust_remote_code=True,
    ).eval()

    def generate_batch(batch_prompts, max_new_tokens):
        enc = tok(batch_prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to("cuda")
        B = enc["input_ids"].shape[1]
        attn = enc["attention_mask"]
        pos_ids = (attn.cumsum(-1) - 1).clamp(min=0)

        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(input_ids=enc["input_ids"], attention_mask=attn,
                        position_ids=pos_ids, use_cache=True)
        past = out.past_key_values
        next_toks = out.logits[:, -1].argmax(-1)
        ttft = (time.perf_counter() - t0) * 1000

        real_lens = attn.sum(-1)
        generated = [[t.item()] for t in next_toks]
        finished = torch.zeros(len(batch_prompts), dtype=torch.bool, device="cuda")
        max_kv = enc["input_ids"].shape[1]

        for step in range(max_new_tokens - 1):
            finished |= (next_toks == tok.eos_token_id)
            if finished.all(): break

            new_col = (~finished).long().unsqueeze(1)
            attn = torch.cat([attn, new_col], dim=1)
            decode_pos = (real_lens + step).unsqueeze(1)
            feed = next_toks.clone(); feed[finished] = tok.pad_token_id

            with torch.inference_mode():
                out2 = model(
                    input_ids=feed.unsqueeze(1), attention_mask=attn,
                    position_ids=decode_pos, past_key_values=past, use_cache=True,
                )
            past = out2.past_key_values
            next_toks = out2.logits[:, -1].argmax(-1)
            for i in range(len(batch_prompts)):
                if not finished[i]: generated[i].append(next_toks[i].item())

        total = (time.perf_counter() - t0) * 1000
        n_toks = [len(g) for g in generated]
        tpot = (total - ttft) / max(max(n_toks) - 1, 1)
        return n_toks, ttft, tpot

    # Warmup
    generate_batch(prompts[:2], 16)

    results = []
    for bs in batch_sizes:
        ttfts, tpots, toks = [], [], []
        t_start = time.perf_counter()

        for i in range(0, len(prompts), bs):
            chunk = prompts[i:i+bs]
            ns, ttft, tpot = generate_batch(chunk, max_tokens)
            ttfts.append(ttft); tpots.append(tpot); toks.extend(ns)

        wall = time.perf_counter() - t_start
        tps = sum(toks) / wall
        logger.info("L2 batch=%d : %.1f tok/s | TTFT p50 %.0f ms | TPOT p50 %.1f ms",
                    bs, tps, np.percentile(ttfts, 50), np.percentile(tpots, 50))

        results.append(BenchResult(
            runner=f"L2_Batch{bs}", concurrency=bs,
            throughput_tps=tps,
            ttft_p50_ms=float(np.percentile(ttfts, 50)),
            tpot_p50_ms=float(np.percentile(tpots, 50)),
            total_tokens=sum(toks), wall_time_s=wall,
        ))
    return results


def run_vllm(
    prompts: List[str], max_tokens: int,
    concurrency_levels: List[int],
) -> List[BenchResult]:
    """
    vLLM : état de l'art. Linux + GPU uniquement.
    On simule différents niveaux de concurrence en envoyant
    N requêtes simultanément via l'API async de vLLM.

    La comparaison honnête (§14 du CdC) :
      - vLLM sera plus rapide en absolu (noyaux CUDA, années d'optimisation)
      - Ce qui compte : la FORME de la courbe débit~concurrence est-elle similaire ?
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        logger.warning("vLLM non installé. Installer avec : pip install vllm")
        return []

    logger.info("vLLM : chargement %s…", MODEL_NAME)
    llm = LLM(model=MODEL_NAME, dtype="float16", max_model_len=1024)
    sampling = SamplingParams(max_tokens=max_tokens, temperature=0)

    results = []
    for n_concurrent in concurrency_levels:
        # Répéter les prompts pour atteindre n_concurrent
        batch = (prompts * (n_concurrent // len(prompts) + 1))[:n_concurrent]

        # Warmup
        llm.generate(batch[:2], SamplingParams(max_tokens=16, temperature=0))

        t_start = time.perf_counter()
        outputs = llm.generate(batch, sampling)
        wall = time.perf_counter() - t_start

        toks = [len(o.outputs[0].token_ids) for o in outputs]
        tps = sum(toks) / wall

        # vLLM n'expose pas TTFT/TPOT facilement via l'API sync
        # On note N/A et on documente dans le rapport
        logger.info("vLLM n=%d : %.1f tok/s", n_concurrent, tps)

        results.append(BenchResult(
            runner="vLLM", concurrency=n_concurrent,
            throughput_tps=tps,
            ttft_p50_ms=float("nan"),   # non disponible via API sync
            tpot_p50_ms=float("nan"),
            total_tokens=sum(toks), wall_time_s=wall,
        ))

    return results


# ══════════════════════════════════════════════════════════════════════════════
# CELLULE 8 — Rapport comparatif
# ══════════════════════════════════════════════════════════════════════════════

def print_report(
    l0: BenchResult,
    l1: BenchResult,
    l2_results: List[BenchResult],
    vllm_results: List[BenchResult],
) -> dict:
    """Tableau comparatif honest (§14 du CdC)."""

    print()
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║  Velox L7 — Benchmark comparatif                                        ║")
    print(f"║  Modèle : {MODEL_NAME:<62}║")
    print(f"║  Device : {DEVICE} | dtype : {DTYPE:<54}║")
    print("╠══════════════════════════════════════════════════════════════════════════╣")
    print(f"  {'Runner':<20} {'Concurrence':>12} {'Débit (tok/s)':>14} {'TTFT p50 (ms)':>14} {'TPOT p50 (ms)':>14}")
    print("  " + "─" * 76)

    all_results = [l0, l1] + l2_results + vllm_results
    report_data = []

    for r in all_results:
        ttft_str = f"{r.ttft_p50_ms:.1f}" if not (isinstance(r.ttft_p50_ms, float) and r.ttft_p50_ms != r.ttft_p50_ms) else "N/A"
        tpot_str = f"{r.tpot_p50_ms:.1f}" if not (isinstance(r.tpot_p50_ms, float) and r.tpot_p50_ms != r.tpot_p50_ms) else "N/A"
        print(f"  {r.runner:<20} {r.concurrency:>12} {r.throughput_tps:>14.1f} {ttft_str:>14} {tpot_str:>14}")
        report_data.append({
            "runner": r.runner, "concurrency": r.concurrency,
            "throughput_tps": round(r.throughput_tps, 2),
            "ttft_p50_ms": round(r.ttft_p50_ms, 1) if r.ttft_p50_ms == r.ttft_p50_ms else None,
            "tpot_p50_ms": round(r.tpot_p50_ms, 1) if r.tpot_p50_ms == r.tpot_p50_ms else None,
        })

    print("╠══════════════════════════════════════════════════════════════════════════╣")

    # Gains relatifs
    if l2_results:
        best_l2 = max(l2_results, key=lambda r: r.throughput_tps)
        gain_l2_vs_l0 = best_l2.throughput_tps / l0.throughput_tps
        gain_l1_vs_l0 = l1.throughput_tps / l0.throughput_tps
        tpot_gain = l0.tpot_p50_ms / l1.tpot_p50_ms if l1.tpot_p50_ms > 0 else 1

        print(f"  H1 — TPOT KV cache vs naïf  : {tpot_gain:.1f}× (TPOT {l0.tpot_p50_ms:.1f} → {l1.tpot_p50_ms:.1f} ms)")
        print(f"  H2 — Débit L2 best vs L0     : {gain_l2_vs_l0:.1f}× (batch={best_l2.concurrency})")

        if vllm_results:
            vllm_best = max(vllm_results, key=lambda r: r.throughput_tps)
            gap = vllm_best.throughput_tps / best_l2.throughput_tps
            print(f"  Écart vLLM vs Velox best     : {gap:.1f}× (normal — noyaux CUDA custom)")
            print(f"  Forme de courbe : voir graphe ci-dessous")

    print("╠══════════════════════════════════════════════════════════════════════════╣")
    print("  Note d'honnêteté (§14 du CdC) :")
    print("  · L'écart absolu vs vLLM est attendu et documenté.")
    print("  · Ce qui est comparé : la FORME de la courbe débit~concurrence.")
    print("  · Si les deux courbes ont la même forme (montée puis plateau),")
    print("    le bon mécanisme est reproduit, quelle que soit la valeur absolue.")
    print("╚══════════════════════════════════════════════════════════════════════════╝")

    # Graphe ASCII débit vs concurrence
    print()
    _ascii_throughput_chart(l2_results, vllm_results)

    return {"results": report_data, "model": MODEL_NAME, "device": DEVICE}


def _ascii_throughput_chart(l2_results: List[BenchResult], vllm_results: List[BenchResult]) -> None:
    """Graphe ASCII débit vs concurrence — Velox vs vLLM."""
    if not l2_results:
        return

    all_tps = [r.throughput_tps for r in l2_results + vllm_results]
    if not all_tps:
        return

    max_tps = max(all_tps)
    width = 40

    print("  Débit (tok/s) vs Concurrence :")
    print(f"  {'0':>4}{'':>{width // 2}}{max_tps:.0f}")
    print("  " + "─" * (width + 6))

    for r in l2_results:
        bar = int(r.throughput_tps / max_tps * width)
        label = f"Velox B={r.concurrency}"
        print(f"  {label:<12} │{'█' * bar}{' ' * (width - bar)}│ {r.throughput_tps:.0f}")

    for r in vllm_results:
        bar = int(r.throughput_tps / max_tps * width)
        label = f"vLLM n={r.concurrency}"
        print(f"  {label:<12} │{'▓' * bar}{' ' * (width - bar)}│ {r.throughput_tps:.0f}")

    print()


# ══════════════════════════════════════════════════════════════════════════════
# CELLULE 9 — MAIN (tout enchaîner)
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    p = argparse.ArgumentParser(description="Velox L7 — benchmark vs vLLM.")
    p.add_argument("--skip-vllm", action="store_true",
                   help="Skip vLLM (si pas installé ou pas sur Linux/GPU).")
    p.add_argument("--fast", action="store_true",
                   help="Mode rapide : moins de prompts, moins de tokens.")
    p.add_argument("--output", default="l7_benchmark.json",
                   help="Fichier JSON de sortie.")
    args = p.parse_args()

    max_tokens = 64 if args.fast else MAX_TOKENS
    prompts = BENCH_PROMPTS[:4] if args.fast else BENCH_PROMPTS
    batch_sizes = [1, 4, 8] if args.fast else [1, 4, 8, 16]

    print(f"\n  Velox L7 — Benchmark")
    print(f"  Modèle : {MODEL_NAME}")
    print(f"  {len(prompts)} prompts | max_tokens={max_tokens} | mode={'fast' if args.fast else 'full'}\n")

    # ── Vérification GPU ──────────────────────────────────────────────────────
    import torch
    global DEVICE
    if not torch.cuda.is_available():
        print("  ⚠  GPU non disponible. Benchmark CPU — résultats modestes.")
        print("     Pour les vrais chiffres : exécuter sur Colab T4.\n")
        DEVICE = "cpu"

    print(f"  Device : {DEVICE.upper()}\n")

    # ── L0 Naïf ──────────────────────────────────────────────────────────────
    print("── Étape 1/4 : L0 Naïf ─────────────────────────────────────────────")
    l0 = run_l0_naive(prompts, max_tokens, WARMUP)

    # ── L1 KV Cache ──────────────────────────────────────────────────────────
    print("\n── Étape 2/4 : L1 KV Cache ─────────────────────────────────────────")
    l1 = run_l1_kv(prompts, max_tokens, WARMUP)

    # ── L2 Static Batching ────────────────────────────────────────────────────
    print("\n── Étape 3/4 : L2 Static Batching ──────────────────────────────────")
    l2_results = run_l2_static_batching(prompts, max_tokens, WARMUP, batch_sizes)

    # ── vLLM ─────────────────────────────────────────────────────────────────
    print("\n── Étape 4/4 : vLLM ────────────────────────────────────────────────")
    if args.skip_vllm:
        print("  Skip vLLM (--skip-vllm).")
        vllm_results = []
    else:
        vllm_results = run_vllm(prompts, max_tokens, batch_sizes)

    # ── Rapport ───────────────────────────────────────────────────────────────
    print("\n── Résultats ────────────────────────────────────────────────────────")
    report = print_report(l0, l1, l2_results, vllm_results)

    # Sauvegarder
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n   Rapport sauvegardé → {args.output}")


if __name__ == "__main__":
    main()
