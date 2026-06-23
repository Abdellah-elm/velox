"""
    python velox_l0.py
    python velox_l0.py --model gpt2 --device cpu --dtype float32 --max-tokens 50
    python velox_l0.py --device cuda --max-tokens 256 --output results_l0.json
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

logger = logging.getLogger(__name__)


@dataclass
class VeloxConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "auto"    
    dtype: str = "float16"  

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"float16": torch.float16, "bfloat16": torch.bfloat16,
                "float32": torch.float32}[self.dtype]

    @property
    def resolved_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


#══════════════════════════════════════════════════════════════════════════════

SHORT_PROMPTS = [
    "What is machine learning?",
    "Explain attention in transformers in one paragraph.",
    "What is the difference between a CPU and a GPU?",
    "Name three production use cases for large language models.",
    "What is token generation latency and why does it matter?",
]

MEDIUM_PROMPTS = [
    "Explain how the transformer architecture works, including self-attention, "
    "positional encoding, and feed-forward layers. Keep it under 200 words.",
    "What are the main engineering challenges in deploying large language models "
    "at scale? Discuss latency, throughput, memory, and cost.",
    "Explain the PagedAttention mechanism used in vLLM and why it improves "
    "GPU memory efficiency compared to naive KV cache allocation.",
    "Describe how continuous batching differs from static batching in LLM "
    "inference servers, and why it leads to higher throughput.",
]

DEFAULT_PROMPTS = SHORT_PROMPTS + MEDIUM_PROMPTS[:3]


CORRECTNESS_PROMPTS = [
    "The capital of Morocco is",
    "Explain gradient descent in three sentences.",
    "Translate to French: 'The model generates tokens autoregressively.'",
]




@dataclass
class GenerationResult:
    prompt: str
    output_text: str
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float    # Time To First Token : prefill + 1er step de décodage
    tpot_ms: float    # Time Per Output Token : moyenne des steps de décodage suivants
    total_ms: float
    throughput_tps: float


# ══════════════════════════════════════════════════════════════════════════════

# Comment on mesure TTFT avec model.generate() :
#   HuggingFace appelle StoppingCriteria.__call__ après chaque token généré.
#   Le premier appel = premier token prêt = TTFT.
#   On enregistre time.perf_counter() à ce moment précis.
# ══════════════════════════════════════════════════════════════════════════════

class _FirstTokenTimer(StoppingCriteria):
    def __init__(self, t_start: float) -> None:
        self._t_start = t_start
        self.ttft_ms: float = 0.0
        self._recorded: bool = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if not self._recorded:
            self.ttft_ms = (time.perf_counter() - self._t_start) * 1000.0
            self._recorded = True
        return False  # ne jamais arrêter la génération depuis ce callback


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — NAIVE MODEL RUNNER (L0)
#
# Ce que fait cette classe :
#   1. Charge le modèle HuggingFace
#   2. Pour chaque requête : tokenise → model.generate() → détokenise
#   3. Mesure TTFT, TPOT, total
#
# Ce qu'elle ne fait PAS (et qui sera ajouté lot par lot) :
#   - Pas de KV cache manuel (L1)
#   - Pas de batching (L2/L3)
#   - Pas d'ordonnanceur (L3)
#   - Pas d'API REST (L5)
# ══════════════════════════════════════════════════════════════════════════════

class NaiveModelRunner:
    def __init__(self, config: VeloxConfig) -> None:
        self.config = config
        device = config.resolved_device
        logger.info("Chargement de %s sur %s (%s)…", config.model_name, device, config.dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=config.torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        self._device = device

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info("Chargé : %.1fM params | %s | %s", n_params / 1e6, device, config.dtype)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        greedy: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        use_chat_template: bool = True,
    ) -> GenerationResult:

        if seed is not None and not greedy:
            torch.manual_seed(seed)

        t_start = time.perf_counter()

        # Tokenisation
        formatted = self._apply_chat_template(prompt, use_chat_template)
        enc = self.tokenizer(formatted, return_tensors="pt").to(self._device)
        prompt_tokens = enc["input_ids"].shape[1]

        # Timer TTFT injecté via StoppingCriteria
        timer = _FirstTokenTimer(t_start)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([timer]),
        )
        if greedy:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)

        # Génération : model.generate() recalcule tout à chaque pas (référence L0)
        output_ids = self.model.generate(**enc, **gen_kwargs)
        t_end = time.perf_counter()

        # Décode uniquement les tokens générés (pas le prompt)
        new_ids = output_ids[0, prompt_tokens:]
        output_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        output_tokens = new_ids.shape[0]

        total_ms = (t_end - t_start) * 1000.0
        ttft_ms = timer.ttft_ms if timer._recorded else total_ms
        tpot_ms = (total_ms - ttft_ms) / max(output_tokens - 1, 1)

        return GenerationResult(
            prompt=prompt,
            output_text=output_text,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            total_ms=total_ms,
            throughput_tps=output_tokens / max(total_ms / 1000.0, 1e-9),
        )

    def _apply_chat_template(self, prompt: str, use_chat_template: bool) -> str:
        if not use_chat_template:
            return prompt
        if not getattr(self.tokenizer, "chat_template", None):
            return prompt
        try:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            return prompt

    def model_info(self) -> dict:
        cfg = self.model.config
        return {
            "model_name": self.config.model_name,
            "device": self._device,
            "dtype": self.config.dtype,
            "params_M": round(sum(p.numel() for p in self.model.parameters()) / 1e6, 1),
            "num_layers": getattr(cfg, "num_hidden_layers", "?"),
            "num_attention_heads": getattr(cfg, "num_attention_heads", "?"),
            "num_kv_heads": getattr(cfg, "num_key_value_heads", "?"),
            "hidden_size": getattr(cfg, "hidden_size", "?"),
            "vocab_size": getattr(cfg, "vocab_size", "?"),
        }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — STATS PERCENTILES
#
# Pourquoi p50/p90/p99 et pas juste la moyenne ?
#   La moyenne masque les cas pathologiques.
#   p99 = pire cas réaliste. Si p99 >> p50, il y a un outlier à investiguer.
#   En prod, c'est p99 qui détermine le SLA, pas la moyenne.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PercentileStats:
    p50: float
    p90: float
    p99: float
    mean: float
    min: float
    max: float

    @classmethod
    def from_list(cls, values: List[float]) -> "PercentileStats":
        arr = np.array(values)
        return cls(
            p50=float(np.percentile(arr, 50)),
            p90=float(np.percentile(arr, 90)),
            p99=float(np.percentile(arr, 99)),
            mean=float(arr.mean()),
            min=float(arr.min()),
            max=float(arr.max()),
        )


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK HARNESS
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark(
    config: VeloxConfig,
    prompts: List[str],
    max_new_tokens: int = 128,
    greedy: bool = True,
    warmup: int = 2,
    use_chat_template: bool = True,
) -> dict:

    runner = NaiveModelRunner(config)

    # Warmup — déclenche la compilation CUDA lazy et la mise en cache des poids
    logger.info("Warmup (%d requêtes)…", warmup)
    for p in (prompts * 2)[:warmup]:
        runner.generate(p, max_new_tokens=min(max_new_tokens, 32),
                        greedy=True, use_chat_template=use_chat_template)
    logger.info("Warmup terminé.")

    results: List[GenerationResult] = []
    t_bench_start = time.perf_counter()

    for i, prompt in enumerate(prompts):
        logger.info("[%d/%d] génération…", i + 1, len(prompts))
        r = runner.generate(prompt, max_new_tokens=max_new_tokens,
                            greedy=greedy, use_chat_template=use_chat_template)
        results.append(r)
        logger.info("  → %d tok | TTFT %.0f ms | TPOT %.1f ms | %.1f tok/s",
                    r.output_tokens, r.ttft_ms, r.tpot_ms, r.throughput_tps)

    total_wall = time.perf_counter() - t_bench_start
    total_tokens = sum(r.output_tokens for r in results)

    return {
        "lot": "L0",
        "mode": "naive_sequential",
        "model": config.model_name,
        "device": config.resolved_device,
        "dtype": config.dtype,
        "num_requests": len(results),
        "max_new_tokens": max_new_tokens,
        "greedy": greedy,
        "throughput_tps": round(total_tokens / total_wall, 2),
        "total_output_tokens": total_tokens,
        "total_wall_s": round(total_wall, 2),
        "ttft_ms": asdict(PercentileStats.from_list([r.ttft_ms for r in results])),
        "tpot_ms": asdict(PercentileStats.from_list([r.tpot_ms for r in results])),
        "e2e_ms":  asdict(PercentileStats.from_list([r.total_ms for r in results])),
        "per_request": [
            {"prompt_tokens": r.prompt_tokens, "output_tokens": r.output_tokens,
             "ttft_ms": round(r.ttft_ms, 1), "tpot_ms": round(r.tpot_ms, 1),
             "total_ms": round(r.total_ms, 1),
             "preview": r.output_text[:80] + "…" if len(r.output_text) > 80 else r.output_text}
            for r in results
        ],
        "model_info": runner.model_info(),
    }


def print_report(report: dict) -> None:
    t = report["ttft_ms"]
    p = report["tpot_ms"]
    e = report["e2e_ms"]
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║        Velox · L0 Baseline · Naïf séquentiel                    ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  Modèle    : {report['model']:<52}║")
    print(f"║  Device    : {report['device']} ({report['dtype']}){'':<46}║")
    print(f"║  Requêtes  : {report['num_requests']} | max_tokens: {report['max_new_tokens']}{'':<36}║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print("  ┌────────────────────┬──────────┬──────────┬──────────┐")
    print("  │ Métrique           │   p50    │   p90    │   p99    │")
    print("  ├────────────────────┼──────────┼──────────┼──────────┤")
    print(f"  │ TTFT (ms)          │ {t['p50']:>8.1f} │ {t['p90']:>8.1f} │ {t['p99']:>8.1f} │")
    print(f"  │ TPOT (ms)          │ {p['p50']:>8.1f} │ {p['p90']:>8.1f} │ {p['p99']:>8.1f} │")
    print(f"  │ End-to-end (ms)    │ {e['p50']:>8.1f} │ {e['p90']:>8.1f} │ {e['p99']:>8.1f} │")
    print("  └────────────────────┴──────────┴──────────┴──────────┘")
    print(f"\n  Débit séquentiel : {report['throughput_tps']:.2f} tok/s")
    print(f"  Tokens total     : {report['total_output_tokens']}")
    print(f"  Temps total      : {report['total_wall_s']:.1f} s")
    print()
    print("  ⚠  Référence L0 — plancher pour tous les lots suivants.")
    print("╚══════════════════════════════════════════════════════════════════╝\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(description="Velox L0 — baseline naïf séquentiel.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--info-only", action="store_true", help="Affiche les infos du modèle et quitte.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config = VeloxConfig(model_name=args.model, device=args.device, dtype=args.dtype)

    if args.info_only:
        runner = NaiveModelRunner(config)
        for k, v in runner.model_info().items():
            print(f"  {k:<30} {v}")
        return

    report = run_benchmark(
        config=config,
        prompts=DEFAULT_PROMPTS,
        max_new_tokens=args.max_tokens,
        warmup=args.warmup,
        use_chat_template=not args.no_chat_template,
    )

    print_report(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"  Rapport sauvegardé → {args.output}\n")


if __name__ == "__main__":
    main()
