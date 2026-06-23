"""
  # 1. Vérifier la correction d'abord (obligatoire avant tout benchmark)
  python velox_l1.py --model gpt2 --device cpu --dtype float32 --no-chat-template --correctness-only

  # 2. Benchmark comparatif L0 vs L1
  python velox_l1.py --model gpt2 --device cpu --dtype float32 --no-chat-template --max-tokens 200

  # 3. Voir l'effet du cache sur les longues séquences
  python velox_l1.py --model gpt2 --device cpu --dtype float32 --no-chat-template --max-tokens 500
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch


try:
    from velox_l0 import (
        VeloxConfig,
        GenerationResult,
        NaiveModelRunner,
        DEFAULT_PROMPTS,
        CORRECTNESS_PROMPTS,
        PercentileStats,
        run_benchmark,
        print_report,
    )
except ImportError:
    print("ERREUR : velox_l0.py introuvable dans le dossier courant.")
    print("Mets velox_l0.py et velox_l1.py dans le même dossier.")
    sys.exit(1)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# KVModelRunner — L1
#
# LA différence avec NaiveModelRunner :
#
#   L0 (naive) :
#     model.generate() recalcule l'attention sur TOUT l'historique à chaque pas.
#     Coût par pas = O(n²) où n = longueur totale de la séquence.
#
#   L1 (KV cache) :
#     On appelle model.forward() manuellement, step par step.
#     On passe past_key_values d'un step à l'autre.
#     Le modèle n'a plus besoin de recalculer les K/V des tokens précédents.
#     Coût par pas = O(1) — seulement le nouveau token.
#
# STRUCTURE EN 2 PHASES :
#
#   Phase 1 — PREFILL
#     forward(prompt_entier) → produit les logits + past_key_values initial
#     Ce forward est O(n²), fait UNE SEULE fois.
#     On mesure TTFT ici.
#
#   Phase 2 — DECODE LOOP
#     for each step:
#       forward(1_token, past_kv=cache) → logits + cache étendu
#     O(1) par step, indépendamment de la longueur de la séquence.
#     On mesure TPOT ici — doit rester PLAT même pour les longues séquences.
#
# PIÈGE PRINCIPAL — position_ids :
#   Qwen2.5 et LLaMA utilisent RoPE (Rotary Position Embedding).
#   Chaque token doit connaître SA position dans la séquence pour que
#   l'encodage rotationnel soit correct.
#   Sans position_ids explicite, certaines versions de transformers
#   utilisent position 0 pour chaque decode step → logits différents
#   → output différent de L0 → BUG silencieux.
#   GPT-2 utilise des positions absolues apprises et les infère depuis
#   past_key_values, donc ce piège est moins critique — mais on passe
#   position_ids quand même pour la portabilité vers Qwen2.5.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StepTiming:
    step: int
    context_len: int   
    step_ms: float     


class KVModelRunner:


    def __init__(self, config: VeloxConfig) -> None:
        self.config = config
        device = config.resolved_device
        logger.info("L1 KVModelRunner : chargement de %s sur %s…", config.model_name, device)

        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            dtype=config.torch_dtype,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()
        self._device = device
        logger.info("Chargé : %.1fM params", sum(p.numel() for p in self.model.parameters()) / 1e6)

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


        formatted = self._apply_chat_template(prompt, use_chat_template)
        enc = self.tokenizer(formatted, return_tensors="pt").to(self._device)
        input_ids: torch.Tensor = enc["input_ids"]           # [1, prompt_len]
        attention_mask: torch.Tensor = enc["attention_mask"] # [1, prompt_len]
        prompt_len: int = input_ids.shape[1]

        # ── PHASE 1 : PREFILL ─────────────────────────────────────────────────

        prefill_out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        past_kv = prefill_out.past_key_values

        first_logits = prefill_out.logits[:, -1, :]
        next_token_id = self._sample(first_logits, greedy, temperature, top_p)

        t_first_token = time.perf_counter()
        ttft_ms = (t_first_token - t_start) * 1000.0

        generated: List[int] = [next_token_id]
        step_timings: List[StepTiming] = []

        # ── PHASE 2 : DECODE LOOP ─────────────────────────────────────────────

        for step in range(max_new_tokens - 1):
            if next_token_id == self.tokenizer.eos_token_id:
                break

            t_step = time.perf_counter()

            current_position = prompt_len + step


            attention_mask = torch.cat([
                attention_mask,
                torch.ones(1, 1, device=self._device, dtype=attention_mask.dtype),
            ], dim=1) 

            decode_out = self.model(
                input_ids=torch.tensor([[next_token_id]], device=self._device),  # [1, 1]
                attention_mask=attention_mask,
                position_ids=torch.tensor([[current_position]], device=self._device),
                past_key_values=past_kv,
                use_cache=True,
            )

            step_ms = (time.perf_counter() - t_step) * 1000.0

            past_kv = decode_out.past_key_values

            next_token_id = self._sample(decode_out.logits[:, -1, :], greedy, temperature, top_p)
            generated.append(next_token_id)

            step_timings.append(StepTiming(
                step=step,
                context_len=prompt_len + len(generated),
                step_ms=step_ms,
            ))

        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000.0
        output_text = self.tokenizer.decode(generated, skip_special_tokens=True)
        output_tokens = len(generated)
        tpot_ms = (total_ms - ttft_ms) / max(output_tokens - 1, 1)

        result = GenerationResult(
            prompt=prompt,
            output_text=output_text,
            prompt_tokens=prompt_len,
            output_tokens=output_tokens,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            total_ms=total_ms,
            throughput_tps=output_tokens / max(total_ms / 1000.0, 1e-9),
        )

        result._step_timings = step_timings  
        return result



    def _sample(self, logits: torch.Tensor, greedy: bool,
                temperature: float = 1.0, top_p: float = 1.0) -> int:
        if greedy:
            return int(logits.argmax(dim=-1).item())
        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature
        if top_p < 1.0:
            probs = torch.softmax(logits, dim=-1)
            sorted_p, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumulative = torch.cumsum(sorted_p, dim=-1)
            to_remove = (cumulative - sorted_p) > top_p
            sorted_p[to_remove] = 0.0
            probs = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_p)
            return int(torch.multinomial(probs, 1).item())
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    def _apply_chat_template(self, prompt: str, use_chat_template: bool) -> str:
        if not use_chat_template or not getattr(self.tokenizer, "chat_template", None):
            return prompt
        try:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            return prompt


# ══════════════════════════════════════════════════════════════════════════════
# CORRECTNESS CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check_correctness(
    config: VeloxConfig,
    max_new_tokens: int = 50,
    use_chat_template: bool = True,
) -> bool:
    print("\n── Invariante de correction L0 vs L1 ──────────────────────────────")
    print(f"   Modèle : {config.model_name} | {max_new_tokens} tokens | greedy")
    print()

    naive = NaiveModelRunner(config)
    kv = KVModelRunner(config)
    all_ok = True

    for i, prompt in enumerate(CORRECTNESS_PROMPTS):
        ref = naive.generate(prompt, max_new_tokens=max_new_tokens,
                             greedy=True, use_chat_template=use_chat_template)
        cand = kv.generate(prompt, max_new_tokens=max_new_tokens,
                           greedy=True, use_chat_template=use_chat_template)

        if ref.output_text == cand.output_text:
            print(f"  ✓ [{i+1}/{len(CORRECTNESS_PROMPTS)}] {prompt[:55]!r}…")
        else:
            all_ok = False
            # Trouver le premier caractère divergent
            div = next((j for j, (a, b) in enumerate(zip(ref.output_text, cand.output_text))
                        if a != b), min(len(ref.output_text), len(cand.output_text)))
            print(f"  ✗ [{i+1}/{len(CORRECTNESS_PROMPTS)}] {prompt[:55]!r}…")
            print(f"    Diverge au caractère {div}")
            print(f"    L0 : …{ref.output_text[max(0,div-10):div+20]!r}…")
            print(f"    L1 : …{cand.output_text[max(0,div-10):div+20]!r}…")

    print()
    if all_ok:
        print("   PASS — les outputs L0 et L1 sont identiques (greedy).")
        print("     Les chiffres de perf L1 sont fiables.\n")
    else:
        print("   FAIL — bug dans la boucle de décodage L1.")
        print("     NE PAS utiliser les chiffres de perf L1.\n")

    return all_ok


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK COMPARATIF L0 vs L1
# ══════════════════════════════════════════════════════════════════════════════

def run_comparison(
    config: VeloxConfig,
    max_new_tokens: int = 200,
    warmup: int = 2,
    use_chat_template: bool = True,
) -> dict:

    prompts = DEFAULT_PROMPTS
    kv_runner = KVModelRunner(config)


    print("\nL0 (naïf, model.generate)…")
    report_l0 = run_benchmark(config, prompts, max_new_tokens=max_new_tokens,
                              warmup=warmup, use_chat_template=use_chat_template)
    print_report(report_l0)


    print("L1 (KV cache, boucle manuelle)…")
    for p in (prompts * 2)[:warmup]:
        kv_runner.generate(p, max_new_tokens=min(max_new_tokens, 32),
                           greedy=True, use_chat_template=use_chat_template)


    results_l1, t_l1_start = [], time.perf_counter()
    for i, p in enumerate(prompts):
        logger.info("L1 [%d/%d]…", i + 1, len(prompts))
        r = kv_runner.generate(p, max_new_tokens=max_new_tokens,
                               greedy=True, use_chat_template=use_chat_template)
        results_l1.append(r)
        logger.info("  → %d tok | TTFT %.0f ms | TPOT %.1f ms",
                    r.output_tokens, r.ttft_ms, r.tpot_ms)
    t_l1_total = time.perf_counter() - t_l1_start


    l1_ttft  = PercentileStats.from_list([r.ttft_ms  for r in results_l1])
    l1_tpot  = PercentileStats.from_list([r.tpot_ms  for r in results_l1])
    l1_e2e   = PercentileStats.from_list([r.total_ms for r in results_l1])
    l1_tps   = sum(r.output_tokens for r in results_l1) / t_l1_total

    l0_ttft = report_l0["ttft_ms"]["p50"]
    l0_tpot = report_l0["tpot_ms"]["p50"]
    l0_tps  = report_l0["throughput_tps"]

    tpot_speedup = l0_tpot / max(l1_tpot.p50, 0.001)

    print("\n╔══ L0 vs L1 — Résultats ═════════════════════════════════════════╗")
    print(f"  {'Métrique':<24} {'L0 Naïf':>10} {'L1 KV Cache':>12} {'Gain':>8}")
    print("  " + "─" * 56)
    print(f"  {'TTFT p50 (ms)':<24} {l0_ttft:>10.1f} {l1_ttft.p50:>12.1f} {'(prefill identique)':>8}")
    print(f"  {'TPOT p50 (ms)':<24} {l0_tpot:>10.1f} {l1_tpot.p50:>12.1f} {tpot_speedup:>7.1f}×")
    print(f"  {'Débit (tok/s)':<24} {l0_tps:>10.1f} {l1_tps:>12.1f} {l1_tps/max(l0_tps,0.001):>7.1f}×")
    print("╚" + "═" * 66 + "╝")

    print("\n  TPOT par longueur de contexte (L1 cache) :")
    print(f"  {'Contexte':>12}  {'TPOT médian (ms)':>18}  {'N steps':>8}")
    print("  " + "─" * 42)

    all_steps = [
        s for r in results_l1
        if hasattr(r, "_step_timings")
        for s in r._step_timings
    ]
    bins = [(0, 64), (64, 128), (128, 256), (256, 512), (512, 9999)]
    for lo, hi in bins:
        steps_in_bin = [s.step_ms for s in all_steps if lo <= s.context_len < hi]
        if not steps_in_bin:
            continue
        tpot_med = float(np.median(steps_in_bin))
        print(f"  {lo:>6}–{hi:<6}  {tpot_med:>18.1f}  {len(steps_in_bin):>8}")

    print()
    print("  👆 Si ces valeurs sont proches (±20%), le cache fonctionne.")
    print("     Si elles croissent → problème dans la boucle de décodage.\n")

    return {
        "l0": report_l0,
        "l1": {
            "throughput_tps": round(l1_tps, 2),
            "ttft_p50_ms": round(l1_ttft.p50, 1),
            "tpot_p50_ms": round(l1_tpot.p50, 1),
            "e2e_p50_ms":  round(l1_e2e.p50, 1),
            "tpot_speedup": round(tpot_speedup, 2),
            "num_requests": len(results_l1),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(description="Velox L1 — KV cache benchmark.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--correctness-only", action="store_true",
                   help="Vérifie uniquement la correction (pas de benchmark complet).")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config = VeloxConfig(model_name=args.model, device=args.device, dtype=args.dtype)


    ok = check_correctness(
        config=config,
        max_new_tokens=50,
        use_chat_template=not args.no_chat_template,
    )

    if args.correctness_only:
        sys.exit(0 if ok else 1)

    if not ok:
        print("Correction échouée — benchmark annulé.")
        sys.exit(1)


    results = run_comparison(
        config=config,
        max_new_tokens=args.max_tokens,
        warmup=args.warmup,
        use_chat_template=not args.no_chat_template,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"  Rapport → {args.output}")


if __name__ == "__main__":
    main()
