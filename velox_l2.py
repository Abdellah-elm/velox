"""

L'idée centrale :
  L0/L1 : une séquence → un forward() par step.
  L2     : N séquences → UN forward() par step sur le batch entier.
  GPU (ou CPU) fait N fois plus de travail par appel,

  usage:
  python velox_l2.py --model gpt2 --device cpu --dtype float32 --no-chat-template --correctness-only
  python velox_l2.py --model gpt2 --device cpu --dtype float32 --no-chat-template --max-tokens 200
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
        VeloxConfig, GenerationResult, NaiveModelRunner,
        DEFAULT_PROMPTS, CORRECTNESS_PROMPTS, PercentileStats, print_report,
    )
    from velox_l1 import KVModelRunner
except ImportError as e:
    print(f"ERREUR : {e}")
    print("Mets velox_l0.py, velox_l1.py et velox_l2.py dans le même dossier.")
    sys.exit(1)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# StaticBatcher — L2
#
# QU'EST-CE QUE LE BATCHING STATIQUE ?
#
#   On regroupe N prompts dans un seul tenseur [N, max_prompt_len].
#   Un seul forward() traite les N séquences ensemble.
#   Les matrices de poids sont multipliées UNE FOIS pour N séquences
#   au lieu de N fois pour 1 séquence.
# ══════════════════════════════════════════════════════════════════════════════

class StaticBatcher:


    def __init__(self, config: VeloxConfig) -> None:
        self.config = config
        device = config.resolved_device
        logger.info("L2 StaticBatcher : chargement de %s sur %s…", config.model_name, device)

        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True,
        )


        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name, dtype=config.torch_dtype, trust_remote_code=True,
        ).to(device)
        self.model.eval()
        self._device = device

        logger.info(
            "Chargé : %.1fM params | %s | %s",
            sum(p.numel() for p in self.model.parameters()) / 1e6,
            device, config.dtype,
        )

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 128,
        greedy: bool = True,
        use_chat_template: bool = True,
    ) -> List[GenerationResult]:

        B = len(prompts)
        t_batch_start = time.perf_counter()


        formatted = [self._apply_chat_template(p, use_chat_template) for p in prompts]
        enc = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,  
            truncation=True,
            max_length=getattr(self.config, "max_seq_len", 1024),
        ).to(self._device)

        input_ids: torch.Tensor = enc["input_ids"]       
        attention_mask: torch.Tensor = enc["attention_mask"] 
        max_prompt_len: int = input_ids.shape[1]

        real_prompt_lens: torch.Tensor = attention_mask.sum(dim=1)  # [B]


        prefill_position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)  


        t_prefill = time.perf_counter()
        prefill_out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=prefill_position_ids,
            use_cache=True,
        )
        past_kv = prefill_out.past_key_values
        t_first_token = time.perf_counter()


        first_logits = prefill_out.logits[:, -1, :]
        next_tokens: torch.Tensor = self._sample_batch(first_logits, greedy)  # [B]

        ttft_ms = (t_first_token - t_prefill) * 1000.0


        finished = torch.zeros(B, dtype=torch.bool, device=self._device)
        generated: List[List[int]] = [[t.item()] for t in next_tokens]  # [B][step]

  
        decode_step_times: List[float] = []

        for step in range(max_new_tokens - 1):
            finished |= (next_tokens == self.tokenizer.eos_token_id)
            if finished.all():
                break

            t_step = time.perf_counter()


            decode_position_ids = (real_prompt_lens + step).unsqueeze(1) 


            new_col = (~finished).long().unsqueeze(1) 
            attention_mask = torch.cat([attention_mask, new_col], dim=1)


            feed_tokens = next_tokens.clone()
            feed_tokens[finished] = self.tokenizer.pad_token_id

            decode_out = self.model(
                input_ids=feed_tokens.unsqueeze(1),         
                attention_mask=attention_mask,            
                position_ids=decode_position_ids,       
                past_key_values=past_kv,
                use_cache=True,
            )

            decode_step_times.append((time.perf_counter() - t_step) * 1000.0)
            past_kv = decode_out.past_key_values

            logits = decode_out.logits[:, -1, :]            
            next_tokens = self._sample_batch(logits, greedy)  


            for i in range(B):
                if not finished[i]:
                    generated[i].append(next_tokens[i].item())

        t_batch_end = time.perf_counter()

        results: List[GenerationResult] = []
        for i in range(B):
            output_text = self.tokenizer.decode(generated[i], skip_special_tokens=True)
            output_tokens = len(generated[i])
 
            seq_total_ms = (t_batch_end - t_batch_start) * 1000.0
            tpot_ms = (sum(decode_step_times) / len(decode_step_times)) if decode_step_times else ttft_ms

            results.append(GenerationResult(
                prompt=prompts[i],
                output_text=output_text,
                prompt_tokens=int(real_prompt_lens[i].item()),
                output_tokens=output_tokens,
                ttft_ms=ttft_ms,
                tpot_ms=tpot_ms,
                total_ms=seq_total_ms,
                throughput_tps=output_tokens / max(seq_total_ms / 1000.0, 1e-9),
            ))

        return results

    def _sample_batch(self, logits: torch.Tensor, greedy: bool) -> torch.Tensor:
        """Retourne un tensor [B] de token IDs."""
        if greedy:
            return logits.argmax(dim=-1)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(1)

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

def check_correctness_batch(
    config: VeloxConfig,
    max_new_tokens: int = 40,
    use_chat_template: bool = True,
) -> bool:
    print("\n── Invariante de correction L1 vs L2 (batch) ──────────────────────")
    print(f"   {max_new_tokens} tokens | greedy | test avec prompts de longueurs variées\n")

    naive = NaiveModelRunner(config)
    batcher = StaticBatcher(config)


    test_prompts = [
        "Hi",                                     
        "What is machine learning?",               
        "Explain gradient descent in three sentences.", 
        "The transformer architecture uses self-attention to process sequences. Explain why.", 
    ]


    refs = [
        naive.generate(p, max_new_tokens=max_new_tokens, greedy=True,
                       use_chat_template=use_chat_template)
        for p in test_prompts
    ]


    batch_results = batcher.generate_batch(
        test_prompts, max_new_tokens=max_new_tokens,
        greedy=True, use_chat_template=use_chat_template,
    )

    all_ok = True
    for i, (ref, cand, prompt) in enumerate(zip(refs, batch_results, test_prompts)):
        if ref.output_text == cand.output_text:
            print(f"  ✓ [{i+1}/{len(test_prompts)}] {prompt[:50]!r}…")
        else:
            all_ok = False
            div = next(
                (j for j, (a, b) in enumerate(zip(ref.output_text, cand.output_text)) if a != b),
                min(len(ref.output_text), len(cand.output_text)),
            )
            print(f"  ✗ [{i+1}/{len(test_prompts)}] {prompt[:50]!r}…")
            print(f"    Diverge au char {div}")
            print(f"    solo  : …{ref.output_text[max(0,div-10):div+25]!r}…")
            print(f"    batch : …{cand.output_text[max(0,div-10):div+25]!r}…")

    print()
    if all_ok:
        print("  ✅ PASS — outputs identiques en solo et en batch.")
        print("     Les position_ids de padding sont corrects.\n")
    else:
        print("  ❌ FAIL — le batching change les outputs.")
        print("     Vérifier position_ids et padding_side.\n")

    return all_ok


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

def run_batch_sweep(
    config: VeloxConfig,
    max_new_tokens: int = 100,
    use_chat_template: bool = True,
) -> dict:

    prompts = DEFAULT_PROMPTS  
    batcher = StaticBatcher(config)

 
    logger.info("Warmup…")
    batcher.generate_batch(prompts[:2], max_new_tokens=20,
                           greedy=True, use_chat_template=use_chat_template)

    results_by_batch = {}

    print("\n  Sweep de batch size :")
    print(f"  {'Batch':>6}  {'Tokens/s':>10}  {'TTFT p50 ms':>12}  {'Gain vs B=1':>12}")
    print("  " + "─" * 46)

    tps_at_b1 = None

    for batch_size in [1, 2, 4, 8]:

        all_results: List[GenerationResult] = []
        t_start = time.perf_counter()

        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start:start + batch_size]
            batch_res = batcher.generate_batch(
                chunk, max_new_tokens=max_new_tokens,
                greedy=True, use_chat_template=use_chat_template,
            )
            all_results.extend(batch_res)

        total_time = time.perf_counter() - t_start
        total_tokens = sum(r.output_tokens for r in all_results)
        tps = total_tokens / total_time
        ttft_p50 = float(np.median([r.ttft_ms for r in all_results]))

        if batch_size == 1:
            tps_at_b1 = tps

        gain = tps / tps_at_b1 if tps_at_b1 else 1.0
        print(f"  {batch_size:>6}  {tps:>10.1f}  {ttft_p50:>12.1f}  {gain:>11.1f}×")

        results_by_batch[batch_size] = {
            "throughput_tps": round(tps, 2),
            "ttft_p50_ms": round(ttft_p50, 1),
            "gain_vs_b1": round(gain, 2),
        }

    print()
    print("    Limitation du batching STATIQUE (visible si prompts de longueurs variées) :")
    print("     Les séquences courtes attendent les longues → GPU/CPU idle inutilement.")
    print("     L3 (continuous batching) élimine cette attente.\n")

    return results_by_batch


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(description="Velox L2 — static batching benchmark.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max-tokens", type=int, default=100)
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--correctness-only", action="store_true")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config = VeloxConfig(model_name=args.model, device=args.device, dtype=args.dtype)


    ok = check_correctness_batch(
        config, max_new_tokens=40,
        use_chat_template=not args.no_chat_template,
    )

    if args.correctness_only:
        sys.exit(0 if ok else 1)

    if not ok:
        print("Correction échouée — benchmark annulé.")
        sys.exit(1)

    results = run_batch_sweep(
        config,
        max_new_tokens=args.max_tokens,
        use_chat_template=not args.no_chat_template,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"  Rapport → {args.output}")


if __name__ == "__main__":
    main()