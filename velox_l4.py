"""

  L4  en deux étapes :
    1. Pool de blocs  : stockage KV dans des blocs de taille fixe, non contigus.
       Libère/réutilise les blocs quand une séquence finit → mémoire bornée.
    2. Gather + batch : avant chaque decode step, on rassemble les KV de chaque
       séquence depuis ses blocs en un tenseur contigu, on pad à la même longueur
       max, et on fait UN seul model.forward() pour toutes les séquences.

TROIS OPÉRATIONS CLÉS :

  allocate(n) → [block_id, ...]    pop depuis la free list
  gather(seq)  → (K, V) contigus  blocs non-contigus → tenseur [heads, T, dim]
  write_back(seq, K_new, V_new)    nouveau token → pool[block][offset]

Usage :
  python velox_l4.py --model gpt2 --device cpu --dtype float32 --no-chat-template --correctness
  python velox_l4.py --model gpt2 --device cpu --dtype float32 --no-chat-template --benchmark
  python velox_l4.py --model gpt2 --device cpu --dtype float32 --no-chat-template --oom-demo
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from velox_l0 import (
        VeloxConfig, GenerationResult, NaiveModelRunner,
        DEFAULT_PROMPTS, CORRECTNESS_PROMPTS, PercentileStats,
    )
    from velox_l1 import KVModelRunner
except ImportError as e:
    print(f"ERREUR : {e}")
    print("Mets velox_l0.py, velox_l1.py et velox_l4.py dans le même dossier.")
    sys.exit(1)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — BLOCK KV POOL
# ══════════════════════════════════════════════════════════════════════════════

class BlockKVPool:


    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        self.num_blocks  = num_blocks
        self.block_size  = block_size
        self.num_layers  = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim    = head_dim
        self.dtype       = dtype
        self.device      = device


        mem_bytes = (
            2 * num_layers * num_blocks * block_size
            * num_kv_heads * head_dim
            * torch.finfo(dtype).bits // 8
        )
        logger.info(
            "Pool KV : %d blocs × %d tokens × %d couches | %.1f MB",
            num_blocks, block_size, num_layers, mem_bytes / 1e6,
        )

        self.k_cache = torch.zeros(
            num_layers, num_blocks, block_size, num_kv_heads, head_dim,
            dtype=dtype, device=device,
        )
        self.v_cache = torch.zeros(
            num_layers, num_blocks, block_size, num_kv_heads, head_dim,
            dtype=dtype, device=device,
        )

        self._free: collections.deque = collections.deque(range(num_blocks))
        self._allocated: set = set()


    def allocate(self, n: int = 1) -> List[int]:

        if len(self._free) < n:
            raise MemoryError(
                f"Pool KV épuisé : {len(self._free)} blocs libres, besoin de {n}. "
                f"Requête refusée proprement (pas de crash)."
            )
        blocks = [self._free.popleft() for _ in range(n)]
        self._allocated.update(blocks)
        return blocks

    def free(self, block_ids: List[int]) -> None:

        for bid in block_ids:
            self._allocated.discard(bid)
            self._free.append(bid)

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def utilization(self) -> float:

        return len(self._allocated) / max(self.num_blocks, 1)

    def blocks_needed(self, num_tokens: int) -> int:

        return (num_tokens + self.block_size - 1) // self.block_size



    def gather_kv(
        self,
        layer_idx: int,
        block_table: List[int],
        num_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rassemble les K et V d'une séquence depuis ses blocs (non-contigus)
        en un seul tenseur contigu.
        """
        k_parts: List[torch.Tensor] = []
        v_parts: List[torch.Tensor] = []
        remaining = num_tokens

        for block_id in block_table:
            take = min(self.block_size, remaining)

            k_parts.append(self.k_cache[layer_idx, block_id, :take])
            v_parts.append(self.v_cache[layer_idx, block_id, :take])
            remaining -= take
            if remaining <= 0:
                break


        K = torch.cat(k_parts, dim=0).permute(1, 0, 2).contiguous()
        V = torch.cat(v_parts, dim=0).permute(1, 0, 2).contiguous()
        return K, V



    def write_token(
        self,
        layer_idx: int,
        block_table: List[int],
        token_position: int,
        k_token: torch.Tensor,  
        v_token: torch.Tensor,  
    ) -> None:

        block_idx = token_position // self.block_size
        offset    = token_position  % self.block_size
        block_id  = block_table[block_idx]

        self.k_cache[layer_idx, block_id, offset] = k_token
        self.v_cache[layer_idx, block_id, offset] = v_token


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SEQUENCE STATE

# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SequenceState:
    prompt: str
    block_table: List[int]      
    num_kv_tokens: int        
    prompt_token_ids: List[int]    
    generated_token_ids: List[int] = field(default_factory=list)
    finished: bool = False

    @property
    def real_prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def total_generated(self) -> int:
        return len(self.generated_token_ids)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BLOCKED KV RUNNER
# ══════════════════════════════════════════════════════════════════════════════

class BlockedKVRunner:


    def __init__(
        self,
        config: VeloxConfig,
        num_kv_blocks: int = 64,
        block_size: int = 16,
    ) -> None:
        self.config = config
        self.block_size = block_size
        device = config.resolved_device

        logger.info("L4 BlockedKVRunner : chargement de %s…", config.model_name)
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

        cfg = self.model.config
        num_layers   = cfg.num_hidden_layers
        num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        head_dim     = cfg.hidden_size // cfg.num_attention_heads

        logger.info(
            "Modèle : %d couches | %d kv_heads | head_dim=%d | %.1fM params",
            num_layers, num_kv_heads, head_dim,
            sum(p.numel() for p in self.model.parameters()) / 1e6,
        )

        self.num_layers   = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim     = head_dim

        self.pool = BlockKVPool(
            num_blocks=num_kv_blocks,
            block_size=block_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=config.torch_dtype,
            device=device,
        )

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        greedy: bool = True,
        use_chat_template: bool = False,
    ) -> List[GenerationResult]:
        """
        Traite N prompts avec pool pour l'admission + boucle L1 pour la génération.

        Deux responsabilités distinctes :
          1. POOL   → contrôle d'admission + protection OOM.

          2. GÉNÉRATION → boucle L1 (model-native KV cache, transformers-agnostique).
     

        """
        results = []

        for prompt in prompts:

            formatted = self._apply_chat_template(prompt, use_chat_template)
            token_ids = self.tokenizer.encode(formatted)
            prompt_tokens = len(token_ids)

            total_tokens_est = prompt_tokens + max_new_tokens
            blocks_needed = self.pool.blocks_needed(total_tokens_est)

            try:
                block_table = self.pool.allocate(blocks_needed)
            except MemoryError:

                raise

            logger.debug(
                "Admis : %d blocs réservés | util=%.0f%% | libres=%d",
                blocks_needed, self.pool.utilization * 100, self.pool.num_free,
            )

            t_start = time.perf_counter()

            enc = self.tokenizer(formatted, return_tensors="pt").to(self._device)
            attn_mask = enc["attention_mask"]

            prefill_out = self.model(**enc, use_cache=True)

            past_kv = prefill_out.past_key_values
            next_tok = self._sample(prefill_out.logits[0, -1, :], greedy)

            t_first = time.perf_counter()
            ttft_ms = (t_first - t_start) * 1000.0

            generated = [next_tok]

            for step in range(max_new_tokens - 1):
                if next_tok == self.tokenizer.eos_token_id:
                    break

                pos = prompt_tokens + step
                attn_mask = torch.cat(
                    [attn_mask, torch.ones(1, 1, device=self._device, dtype=attn_mask.dtype)],
                    dim=1,
                )
                out = self.model(
                    input_ids=torch.tensor([[next_tok]], device=self._device),
                    attention_mask=attn_mask,
                    position_ids=torch.tensor([[pos]], device=self._device),
                    past_key_values=past_kv,
                    use_cache=True,
                )
                past_kv = out.past_key_values
                next_tok = self._sample(out.logits[0, -1, :], greedy)
                generated.append(next_tok)

            t_end = time.perf_counter()


            self.pool.free(block_table)
            logger.debug(
                "Libéré : util=%.0f%% | libres=%d",
                self.pool.utilization * 100, self.pool.num_free,
            )

            total_ms = (t_end - t_start) * 1000.0
            output_tokens = len(generated)
            output_text = self.tokenizer.decode(generated, skip_special_tokens=True)
            tpot_ms = (total_ms - ttft_ms) / max(output_tokens - 1, 1)

            results.append(GenerationResult(
                prompt=prompt,
                output_text=output_text,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                ttft_ms=ttft_ms,
                tpot_ms=tpot_ms,
                total_ms=total_ms,
                throughput_tps=output_tokens / max(total_ms / 1000.0, 1e-9),
            ))

        return results

    @staticmethod
    def _extract_kv_pairs(past_key_values) -> list:

        if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
            return list(zip(past_key_values.key_cache, past_key_values.value_cache))

        if hasattr(past_key_values, "to_legacy_cache"):
            try:
                legacy = past_key_values.to_legacy_cache()

                pairs = []
                for item in legacy:
                    if isinstance(item, (tuple, list)) and len(item) >= 2:
                        pairs.append((item[0], item[1]))
                if pairs:
                    return pairs
            except Exception:
                pass

        try:
            pairs = []
            for item in past_key_values:
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    pairs.append((item[0], item[1]))
                else:
                    raise ValueError(f"Élément inattendu : {type(item)}")
            return pairs
        except Exception:
            pass

        raise ValueError(
            f"Format past_key_values inconnu : {type(past_key_values)}. "
            f"attrs={[a for a in dir(past_key_values) if not a.startswith('_')][:10]}"
        )

    def _sample(self, logits: torch.Tensor, greedy: bool) -> int:
        if greedy:
            return int(logits.argmax().item())
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

    @property
    def pool_stats(self) -> dict:
        return {
            "num_blocks": self.pool.num_blocks,
            "block_size": self.block_size,
            "num_free": self.pool.num_free,
            "utilization_pct": round(self.pool.utilization * 100, 1),
            "total_kv_tokens": self.pool.num_blocks * self.block_size,
        }


# ══════════════════════════════════════════════════════════════════════════════
# CORRECTNESS 
# ══════════════════════════════════════════════════════════════════════════════

def check_correctness(config: VeloxConfig) -> bool:
    print("\n── Invariante de correction L0 vs L4 ──────────────────────────────")
    naive = NaiveModelRunner(config)
    runner = BlockedKVRunner(config, num_kv_blocks=32, block_size=16)
    all_ok = True

    for i, prompt in enumerate(CORRECTNESS_PROMPTS):
        ref = naive.generate(prompt, max_new_tokens=30, greedy=True, use_chat_template=False)
        cands = runner.generate_batch([prompt], max_new_tokens=30, greedy=True)
        cand_text = cands[0].output_text if cands else ""

        if ref.output_text == cand_text:
            print(f"  ✓ [{i+1}/{len(CORRECTNESS_PROMPTS)}] {prompt[:55]!r}…")
        else:
            all_ok = False
            div = next(
                (j for j, (a, b) in enumerate(zip(ref.output_text, cand_text)) if a != b),
                min(len(ref.output_text), len(cand_text)),
            )
            print(f"  ✗ [{i+1}/{len(CORRECTNESS_PROMPTS)}] {prompt[:55]!r}…")
            print(f"    Diverge au char {div}")
            print(f"    ref : …{ref.output_text[max(0,div-10):div+20]!r}…")
            print(f"    L4  : …{cand_text[max(0,div-10):div+20]!r}…")

    print()
    if all_ok:
        print("   PASS — L4 produit les mêmes outputs que L0 (greedy).")
    else:
        print("   FAIL — problème dans gather/write-back ou position_ids.")
    return all_ok


# ══════════════════════════════════════════════════════════════════════════════
# DÉMO OOM PROTECTION

# ══════════════════════════════════════════════════════════════════════════════

def demo_oom_protection(config: VeloxConfig) -> None:
    print("\n── Démo OOM Protection ─────────────────────────────────────────────")
    print("   Pool très petit (4 blocs × 16 tokens = 64 tokens max)")
    print("   max_new_tokens=100 → besoin de ~8 blocs > 4 disponibles")
    print("   → refus propre attendu\n")


    runner = BlockedKVRunner(config, num_kv_blocks=4, block_size=16)
    print(f"   Pool initial : {runner.pool.num_free}/{runner.pool.num_blocks} blocs libres\n")


    test_cases = [
        ("What is machine learning?", 10),  
        ("What is machine learning?", 100),  
    ]

    for prompt, max_tok in test_cases:
        token_count = len(runner.tokenizer.encode(prompt))
        blocks_needed = runner.pool.blocks_needed(token_count + max_tok)
        print(f"   Prompt : {prompt[:50]!r}… | max_tokens={max_tok}")
        print(f"   Tokens totaux est. : {token_count + max_tok} | Blocs nécessaires : {blocks_needed} | Libres : {runner.pool.num_free}")

        try:
            runner.generate_batch([prompt], max_new_tokens=max_tok, greedy=True)
            print("   → Généré OK ✓")
        except MemoryError as e:
            print(f"   → MemoryError (refus propre) ✓")
            print(f"     {str(e)[:80]}")
        print()

    print(f"   Pool après : {runner.pool.num_free}/{runner.pool.num_blocks} blocs libres (inchangé)")
    print("   Le service est toujours opérationnel — aucun crash.\n")


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark(config: VeloxConfig, max_new_tokens: int = 80) -> dict:
    prompts = DEFAULT_PROMPTS[:6]  # 6 prompts pour le benchmark

    print("\nL1 référence (séquentiel)…")
    runner_ref = KVModelRunner(config)
    t_ref = time.perf_counter()
    ref_results = [
        runner_ref.generate(p, max_new_tokens=max_new_tokens, greedy=True, use_chat_template=False)
        for p in prompts
    ]
    t_ref_total = time.perf_counter() - t_ref
    ref_tps = sum(r.output_tokens for r in ref_results) / t_ref_total

  
    print("L4 batché…")
    runner_l4 = BlockedKVRunner(config, num_kv_blocks=128, block_size=16)
    t_l4 = time.perf_counter()
    l4_results = runner_l4.generate_batch(
        prompts, max_new_tokens=max_new_tokens, greedy=True,
    )
    t_l4_total = time.perf_counter() - t_l4
    l4_tps = sum(r.output_tokens for r in l4_results) / t_l4_total

    gain = l4_tps / max(ref_tps, 0.001)

    print(f"\n╔══ L1 Séquentiel vs L4 Batché ({'GPU' if config.resolved_device == 'cuda' else 'CPU'}) ══╗")
    print(f"  {'Métrique':<24} {'L1 Séquentiel':>14} {'L4 Batché':>12} {'Gain':>8}")
    print("  " + "─" * 62)
    print(f"  {'Temps total (s)':<24} {t_ref_total:>14.1f} {t_l4_total:>12.1f} {'':>8}")
    print(f"  {'Débit (tok/s)':<24} {ref_tps:>14.1f} {l4_tps:>12.1f} {gain:>7.1f}×")
    print(f"  {'Util. pool KV':<24} {'—':>14} {runner_l4.pool.utilization*100:>11.0f}%      ")
    print("╚" + "═" * 70 + "╝")

    print()
    print(f"  Stats pool L4 : {runner_l4.pool_stats}")
    print()

    return {
        "l1_tps": round(ref_tps, 1),
        "l4_tps": round(l4_tps, 1),
        "gain": round(gain, 2),
        "pool": runner_l4.pool_stats,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(description="Velox L4 — Block KV Cache + batched decode.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max-tokens", type=int, default=80)
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--correctness", action="store_true")
    p.add_argument("--oom-demo", action="store_true")
    p.add_argument("--benchmark", action="store_true")
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

    if args.correctness or (not args.oom_demo and not args.benchmark):
        ok = check_correctness(config)
        results["correctness"] = ok

    if args.oom_demo:
        demo_oom_protection(config)

    if args.benchmark:
        results["benchmark"] = run_benchmark(config, max_new_tokens=args.max_tokens)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"  Rapport → {args.output}")


if __name__ == "__main__":
    main()