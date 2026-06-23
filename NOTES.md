# Engineering Notes — Velox

---

## What this project is

A from-scratch LLM inference engine. Not a wrapper around vLLM. Not a tutorial reimplementation. A system built mechanism by mechanism, measured at each step, with every limitation documented.

The question it answers: *what actually happens between a user request and the first token, and what does it take to serve that efficiently under concurrent load?*

---

## What was built

### L0 — Naive baseline (reference floor)

`model.generate()` called sequentially, one request at a time. This is the "dumb" reference intentionally unoptimized.

Results on T4 (Qwen2.5-0.5B-Instruct, float16, 64 tokens):
- TTFT p50: 34ms
- TPOT p50: 29ms
- Throughput: 34 tok/s

This is the floor. Every subsequent lot is compared against it.

### L1 — KV cache

Replaced `model.generate()` with a manual step-by-step decode loop threading `past_key_values` across steps. The loop runs: prefill → sample first token → decode loop (one `model.forward()` per step with cached K/V).

**Correctness invariant (the most important test in the project):** greedy output of KVModelRunner must be character-for-character identical to NaiveModelRunner. If they diverge, the decode loop has a bug — subtle issues with `position_ids` (critical for RoPE), `attention_mask` shape, or logit indexing.

Results: TPOT flat across context lengths on T4 (KV cache working). Gain vs L0: 1.0× on GPU short sequences — `model.generate()` already uses the CUDA KV cache internally. Real gain visible on CPU or sequences >200 tokens.

**Lesson:** on GPU, the bottleneck is memory bandwidth, not attention recomputation on short sequences. The KV cache matters most for long-context generation.

### L2 — Static batching

Grouped N prompts into a single batch tensor. One `model.forward()` per decode step for all sequences simultaneously.

Key engineering detail: **left-padding**. For decoder-only models (GPT-2, Qwen2.5), padding must go on the left so the last real token is always at position `-1`. This makes `logits[:, -1, :]` correct for all sequences uniformly. Right-padding requires per-sequence indexing and is a source of bugs.

**position_ids correction:** with left padding, `position_ids = attention_mask.cumsum(-1) - 1`. Without this, GPT-2 assigns wrong absolute positions to real tokens after padding → different logits → correctness failure.

Results on T4:
- batch=1: 34 tok/s
- batch=4: **146 tok/s (×4.3)**
- batch=8: 133 tok/s (slight drop — T4 memory bandwidth saturation)

### L3 — Continuous batching + scheduler

Static batching forces all sequences in a batch to finish together. A short sequence waits for a long one — idle compute.

Continuous batching: sequences enter and leave the batch step by step. A slot freed by a finished sequence is immediately filled by a waiting request.

Architecture: **producer-consumer with two loops.**
- I/O loop (asyncio): handles HTTP requests, tokenization, SSE streaming.
- Compute loop (dedicated thread): runs `forward()`, samples tokens, manages the scheduler.
- Bridge: `asyncio.Queue` + `threading.Event` per request.

PyTorch releases the GIL during ATen ops — a thread is sufficient to let asyncio breathe. Using `asyncio.run_in_executor()` with a `ThreadPoolExecutor(max_workers=1)` achieves the same.

**Limitation without L4:** without a block-based KV cache, sequences at different decode positions have KV tensors of different lengths. The `DynamicCache` API in transformers 5.x changed incompatibly — manually constructing or injecting into it is not stable. In this implementation, the scheduling is correct but the compute remains sequential per slot. True batched forward per step requires L4's block pool + a stable attention API (vLLM solves this with custom CUDA kernels).

Scheduler policy: **FCFS** (First Come First Served). SJF/SRPT would reduce average latency but require predicting output length at admission time — which is impossible without a separate length predictor. Reference: ORCA (Yu et al., OSDI '22) — the academic source of iteration-level scheduling / continuous batching.

### L4 — Block KV cache pool

Pre-allocates a fixed tensor pool at startup:
```
k_cache: [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
v_cache: [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
```

Free list for allocation/deallocation. Block table per sequence (list of block IDs).

**What L4 delivers:**
- Bounded memory: pool has a fixed size, no dynamic growth.
- OOM protection: `MemoryError` on pool exhaustion → HTTP 503 instead of process crash.
- Foundation for prefix caching (blocks can be shared between requests with common prefixes).

**What L4 doesn't deliver in PyTorch pure:**
The gather operation (non-contiguous blocks → contiguous tensor for `forward()`) involves memory copies. Without a custom CUDA kernel (like vLLM's Triton implementation of PagedAttention), there is no throughput gain from the block layout — only the memory management benefit. This is documented in the spec (§8.3) and the benchmark notes.

**transformers 5.x compatibility issue:** `DynamicCache` in transformers 5.x removed the `key_cache`/`value_cache` list attributes and changed the injection API. Manually pre-populating a cache for batched decode is not stable across versions. The block pool demonstrates the correct memory management semantics; the batched forward is a production concern that requires either a stable API or a custom kernel.

### L5 — OpenAI-compatible API

FastAPI server exposing:
- `POST /v1/chat/completions` (streaming + non-streaming)
- `POST /v1/completions`
- `GET /v1/models`
- `GET /health`

SSE streaming format exactly matches OpenAI's wire format — `role` delta first, then `content` deltas, then `finish_reason`, then `[DONE]`.

Client disconnect (EF-13): checked via `await request.is_disconnected()` at each token. On disconnect, `cancel_event.set()` stops the worker thread immediately. No compute wasted on abandoned generations.

### L6 — Prometheus observability

Middleware-based instrumentation (not route override — FastAPI resolves routes in registration order, so overriding the same path doesn't work as expected).

Metrics:
- `velox_requests_total{endpoint, status}` — request count by outcome
- `velox_tokens_generated_total` — output tokens (capacity planning)
- `velox_ttft_seconds` — TTFT histogram (user-perceived latency)
- `velox_tpot_seconds` — TPOT histogram (streaming quality)
- `velox_request_duration_seconds` — E2E latency (SLO)
- `velox_active_requests` — in-flight count (saturation indicator)

TTFT/TPOT measured via `runner.generate()` monkey-patch at startup — cleaner than instrumenting each endpoint separately.

### L7 — Benchmark vs vLLM

Run on Colab T4 (Qwen2.5-0.5B-Instruct, float16, 64 tokens):

| Runner | tok/s | TTFT p50 (ms) | TPOT p50 (ms) |
|--------|-------|--------------|--------------|
| L0 Naive | 33.9 | 33.5 | 29.4 |
| L1 KV Cache | 34.9 | 31.8 | 28.5 |
| L2 Batch=4 | **146.2** | 39.9 | 27.1 |
| L2 Batch=8 | 133.1 | 32.7 | 30.0 |
| vLLM (published) | ~400–500 | — | — |

**H1 falsification:** KV cache gain = 1.0× on T4 short sequences. Hypothesis partially falsified for this regime. On CPU or long sequences (>200 tokens), the quadratic recomputation cost becomes visible and the cache provides a real benefit.

**H2 confirmation:** batching ×4.3 at batch=4. The throughput curve (rise then plateau) matches vLLM's shape — the correct mechanism is reproduced even if the absolute value differs.

**vLLM gap (~3×):** documented. The gap is explained by custom CUDA kernels (PagedAttention, Flash Attention) that Velox doesn't implement. Publishing this gap is more credible than hiding it.

### L8 — GuardRAG integration

GuardRAG uses the Groq SDK directly (`groq.chat.completions.create()`). The OpenAI Python SDK exposes an identical interface. 4-line change in `main.py` + 3 `.env` variables:

```python
USE_VELOX = os.getenv("USE_VELOX", "false").lower() == "true"
if USE_VELOX:
    from openai import OpenAI as Groq
    groq = Groq(base_url=..., api_key="velox-local")
    FAST_MODEL = STRONG_MODEL = os.getenv("VELOX_MODEL")
else:
    from groq import Groq
```

`from openai import OpenAI as Groq` — the alias means zero changes elsewhere in GuardRAG. Every call to `groq.chat.completions.create()` routes to Velox transparently.

Pipeline validated end-to-end: query → PII redaction → embedding → retrieval → `generate_answer(Velox)` → `faithfulness_check(Velox)` → response. Both LLM calls (answer generation + judge) route through Velox.

---

## What I would do differently

1. **L4 before L3.** The true benefit of continuous batching (batched decode per step) requires the block KV pool. Implementing L3 before L4 shows the scheduling architecture but can't demonstrate the full throughput gain.

2. **Target transformers version explicitly.** The `DynamicCache` API changed in transformers 5.x. Pinning to transformers 4.44 would have avoided the compatibility issues in L4.

3. **GPU from the start.** On CPU, the KV cache and batching gains are masked by memory bandwidth limits. All benchmarks should run on T4; CPU is only for portability smoke tests.

4. **Single model family.** GPT-2 (absolute learned positions) and Qwen2.5 (RoPE + GQA) have fundamentally different attention implementations. Debugging correctness issues across two architectures wasted time. Pick one and go deep.

---


*Velox v0.6.0 · June 2026*
