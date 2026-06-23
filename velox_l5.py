"""
velox_l5.py — L5 : API REST compatible OpenAI + streaming SSE.

ENDPOINTS :
  POST /v1/chat/completions    ← principal (chat)
  POST /v1/completions         ← completion brute
  GET  /v1/models              ← liste des modèles disponibles
  GET  /health                 ← état du serveur (monitoring)

Installation :
  pip install fastapi uvicorn[standard]

Usage :
   Démarrer le serveur
  python velox_l5.py --serve --model gpt2 --device cpu --dtype float32 --no-chat-template

   Tester sans HTTP (génération directe)
  python velox_l5.py --test-local --model gpt2 --device cpu --dtype float32 --no-chat-template

   Test curl (serveur démarré)
  curl http://localhost:8000/health
  curl -X POST http://localhost:8000/v1/chat/completions \\
       -H "Content-Type: application/json" \\
       -d '{"model":"gpt2","messages":[{"role":"user","content":"Hi"}],"max_tokens":20}'
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import queue
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator, List, Literal, Optional, Union

import torch

try:
    from velox_l0 import VeloxConfig, GenerationResult
    from velox_l1 import KVModelRunner
except ImportError as e:
    print(f"ERREUR : {e}")
    print("Mets velox_l0.py, velox_l1.py et velox_l5.py dans le même dossier.")
    sys.exit(1)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SCHÉMAS OPENAI-COMPATIBLES (Pydantic v2)
# ══════════════════════════════════════════════════════════════════════════════

try:
    from pydantic import BaseModel, Field
except ImportError:
    print("ERREUR : pip install pydantic")
    sys.exit(1)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length"]] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: UsageInfo


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: Optional[Literal["stop", "length"]] = "stop"


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DÉCODAGE PAS À PAS (pour le streaming)
# ══════════════════════════════════════════════════════════════════════════════

def _decode_step_by_step(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    greedy: bool,
    device: str,
    cancel_event: threading.Event,
):

    enc = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = enc["input_ids"].shape[1]
    attn_mask = enc["attention_mask"]

  
    with torch.inference_mode():
        prefill_out = model(**enc, use_cache=True)
    past_kv = prefill_out.past_key_values
    next_tok = _argmax(prefill_out.logits[0, -1, :]) if greedy else _sample(prefill_out.logits[0, -1, :])

    yield next_tok, (next_tok == tokenizer.eos_token_id)

  
    for step in range(max_new_tokens - 1):
        if cancel_event.is_set():
            return
        if next_tok == tokenizer.eos_token_id:
            return

        pos = prompt_len + step
        attn_mask = torch.cat(
            [attn_mask, torch.ones(1, 1, device=device, dtype=attn_mask.dtype)], dim=1
        )
        with torch.inference_mode():
            out = model(
                input_ids=torch.tensor([[next_tok]], device=device),
                attention_mask=attn_mask,
                position_ids=torch.tensor([[pos]], device=device),
                past_key_values=past_kv,
                use_cache=True,
            )
        past_kv = out.past_key_values
        next_tok = _argmax(out.logits[0, -1, :]) if greedy else _sample(out.logits[0, -1, :])
        is_last = (next_tok == tokenizer.eos_token_id or step == max_new_tokens - 2)
        yield next_tok, is_last


def _argmax(logits: torch.Tensor) -> int:
    return int(logits.argmax().item())


def _sample(logits: torch.Tensor) -> int:
    return int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ÉTAT GLOBAL DU SERVEUR
# ══════════════════════════════════════════════════════════════════════════════

class ServerState:
    """Singleton contenant le modèle et le tokeniseur chargés."""

    def __init__(self):
        self.runner: Optional[KVModelRunner] = None
        self.config: Optional[VeloxConfig] = None
        self.model = None
        self.tokenizer = None
        self.use_chat_template: bool = True
        self._ready: bool = False

    def load(self, config: VeloxConfig, use_chat_template: bool = True) -> None:
        self.config = config
        self.use_chat_template = use_chat_template
        logger.info("Chargement du modèle pour L5…")
        self.runner = KVModelRunner(config)
        self.model = self.runner.model
        self.tokenizer = self.runner.tokenizer
        self._ready = True
        logger.info("Modèle prêt.")

    @property
    def ready(self) -> bool:
        return self._ready

    def format_prompt(self, messages: List[ChatMessage]) -> str:
        """Applique le chat template si dispo, sinon concatène."""
        if self.use_chat_template and getattr(self.tokenizer, "chat_template", None):
            try:
                return self.tokenizer.apply_chat_template(
                    [{"role": m.role, "content": m.content} for m in messages],
                    tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                pass
        return "\n".join(f"{m.role}: {m.content}" for m in messages)


_state = ServerState()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — APPLICATION FASTAPI
# ══════════════════════════════════════════════════════════════════════════════

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError:
    print("ERREUR : pip install fastapi uvicorn[standard]")
    sys.exit(1)

app = FastAPI(
    title="Velox Inference Engine",
    description="OpenAI-compatible LLM inference API (L5)",
    version="0.5.0",
)


@app.get("/health")
async def health():
    """Healthcheck pour Docker / GuardRAG / monitoring."""
    return {
        "status": "ok" if _state.ready else "loading",
        "model": _state.config.model_name if _state.config else None,
        "device": _state.config.resolved_device if _state.config else None,
        "timestamp": int(time.time()),
    }


@app.get("/v1/models")
async def list_models():
    """Endpoint /v1/models — requis par la plupart des clients OpenAI."""
    if not _state.ready:
        raise HTTPException(503, "Modèle non chargé")
    model_id = _state.config.model_name
    return {
        "object": "list",
        "data": [{
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "velox",
        }],
    }


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):

    if not _state.ready:
        raise HTTPException(503, "Modèle non chargé. Démarrez avec --serve.")

    prompt = _state.format_prompt(body.messages)

    if body.stream:
        return StreamingResponse(
            _stream_chat(body, prompt, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    loop = asyncio.get_event_loop()
    result: GenerationResult = await loop.run_in_executor(
        None,
        lambda: _state.runner.generate(
            prompt,
            max_new_tokens=body.max_tokens,
            greedy=body.greedy,
            seed=body.seed,
            use_chat_template=False,  
        ),
    )

    resp_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    return ChatCompletionResponse(
        id=resp_id,
        created=int(time.time()),
        model=body.model,
        choices=[ChatChoice(
            message=ChatMessage(role="assistant", content=result.output_text),
            finish_reason="stop",
        )],
        usage=UsageInfo(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.output_tokens,
            total_tokens=result.prompt_tokens + result.output_tokens,
        ),
    )



@app.post("/v1/completions")
async def completions(body: CompletionRequest, request: Request):
    """Endpoint de complétion brute (texte → texte)."""
    if not _state.ready:
        raise HTTPException(503, "Modèle non chargé.")

    prompt = body.prompt if isinstance(body.prompt, str) else body.prompt[0]

    if body.stream:
        return StreamingResponse(
            _stream_completion(body, prompt, request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: _state.runner.generate(
            prompt, max_new_tokens=body.max_tokens, greedy=body.greedy,
            seed=body.seed, use_chat_template=False,
        ),
    )

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=body.model,
        choices=[CompletionChoice(text=result.output_text, finish_reason="stop")],
        usage=UsageInfo(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.output_tokens,
            total_tokens=result.prompt_tokens + result.output_tokens,
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — GÉNÉRATEURS SSE
# ══════════════════════════════════════════════════════════════════════════════

async def _stream_chat(
    body: ChatCompletionRequest,
    prompt: str,
    request: Request,
) -> AsyncGenerator[str, None]:
    """Générateur SSE pour /v1/chat/completions?stream=true."""
    resp_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    cancel = threading.Event()
    token_queue: asyncio.Queue = asyncio.Queue(maxsize=64)
    loop = asyncio.get_event_loop()


    first_chunk = ChatCompletionChunk(
        id=resp_id, created=created, model=body.model,
        choices=[StreamChoice(delta=DeltaMessage(role="assistant"), finish_reason=None)],
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"


    def _worker():
        try:
            gen = _decode_step_by_step(
                model=_state.model,
                tokenizer=_state.tokenizer,
                prompt=prompt,
                max_new_tokens=body.max_tokens,
                greedy=body.greedy,
                device=_state.config.resolved_device,
                cancel_event=cancel,
            )
            for tok_id, is_last in gen:
                if cancel.is_set():
                    break
                asyncio.run_coroutine_threadsafe(
                    token_queue.put((tok_id, is_last)), loop
                )
        except Exception as e:
            asyncio.run_coroutine_threadsafe(
                token_queue.put(("error", str(e))), loop
            )
        finally:
            asyncio.run_coroutine_threadsafe(token_queue.put(None), loop)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()


    try:
        while True:

            if await request.is_disconnected():
                cancel.set()
                logger.info("Client déconnecté — génération annulée.")
                return

            try:
                item = await asyncio.wait_for(token_queue.get(), timeout=0.05)
            except asyncio.TimeoutError:
                continue

            if item is None:

                final_chunk = ChatCompletionChunk(
                    id=resp_id, created=created, model=body.model,
                    choices=[StreamChoice(delta=DeltaMessage(), finish_reason="stop")],
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                return

            if isinstance(item, tuple) and item[0] == "error":
                logger.error("Erreur de génération : %s", item[1])
                return

            tok_id, is_last = item
            text = _state.tokenizer.decode([tok_id], skip_special_tokens=False)
            chunk = ChatCompletionChunk(
                id=resp_id, created=created, model=body.model,
                choices=[StreamChoice(
                    delta=DeltaMessage(content=text),
                    finish_reason="stop" if is_last else None,
                )],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

    finally:
        cancel.set()


async def _stream_completion(
    body: CompletionRequest,
    prompt: str,
    request: Request,
) -> AsyncGenerator[str, None]:
    """Générateur SSE pour /v1/completions?stream=true."""
    resp_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    cancel = threading.Event()
    token_queue: asyncio.Queue = asyncio.Queue(maxsize=64)
    loop = asyncio.get_event_loop()

    def _worker():
        try:
            for tok_id, is_last in _decode_step_by_step(
                _state.model, _state.tokenizer, prompt,
                body.max_tokens, body.greedy,
                _state.config.resolved_device, cancel,
            ):
                if cancel.is_set():
                    break
                asyncio.run_coroutine_threadsafe(
                    token_queue.put((tok_id, is_last)), loop
                )
        finally:
            asyncio.run_coroutine_threadsafe(token_queue.put(None), loop)

    threading.Thread(target=_worker, daemon=True).start()

    try:
        while True:
            if await request.is_disconnected():
                cancel.set()
                return
            try:
                item = await asyncio.wait_for(token_queue.get(), timeout=0.05)
            except asyncio.TimeoutError:
                continue
            if item is None:
                yield "data: [DONE]\n\n"
                return
            tok_id, is_last = item
            text = _state.tokenizer.decode([tok_id], skip_special_tokens=False)
            chunk = {
                "id": resp_id, "object": "text_completion", "created": created,
                "model": body.model,
                "choices": [{"text": text, "index": 0,
                             "finish_reason": "stop" if is_last else None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
    finally:
        cancel.set()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — TEST LOCAL (sans HTTP)
# ══════════════════════════════════════════════════════════════════════════════

def test_local(config: VeloxConfig, use_chat_template: bool = True) -> None:
    """Test de génération locale — valide le pipeline sans HTTP."""
    _state.load(config, use_chat_template)

    print("\n── Test non-streaming ───────────────────────────────────────────────")
    prompt = "What is machine learning?"
    result = _state.runner.generate(
        prompt, max_new_tokens=40, greedy=True, use_chat_template=False,
    )
    print(f"  Prompt : {prompt!r}")
    print(f"  Output : {result.output_text!r}")
    print(f"  TTFT   : {result.ttft_ms:.0f} ms | TPOT : {result.tpot_ms:.1f} ms\n")

    print("── Test streaming (tokens un par un) ───────────────────────────────")
    cancel = threading.Event()
    print(f"  Prompt : {prompt!r}")
    print("  Output : ", end="", flush=True)
    generated = []
    for tok_id, is_last in _decode_step_by_step(
        _state.model, _state.tokenizer, prompt, 40, True,
        config.resolved_device, cancel,
    ):
        text = _state.tokenizer.decode([tok_id], skip_special_tokens=False)
        print(text, end="", flush=True)
        generated.append(tok_id)
        if is_last:
            break
    print(f"\n  Tokens : {len(generated)}\n")

    print("── Test /health (simulé) ───────────────────────────────────────────")
    print(f"  status : ok")
    print(f"  model  : {config.model_name}")
    print(f"  device : {config.resolved_device}\n")

    print("✅ Tests locaux OK — serveur prêt à démarrer avec --serve\n")
    print("  Commande GuardRAG :")
    print("    OPENAI_API_BASE=http://localhost:8000/v1")
    print("    OPENAI_API_KEY=velox-local")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(description="Velox L5 — API REST OpenAI-compatible.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--serve", action="store_true",
                   help="Démarrer le serveur FastAPI.")
    p.add_argument("--test-local", action="store_true",
                   help="Tester la génération sans HTTP.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config = VeloxConfig(model_name=args.model, device=args.device, dtype=args.dtype)
    use_chat = not args.no_chat_template

    if args.test_local:
        test_local(config, use_chat)
        return

    if args.serve:
        try:
            import uvicorn
        except ImportError:
            print("ERREUR : pip install uvicorn[standard]")
            sys.exit(1)


        _state.load(config, use_chat)

        print(f"\n  Velox L5 — serveur démarré")
        print(f"  Modèle : {config.model_name} | {config.resolved_device}")
        print(f"  URL    : http://{args.host}:{args.port}")
        print(f"  Docs   : http://{args.host}:{args.port}/docs")
        print()
        print("  Test rapide :")
        print(f"    curl http://localhost:{args.port}/health")
        print(f'    curl -X POST http://localhost:{args.port}/v1/chat/completions \\')
        print(f'         -H "Content-Type: application/json" \\')
        print(f"         -d '{{\"model\":\"{config.model_name}\",\"messages\":[{{\"role\":\"user\",\"content\":\"Hi\"}}],\"max_tokens\":30}}'")
        print()

        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
        return

    p.print_help()


if __name__ == "__main__":
    main()
