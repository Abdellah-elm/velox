"""
velox_l6.py — L6 : Observabilité Prometheus.

  velox_requests_total{status}         counter   requêtes par statut (ok/error/streaming)
  velox_tokens_generated_total         counter   tokens de sortie générés
  velox_ttft_seconds                   histogram temps jusqu'au premier token
  velox_tpot_seconds                   histogram temps par token de sortie
  velox_request_duration_seconds       histogram latence E2E
  velox_active_requests                gauge     requêtes en cours

Usage :

  python velox_l6.py --serve --model gpt2 --device cpu --dtype float32 --no-chat-template


  curl http://localhost:8000/metrics

"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import AsyncGenerator

import torch

try:
    from velox_l0 import VeloxConfig
    from velox_l5 import (
        app, _state,
        ChatCompletionRequest, CompletionRequest,
        ChatCompletionResponse, ChatCompletionChunk,
        CompletionResponse, ChatChoice, CompletionChoice,
        ChatMessage, DeltaMessage, StreamChoice, UsageInfo,
        _decode_step_by_step,
    )
except ImportError as e:
    print(f"ERREUR : {e}")
    print("Mets velox_l0.py, velox_l5.py et velox_l6.py dans le même dossier.")
    sys.exit(1)

try:
    from prometheus_client import (
        Counter, Histogram, Gauge,
        generate_latest, CONTENT_TYPE_LATEST, REGISTRY,
    )
except ImportError:
    print("ERREUR : pip install prometheus-client")
    sys.exit(1)

try:
    from fastapi import Request
    from fastapi.responses import Response, StreamingResponse
    from starlette.middleware.base import BaseHTTPMiddleware
    import asyncio, threading, uuid
except ImportError:
    print("ERREUR : pip install fastapi")
    sys.exit(1)

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — MÉTRIQUES PROMETHEUS
# ══════════════════════════════════════════════════════════════════════════════

REQUESTS_TOTAL = Counter(
    "velox_requests_total",
    "Nombre total de requêtes par statut.",
    ["endpoint", "status"],
)

TOKENS_GENERATED = Counter(
    "velox_tokens_generated_total",
    "Nombre total de tokens de sortie générés.",
)

TTFT_HISTOGRAM = Histogram(
    "velox_ttft_seconds",
    "Temps jusqu'au premier token (TTFT) en secondes.",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

TPOT_HISTOGRAM = Histogram(
    "velox_tpot_seconds",
    "Temps par token de sortie (TPOT) en secondes.",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2],
)

E2E_HISTOGRAM = Histogram(
    "velox_request_duration_seconds",
    "Latence E2E de la requête en secondes.",
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)

ACTIVE_REQUESTS = Gauge(
    "velox_active_requests",
    "Nombre de requêtes actuellement en cours de traitement.",
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MIDDLEWARE (intercepte TOUTES les requêtes)
# ══════════════════════════════════════════════════════════════════════════════

class VeloxMetricsMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):

        if request.url.path == "/metrics":
            return await call_next(request)

        endpoint = "chat" if "chat" in request.url.path else "other"
        ACTIVE_REQUESTS.inc()
        t_start = time.perf_counter()

        try:
            response = await call_next(request)
            status = "ok" if response.status_code < 400 else "error"
            REQUESTS_TOTAL.labels(endpoint=endpoint, status=status).inc()
            E2E_HISTOGRAM.observe(time.perf_counter() - t_start)
            return response
        except Exception:
            REQUESTS_TOTAL.labels(endpoint=endpoint, status="error").inc()
            raise
        finally:
            ACTIVE_REQUESTS.dec()


app.add_middleware(VeloxMetricsMiddleware)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PATCH generate() pour TTFT + TPOT + tokens
# ══════════════════════════════════════════════════════════════════════════════

def _patch_runner_metrics():

    if not _state.ready or _state.runner is None:
        return

    original_generate = _state.runner.generate

    def instrumented_generate(*args, **kwargs):
        result = original_generate(*args, **kwargs)
        # Enregistrer les métriques depuis le résultat
        TTFT_HISTOGRAM.observe(result.ttft_ms / 1000.0)
        TPOT_HISTOGRAM.observe(result.tpot_ms / 1000.0)
        TOKENS_GENERATED.inc(result.output_tokens)
        return result

    _state.runner.generate = instrumented_generate
    logger.info("Métriques TTFT/TPOT/tokens activées sur runner.generate()")




@app.get("/metrics")
async def metrics():
    """Endpoint Prometheus — scrape par GET /metrics."""
    data = generate_latest(REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(description="Velox L6 — serveur avec métriques Prometheus.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--serve", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config = VeloxConfig(model_name=args.model, device=args.device, dtype=args.dtype)

    if args.serve:
        try:
            import uvicorn
        except ImportError:
            print("ERREUR : pip install uvicorn[standard]")
            sys.exit(1)

        _state.load(config, not args.no_chat_template)
        _patch_runner_metrics() 

        print(f"\n  Velox L6 — serveur + métriques Prometheus")
        print(f"  Modèle  : {config.model_name} | {config.resolved_device}")
        print(f"  API     : http://{args.host}:{args.port}/v1/chat/completions")
        print(f"  Métriques : http://{args.host}:{args.port}/metrics")
        print(f"  Docs    : http://{args.host}:{args.port}/docs")
        print()
        print("  Test métriques :")
        print(f"    curl http://localhost:{args.port}/metrics | grep velox")
        print()

        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
        return

    p.print_help()


if __name__ == "__main__":
    main()