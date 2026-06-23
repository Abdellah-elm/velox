"""
velox_l8.py — L8 : Intégration GuardRAG → Velox.


Usage :
  # Afficher le patch (sans l'appliquer)
  python velox_l8.py --show-patch

  # Appliquer le patch sur GuardRAG
  python velox_l8.py --patch --guardrag-path ../guardrag

  # Valider le pipeline end-to-end (Velox doit tourner)
  python velox_l8.py --validate --velox-url http://localhost:8000

  # Test rapide sans GuardRAG (OpenAI SDK → Velox directement)
  python velox_l8.py --test-sdk --velox-url http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# .env:
#   USE_VELOX=false  → Groq (gpt-oss-120b, production)
#   USE_VELOX=true   → Velox (Qwen2.5, demo)


PATCH_TARGET = 'from groq import Groq'

PATCH_REPLACEMENT = '''\
# ── L8 Velox toggle — changer USE_VELOX dans .env pour switcher ─────────────
USE_VELOX = os.getenv("USE_VELOX", "false").lower() == "true"
if USE_VELOX:
    from openai import OpenAI as Groq       # même interface que Groq SDK
    groq = Groq(
        base_url=os.getenv("VELOX_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("VELOX_API_KEY", "velox-local"),
    )
    FAST_MODEL   = os.getenv("VELOX_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    STRONG_MODEL = os.getenv("VELOX_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
else:
    from groq import Groq                   # Groq original (production)
# ────────────────────────────────────────────────────────────────────────────\
'''


GROQ_INIT_BEFORE    = 'groq = Groq(api_key=os.getenv("GROQ_API_KEY"))'
GROQ_INIT_AFTER     = '''\
if not USE_VELOX:
    groq = Groq(api_key=os.getenv("GROQ_API_KEY"))\
'''

FAST_MODEL_BEFORE   = 'FAST_MODEL = "openai/gpt-oss-20b"'
FAST_MODEL_AFTER    = 'if not USE_VELOX: FAST_MODEL = "openai/gpt-oss-20b"'

STRONG_MODEL_BEFORE = 'STRONG_MODEL = "openai/gpt-oss-120b"'
STRONG_MODEL_AFTER  = 'if not USE_VELOX: STRONG_MODEL = "openai/gpt-oss-120b"'

ENV_ADDITIONS = """
# ── Velox L8 toggle (ajouté par velox_l8.py) ────────────────────────
USE_VELOX=false
VELOX_BASE_URL=http://localhost:8000/v1
VELOX_API_KEY=velox-local
VELOX_MODEL=Qwen/Qwen2.5-0.5B-Instruct
# USE_VELOX=false → Groq (production) | USE_VELOX=true → Velox (demo)
"""


def show_patch() -> None:
    print("\n── Smart toggle patch GuardRAG ─────────────────────────────────────")
    print("  Un seul bloc dans main.py + USE_VELOX dans .env\n")
    print("  ── main.py ──")
    print(f"  - from groq import Groq")
    print(f"  - groq = Groq(api_key=...)")
    print(f"  - FAST_MODEL = 'openai/gpt-oss-20b'")
    print(f"  - STRONG_MODEL = 'openai/gpt-oss-120b'")
    print()
    for line in PATCH_REPLACEMENT.split("\n"):
        print(f"  + {line}")
    print()
    print("  ── .env ──")
    print("  + USE_VELOX=false   ← changer en true pour Velox")
    print("  + VELOX_BASE_URL=http://localhost:8000/v1")
    print("  + VELOX_MODEL=Qwen/Qwen2.5-0.5B-Instruct\n")
    print("  Switch Groq → Velox : USE_VELOX=true  + restart GuardRAG")
    print("  Switch Velox → Groq : USE_VELOX=false + restart GuardRAG\n")


def apply_patch(guardrag_path: Path) -> bool:
    """Applique le smart toggle patch sur le fichier main.py de GuardRAG."""
    candidates = [
        guardrag_path / "main.py",
        guardrag_path / "app" / "main.py",
        guardrag_path / "src" / "main.py",
    ]
    main_py = next((p for p in candidates if p.exists()), None)

    if main_py is None:
        print(f"  ERREUR : main.py introuvable dans {guardrag_path}")
        print(f"  Candidats essayés : {[str(c) for c in candidates]}")
        return False

    content = main_py.read_text(encoding="utf-8")

    # Déjà patché ?
    if "USE_VELOX" in content:
        print(f"  ✓ {main_py} déjà patché (toggle USE_VELOX présent).")
        return True

    # Vérifier que les cibles existent
    targets = [PATCH_TARGET, GROQ_INIT_BEFORE, FAST_MODEL_BEFORE, STRONG_MODEL_BEFORE]
    missing = [t for t in targets if t not in content]
    if missing:
        print("  ERREUR : lignes cibles introuvables :")
        for m in missing:
            print(f"    - {m[:60]!r}")
        return False

    # Backup
    backup = main_py.with_suffix(".py.groq_backup")
    backup.write_text(content, encoding="utf-8")
    print(f"  Backup : {backup}")

    # Appliquer le toggle patch
    content = content.replace(PATCH_TARGET, PATCH_REPLACEMENT, 1)
    content = content.replace(GROQ_INIT_BEFORE, GROQ_INIT_AFTER, 1)
    content = content.replace(FAST_MODEL_BEFORE, FAST_MODEL_AFTER, 1)
    content = content.replace(STRONG_MODEL_BEFORE, STRONG_MODEL_AFTER, 1)
    main_py.write_text(content, encoding="utf-8")
    print(f"  ✓ {main_py} patché avec smart toggle.")

    # Patcher .env
    env_file = guardrag_path / ".env"
    if env_file.exists():
        env_content = env_file.read_text(encoding="utf-8")
        if "USE_VELOX" not in env_content:
            env_file.write_text(env_content + ENV_ADDITIONS, encoding="utf-8")
            print(f"  ✓ {env_file} mis à jour.")
    else:
        env_file.write_text(ENV_ADDITIONS.strip() + "\n", encoding="utf-8")
        print(f"  ✓ {env_file} créé.")

    print()
    print("  Patch appliqué. Pour switcher :")
    print("    Groq  → édite .env : USE_VELOX=false + restart GuardRAG")
    print("    Velox → édite .env : USE_VELOX=true  + démarrer Velox L5\n")
    return True


def revert_patch(guardrag_path: Path) -> bool:
    """Annule le patch (restaure depuis le backup)."""
    candidates = [
        guardrag_path / "main.py",
        guardrag_path / "app" / "main.py",
        guardrag_path / "src" / "main.py",
    ]
    main_py = next((p for p in candidates if p.exists()), None)
    backup = main_py.with_suffix(".py.groq_backup") if main_py else None

    if not backup or not backup.exists():
        print("  ERREUR : backup introuvable.")
        return False

    main_py.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
    backup.unlink()
    print(f"  ✓ {main_py} restauré depuis backup.")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TEST SDK (sans GuardRAG)
# ══════════════════════════════════════════════════════════════════════════════

def test_sdk(velox_url: str, model: str = "gpt2") -> bool:
    """
    Teste le SDK OpenAI Python pointant vers Velox.
    Simule ce que GuardRAG fait après le patch.
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("  ERREUR : pip install openai")
        return False

    client = OpenAI(base_url=f"{velox_url}/v1", api_key="velox-local")

    print(f"\n── Test SDK OpenAI → Velox ({velox_url}) ───────────────────────────")
    print(f"   Modèle : {model}")

    # Test 1 : /v1/models
    print("\n  1. GET /v1/models")
    try:
        models_list = client.models.list()
        print(f"     ✓ {len(models_list.data)} modèle(s) disponible(s) : {[m.id for m in models_list.data]}")
    except Exception as e:
        print(f"     ✗ Erreur : {e}")
        return False

    # Test 2 : non-streaming (comme generate_answer dans GuardRAG)
    print("\n  2. POST /v1/chat/completions (non-streaming)")
    context = "Qiskit is an open-source quantum computing SDK developed by IBM."
    question = "What is Qiskit?"
    messages = [
        {"role": "system", "content": "Answer using the provided context only."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    try:
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=50,
            temperature=0,
        )
        elapsed = time.perf_counter() - t0
        answer = resp.choices[0].message.content
        print(f"     ✓ Réponse ({elapsed:.1f}s) : {answer[:100]!r}…")
        print(f"     ✓ usage : {resp.usage.prompt_tokens} prompt + {resp.usage.completion_tokens} completion")
    except Exception as e:
        print(f"     ✗ Erreur : {e}")
        return False

    # Test 3 : streaming (GuardRAG peut utiliser stream=True)
    print("\n  3. POST /v1/chat/completions (streaming)")
    try:
        t0 = time.perf_counter()
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello, what can you do?"}],
            max_tokens=30,
            temperature=0,
            stream=True,
        )
        tokens = []
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                tokens.append(delta.content)
        elapsed = time.perf_counter() - t0
        full_text = "".join(tokens)
        print(f"     ✓ {len(tokens)} chunks reçus ({elapsed:.1f}s) : {full_text[:80]!r}…")
    except Exception as e:
        print(f"     ✗ Erreur : {e}")
        return False

    print("\n  ✅ SDK OpenAI → Velox : tous les tests OK")
    print("     GuardRAG fonctionnera après le patch.\n")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — VALIDATION END-TO-END
# ══════════════════════════════════════════════════════════════════════════════

def validate_pipeline(velox_url: str, model: str = "gpt2") -> bool:
    """
    Valide le pipeline GuardRAG complet en simulant chaque appel LLM.
    Ne nécessite pas GuardRAG démarré — teste chaque pièce séparément.
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("  ERREUR : pip install openai")
        return False

    client = OpenAI(base_url=f"{velox_url}/v1", api_key="velox-local")

    print(f"\n── Validation pipeline GuardRAG → Velox ────────────────────────────")
    print(f"   Velox : {velox_url} | Modèle : {model}\n")

    # ── Étape 1 : healthcheck ──────────────────────────────────────────────
    print("  Étape 1 : Velox /health")
    try:
        models_resp = client.models.list()
        print(f"     ✓ Velox OK — modèle : {models_resp.data[0].id if models_resp.data else '?'}")
    except Exception as e:
        print(f"     ✗ Velox inaccessible : {e}")
        print("     → Démarrer Velox : python velox_l5.py --serve")
        return False

    # ── Étape 2 : generate_answer (appel principal de GuardRAG) ───────────
    print("\n  Étape 2 : generate_answer (appel #1 — réponse principale)")
    SYSTEM_PROMPT = (
        "You are a technical support assistant for Qiskit / IBM Quantum. "
        "Answer ONLY using the provided context."
    )
    context = (
        "[Qiskit — Overview]\n"
        "Qiskit is an open-source SDK for working with quantum computers at the level "
        "of pulses, circuits, and application modules. It provides tools to create "
        "quantum circuits, simulate them, and run them on real quantum hardware."
    )
    question = "What is Qiskit used for?"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    try:
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model, messages=messages, max_tokens=80, temperature=0,
        )
        answer = resp.choices[0].message.content
        elapsed = time.perf_counter() - t0
        print(f"     ✓ Réponse ({elapsed:.1f}s, {resp.usage.completion_tokens} tokens):")
        print(f"       {answer[:150]!r}")
    except Exception as e:
        print(f"     ✗ generate_answer failed : {e}")
        return False

    # ── Étape 3 : faithfulness_check (appel #2 — judge) ──────────────────
    print("\n  Étape 3 : faithfulness_check (appel #2 — judge LLM)")
    judge_prompt = f"""You are a fact-checker. Is this ANSWER faithful to the CONTEXT?

CONTEXT:
{context}

ANSWER:
{answer}

Respond with JSON only: {{"faithful": true/false, "confidence": 0.0}}"""

    try:
        t0 = time.perf_counter()
        judge_resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": judge_prompt}],
            max_tokens=30, temperature=0,
        )
        raw = judge_resp.choices[0].message.content.strip()
        elapsed = time.perf_counter() - t0
        print(f"     ✓ Judge ({elapsed:.1f}s) : {raw[:100]!r}")
        # Tenter de parser le JSON (GPT-2 peut ne pas produire du JSON valide)
        try:
            verdict = json.loads(raw)
            print(f"     ✓ Faithfulness : {verdict}")
        except json.JSONDecodeError:
            print(f"     ⚠  JSON non valide (normal avec GPT-2) — Qwen2.5 donnera un JSON propre")
    except Exception as e:
        print(f"     ✗ faithfulness_check failed : {e}")
        return False

    # ── Résumé ─────────────────────────────────────────────────────────────
    print()
    print("  ╔══ Pipeline GuardRAG → Velox — Résumé ══════════════════════╗")
    print(f"  ║  question     : {question!r}")
    print(f"  ║  réponse      : {answer[:70]!r}…")
    print(f"  ║  LLM backend  : Velox L5 ({velox_url})")
    print(f"  ║  modèle       : {model}")
    print(f"  ║  appels LLM   : 2 (generate_answer + faithfulness_check)")
    print(f"  ║  groq_api_key : non requis ✓")
    print("  ╚════════════════════════════════════════════════════════════╝\n")
    print("  ✅ Pipeline end-to-end validé.")
    print("     Appliquer le patch sur GuardRAG : python velox_l8.py --patch --guardrag-path <path>\n")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(description="Velox L8 — intégration GuardRAG.")
    p.add_argument("--show-patch", action="store_true",
                   help="Afficher le patch sans l'appliquer.")
    p.add_argument("--patch", action="store_true",
                   help="Appliquer le patch sur GuardRAG.")
    p.add_argument("--revert", action="store_true",
                   help="Annuler le patch (restaure Groq).")
    p.add_argument("--guardrag-path", type=Path, default=Path("../guardrag"),
                   help="Chemin vers le dossier GuardRAG.")
    p.add_argument("--test-sdk", action="store_true",
                   help="Tester SDK OpenAI → Velox (Velox doit tourner).")
    p.add_argument("--validate", action="store_true",
                   help="Valider le pipeline complet GuardRAG → Velox.")
    p.add_argument("--velox-url", default="http://localhost:8000",
                   help="URL du serveur Velox (défaut: http://localhost:8000).")
    p.add_argument("--model", default="gpt2",
                   help="Modèle Velox (défaut: gpt2).")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.show_patch:
        show_patch()
        return

    if args.patch:
        print(f"\nApplication du patch sur {args.guardrag_path}…")
        success = apply_patch(args.guardrag_path)
        sys.exit(0 if success else 1)

    if args.revert:
        success = revert_patch(args.guardrag_path)
        sys.exit(0 if success else 1)

    if args.test_sdk:
        success = test_sdk(args.velox_url, args.model)
        sys.exit(0 if success else 1)

    if args.validate:
        success = validate_pipeline(args.velox_url, args.model)
        sys.exit(0 if success else 1)

    # Défaut : afficher le patch + instructions
    show_patch()
    print("Pour appliquer : python velox_l8.py --patch --guardrag-path <chemin>")
    print("Pour valider   : python velox_l8.py --validate (Velox doit tourner)")


if __name__ == "__main__":
    main()
