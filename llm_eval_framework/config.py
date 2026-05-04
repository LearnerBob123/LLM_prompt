# config.py — Central configuration for the LLM Eval Framework
import os
from dotenv import load_dotenv

load_dotenv()

# ── Ollama Settings ─────────────────────────────────────────────────────────
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Primary generator (main RAG answers)
GENERATOR_MODEL = "llama3.2"
# Claim extractor — best JSON mode adherence
CLAIM_EXTRACTOR_MODEL = "gemma3:4b"
# Lightweight model for consistency re-sampling
CONSISTENCY_MODEL = "gemma3:1b"

# ── Local Model Settings ─────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

# ── Pipeline Settings ────────────────────────────────────────────────────────
TOP_K_DOCS = 3
NUM_CONSISTENCY_SAMPLES = 3

# FAST_MODE = True  → skip parametric, consistency, robustness (3× fewer LLM calls)
# Set to False for full academic-quality results (~10-15 min for 20 prompts)
FAST_MODE = True

# ── Thresholds ───────────────────────────────────────────────────────────────
FAITHFULNESS_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.4

# ── Paths ────────────────────────────────────────────────────────────────────
KNOWLEDGE_BASE_DIR = "data/knowledge_base/"
PROMPTS_FILE = "data/prompts.json"
GROUND_TRUTH_FILE = "data/ground_truth.json"
RESPONSES_FILE = "outputs/responses.json"
CLAIMS_FILE = "outputs/claims.json"
RESULTS_FILE = "outputs/results.json"
