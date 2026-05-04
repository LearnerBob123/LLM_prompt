"""
pipeline.py — End-to-end LLM Evaluation Pipeline

Usage:
    python pipeline.py           # Full run — all prompts
    python pipeline.py --mini    # Mini run — first 5 prompts (fast sanity check)
    python pipeline.py --fast    # Override FAST_MODE to True for this run
    python pipeline.py --full    # Override FAST_MODE to False (parametric + consistency + robustness)
"""
import sys
import os
import json
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ── Allow running from the framework subfolder OR its parent ─────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

import config  # noqa: E402 (must come after sys.path change)
from modules.retriever import Retriever
from modules.generator import generate_answer
from modules.claim_extractor import extract_claims
from modules.faithfulness import faithfulness_and_contradiction
from modules.confidence import confidence_score, claim_confidence_score
from modules.parametric import parametric_score
from modules.consistency import consistency_score
from modules.scorer import franq_score, classify_claim, prompt_score, model_score
from modules.prompt_metrics import (
    grounding_score, hallucination_rate, confidence_calibration,
    relevance_score, completeness_score,
)


def parse_args():
    parser = argparse.ArgumentParser(description="LLM Eval Framework Pipeline")
    parser.add_argument("--mini", action="store_true", help="Run only first 5 prompts")
    parser.add_argument("--fast", action="store_true", help="Force FAST_MODE=True")
    parser.add_argument("--full", action="store_true", help="Force FAST_MODE=False (slow, thorough)")
    parser.add_argument(
        "--data", default=None,
        help="Data folder prefix (default: 'data'). Use 'data2' to run the second dataset."
    )
    return parser.parse_args()


def run(mini: bool = False, force_fast: bool = False, force_full: bool = False, data_dir: str = None):
    # ── Apply mode overrides ─────────────────────────────────────────────────
    if force_fast:
        config.FAST_MODE = True
    if force_full:
        config.FAST_MODE = False

    # ── Apply data directory override ────────────────────────────────────────
    if data_dir and data_dir != "data":
        config.KNOWLEDGE_BASE_DIR = f"{data_dir}/knowledge_base/"
        config.PROMPTS_FILE = f"{data_dir}/prompts.json"
        config.GROUND_TRUTH_FILE = f"{data_dir}/ground_truth.json"
        config.RESULTS_FILE = f"outputs/results_{data_dir}.json"

    print(f"\n{'='*60}")
    print(f"  LLM Evaluation Pipeline")
    print(f"  Generator   : {config.GENERATOR_MODEL}")
    print(f"  Extractor   : {config.CLAIM_EXTRACTOR_MODEL}")
    print(f"  FAST_MODE   : {config.FAST_MODE}")
    print(f"  Mini run    : {mini}")
    print(f"  Data folder : {data_dir or 'data'}")
    print(f"  Output file : {config.RESULTS_FILE}")
    print(f"{'='*60}\n")

    # ── Load models ──────────────────────────────────────────────────────────
    print("[1/5] Loading retriever and embedding models...")
    retriever = Retriever()

    print("[2/5] Loading NLI model (first run downloads ~500 MB)...")
    from modules.faithfulness import _get_nli
    nli = _get_nli()  # warm up singleton
    embedder = SentenceTransformer(config.EMBEDDING_MODEL)

    # ── Load data ────────────────────────────────────────────────────────────
    print("[3/5] Loading prompts and ground truth...")
    with open(config.PROMPTS_FILE, encoding="utf-8") as f:
        prompts = json.load(f)

    try:
        with open(config.GROUND_TRUTH_FILE, encoding="utf-8") as f:
            ground_truths = json.load(f)
    except FileNotFoundError:
        ground_truths = {}

    if mini:
        prompts = prompts[:5]
        print(f"  Mini mode: running {len(prompts)} prompts.\n")

    # ── Run evaluation ───────────────────────────────────────────────────────
    print("[4/5] Evaluating prompts...")
    all_prompt_results = []
    all_prompt_scores = []

    for item in tqdm(prompts, desc="Evaluating"):
        pid = item["id"]
        prompt = item["prompt"]
        ptype = item.get("type", "factual")

        # Step 1 — Retrieve context
        context_docs = retriever.retrieve(prompt)

        # Step 2 — Generate response (with context) + context-free response for parametric
        gen = generate_answer(prompt, context_docs, with_context=True)
        response = gen["response"]
        token_logprobs = gen["token_logprobs"]

        cf_gen = generate_answer(prompt, context_docs=[], with_context=False)
        context_free_response = cf_gen["response"]

        # Step 3 — Extract claims
        claims = extract_claims(response)
        if not claims:
            # Fallback: treat the full response as one claim
            claims = [response[:300]]

        # Step 4 — Per-claim token alignment data (from generator logprobs)
        token_data = gen.get("token_data", [])
        # Consistency is prompt-level; compute once
        consist = consistency_score(prompt, context_docs)

        # Step 5 — Per-claim scoring
        claim_results = []
        faith_scores, conf_scores = [], []

        for claim_text in claims:
            faith, contra = faithfulness_and_contradiction(claim_text, context_docs)
            # Per-claim confidence: align claim text to token sequence when possible
            conf = (
                claim_confidence_score(claim_text, token_data)
                if token_data
                else confidence_score(token_logprobs)
            )
            param = parametric_score(claim_text, context_free_response)
            score = franq_score(faith, conf, param, contra, consist)
            label = classify_claim(faith, param)

            faith_scores.append(faith)
            conf_scores.append(conf)

            claim_results.append({
                "text": claim_text,
                "faithfulness": faith,
                "confidence": conf,
                "parametric": param,
                "contradiction": contra,
                "consistency": consist,
                "franq_score": score,
                "classification": label,
            })

        # Step 6 — Prompt-level aggregated metrics
        p_score = prompt_score([c["franq_score"] for c in claim_results])
        gt = ground_truths.get(pid, "")

        prompt_result = {
            "id": pid,
            "type": ptype,
            "prompt": prompt,
            "response": response,
            "context": context_docs,
            "claims": claim_results,
            "prompt_score": p_score,
            "grounding_score": grounding_score(faith_scores),
            "hallucination_rate": hallucination_rate(faith_scores, conf_scores),
            "calibration": confidence_calibration(conf_scores, faith_scores),
            "consistency_score": consist,
            "relevance_score": relevance_score(prompt, response, embedder),
            "completeness_score": completeness_score(response, gt, embedder),
        }

        all_prompt_results.append(prompt_result)
        all_prompt_scores.append(p_score)

    # ── Save results ─────────────────────────────────────────────────────────
    print("\n[5/5] Saving results...")
    os.makedirs(os.path.dirname(config.RESULTS_FILE), exist_ok=True)
    final = {
        "model_score": model_score(all_prompt_scores),
        "fast_mode": config.FAST_MODE,
        "generator_model": config.GENERATOR_MODEL,
        "claim_extractor_model": config.CLAIM_EXTRACTOR_MODEL,
        "prompts": all_prompt_results,
    }

    with open(config.RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Done! Results saved to: {config.RESULTS_FILE}")
    print(f"  Overall Model Score : {final['model_score']}")
    print(f"  Prompts evaluated   : {len(all_prompt_results)}")
    avg_hr = sum(p["hallucination_rate"] for p in all_prompt_results) / len(all_prompt_results)
    avg_gs = sum(p["grounding_score"] for p in all_prompt_results) / len(all_prompt_results)
    print(f"  Avg Hallucination   : {round(avg_hr, 3)}")
    print(f"  Avg Grounding       : {round(avg_gs, 3)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    args = parse_args()
    run(mini=args.mini, force_fast=args.fast, force_full=args.full, data_dir=args.data)
