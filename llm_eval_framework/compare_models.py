"""
compare_models.py — Run the evaluation pipeline across multiple generator models.

Each model runs over the same fixed prompts and config, saving results to:
    outputs/results_<model_tag>.json   (e.g. results_llama3.2.json)

Usage:
    # Default: compare llama3.2, gemma3:4b, mistral on data prompts (mini run)
    python compare_models.py --mini

    # Specify models explicitly
    python compare_models.py --models llama3.2 mistral qwen2.5 --mini

    # Run on data2 dataset
    python compare_models.py --mini --data data2 --models llama3.2 mistral

    # Full run (all 20 prompts)
    python compare_models.py --models llama3.2 mistral
"""
import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

import config  # noqa: E402

DEFAULT_MODELS = ["llama3.2", "mistral", "qwen2.5"]


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-model comparison runner")
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help=f"Models to compare (default: {DEFAULT_MODELS})"
    )
    parser.add_argument("--mini", action="store_true", help="Run only first 5 prompts")
    parser.add_argument("--data", default=None, help="Data folder (e.g. data2)")
    return parser.parse_args()


def model_tag(model_name: str) -> str:
    """Convert model name to a safe filename tag."""
    return model_name.replace(":", "_").replace("/", "_")


def main():
    args = parse_args()

    # Import run() from pipeline — it handles all the heavy work
    from pipeline import run

    print(f"\n{'='*60}")
    print(f"  Multi-Model Comparison")
    print(f"  Models  : {args.models}")
    print(f"  Mini run: {args.mini}")
    print(f"  Data    : {args.data or 'data'}")
    print(f"{'='*60}\n")

    summary = []

    for model in args.models:
        print(f"\n{'─'*60}")
        print(f"  Running model: {model}")
        print(f"{'─'*60}")

        # Override generator model and results output path
        config.GENERATOR_MODEL = model

        # Build results filename: e.g. results_llama3.2_data2.json
        data_suffix = f"_{args.data}" if args.data else ""
        tag = model_tag(model)
        config.RESULTS_FILE = f"outputs/results_{tag}{data_suffix}.json"

        # Reset data paths if needed (run() will re-apply data_dir override)
        config.KNOWLEDGE_BASE_DIR = "data/knowledge_base/"
        config.PROMPTS_FILE = "data/prompts.json"
        config.GROUND_TRUTH_FILE = "data/ground_truth.json"

        try:
            run(mini=args.mini, data_dir=args.data)
            summary.append({"model": model, "output": config.RESULTS_FILE, "status": "OK"})
        except Exception as e:
            print(f"\n[ERROR] Model {model} failed: {e}")
            summary.append({"model": model, "output": config.RESULTS_FILE, "status": f"FAILED: {e}"})

    # Print final summary table
    print(f"\n{'='*60}")
    print("  Comparison Complete — Summary")
    print(f"{'='*60}")
    print(f"  {'Model':<20} {'Status':<10} Output file")
    print(f"  {'-'*55}")
    for s in summary:
        print(f"  {s['model']:<20} {s['status']:<10} {s['output']}")
    print(f"{'='*60}\n")
    print("  Open dashboard.py to compare results side-by-side.")


if __name__ == "__main__":
    main()
