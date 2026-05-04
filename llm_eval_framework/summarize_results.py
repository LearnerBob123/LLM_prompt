"""Quick script to print a compact summary of outputs/results.json."""
import json

import os
BASE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(BASE, "outputs", "results.json")))
print(f"Model score : {d['model_score']}")
print(f"Generator   : {d['generator_model']}")
print()

for p in d["prompts"]:
    print(f"{p['id']} [{p['type']}]  {p['prompt'][:60]}")
    for c in p["claims"]:
        faith = c["faithfulness"]
        contra = c["contradiction"]
        cls = c["classification"]
        text = c["text"][:60]
        print(f"  faith={faith:.3f} contra={contra:.3f}  {cls}  {text}")
    gs = p["grounding_score"]
    ps = p["prompt_score"]
    print(f"  => grounding={gs:.3f}  prompt_score={ps:.3f}")
    print()
