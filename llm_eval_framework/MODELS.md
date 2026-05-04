# Models Reference

A complete list of every model used in the LLM Eval Framework, what it does, and drop-in alternatives you can swap via `config.py`.

---

## Currently Used Models

### 1. Generator — `mistral`
**Role:** Answers the user's question using retrieved context (the main RAG response).  
**Where:** `modules/generator.py`, `config.GENERATOR_MODEL`  
**Type:** Ollama (local)  
**Size:** 4.7 GB  

### 2. Claim Extractor — `qwen2.5`
**Role:** Decomposes the generated response into individual atomic factual claims (e.g., "ENIAC was built in 1945"). Each claim is scored independently.  
**Where:** `modules/claim_extractor.py`, `config.CLAIM_EXTRACTOR_MODEL`  
**Type:** Ollama (local), run in JSON mode  
**Size:** 4.7 GB  
**Why this model:** Best JSON-mode adherence among tested local models — rarely outputs malformed JSON.

### 3. Consistency Sampler — `gemma3:4b`
**Role:** Generates multiple paraphrased answers at temperature=0.7 to test whether the model gives the same answer regardless of sampling randomness. Skipped in `FAST_MODE=True`.  
**Where:** `modules/consistency.py`, `config.CONSISTENCY_MODEL`  
**Type:** Ollama (local)  
**Size:** 3.3 GB  
**Why this model:** Smaller/faster; consistency sampling runs `NUM_CONSISTENCY_SAMPLES` (default 3) extra LLM calls per prompt, so speed matters.

### 4. Embedding Model — `all-MiniLM-L6-v2`
**Role:** Converts text into dense vector embeddings for:
- FAISS retrieval (finding relevant knowledge-base docs)
- `relevance_score` (prompt ↔ response similarity)
- `completeness_score` (response ↔ ground truth similarity)
- `robustness_score` (original response ↔ paraphrased-prompt response similarity)  

**Where:** `modules/retriever.py`, `modules/prompt_metrics.py`, `config.EMBEDDING_MODEL`  
**Type:** HuggingFace sentence-transformers (auto-downloaded, ~90 MB)  

### 5. NLI Model — `cross-encoder/nli-deberta-v3-base`
**Role:** Classifies whether a document **entails** or **contradicts** a claim. Powers:
- `faithfulness_score` — P(context supports claim)
- `contradiction_score` — P(context contradicts claim)
- `parametric_score` — P(model's parametric knowledge entails claim)  

**Where:** `modules/faithfulness.py`, `modules/contradiction.py`, `modules/parametric.py`, `config.NLI_MODEL`  
**Type:** HuggingFace transformers CrossEncoder (auto-downloaded, ~500 MB, runs on GPU)  

---

## Alternative Options

### Generator Alternatives (`GENERATOR_MODEL`)

| Model | Ollama Tag | Size | Notes |
|---|---|---|---|
| `mistral` ✅ **(current)** | `ollama pull mistral` | 4.7 GB | Solid general-purpose, good instruction following |
| `llama3.2` | `ollama pull llama3.2` | 2.0 GB | Smaller, faster, slightly lower quality |
| `llama3.1:8b` | `ollama pull llama3.1:8b` | 4.7 GB | Strong reasoning, good factual accuracy |
| `phi4` | `ollama pull phi4` | 9.1 GB | Best quality on 16 GB+ VRAM, Microsoft model |
| `deepseek-r1:7b` | `ollama pull deepseek-r1:7b` | 4.9 GB | Excellent on reasoning/math prompts |
| `gemma3:12b` | `ollama pull gemma3:12b` | 8.1 GB | High quality, needs more VRAM |
| `gemma3:4b` | `ollama pull gemma3:4b` | 3.3 GB | Fast, lower quality |
| `qwen2.5:14b` | `ollama pull qwen2.5:14b` | 9.0 GB | Best overall for factual RAG on 16 GB VRAM |
| `mistral-nemo` | `ollama pull mistral-nemo` | 7.1 GB | Long context (128k), good for large docs |

### Claim Extractor Alternatives (`CLAIM_EXTRACTOR_MODEL`)

| Model | Notes |
|---|---|
| `qwen2.5` ✅ **(current)** | Best JSON adherence; rarely produces malformed output |
| `qwen2.5:14b` | Higher quality claims, more granular decomposition |
| `llama3.1:8b` | Good alternative; may need occasional JSON fallback handling |
| `phi4` | Excellent instruction following; produces clean claim lists |
| `mistral` | Works but occasionally wraps JSON in markdown fences (handled by parser) |
| `gemma3:4b` | Acceptable quality, faster than 7B+ models |

**Note:** The extractor must support `format="json"` (Ollama JSON mode). All models above do.

### Consistency Sampler Alternatives (`CONSISTENCY_MODEL`)

| Model | Size | Notes |
|---|---|---|
| `gemma3:4b` ✅ **(current)** | 3.3 GB | Fast, lightweight |
| `gemma3:1b` | 0.8 GB | Fastest option, lower answer quality |
| `llama3.2` | 2.0 GB | Good quality/speed balance |
| `phi3:mini` | 2.2 GB | Very fast on CPU fallback |
| `mistral` | 4.7 GB | Same model as generator (consistent style) |

**Tip:** Since this model only needs to paraphrase/re-answer (not extract structured data), quality matters less than speed. Prefer models ≤4B parameters here.

### Embedding Model Alternatives (`EMBEDDING_MODEL`)

| Model | Size | Dim | Notes |
|---|---|---|---|
| `all-MiniLM-L6-v2` ✅ **(current)** | ~90 MB | 384 | Fast, good for short-medium text |
| `all-MiniLM-L12-v2` | ~120 MB | 384 | Slightly better quality, slightly slower |
| `all-mpnet-base-v2` | ~420 MB | 768 | Higher quality embeddings, ~3× slower |
| `BAAI/bge-small-en-v1.5` | ~130 MB | 384 | State-of-art small model (MTEB leaderboard) |
| `BAAI/bge-base-en-v1.5` | ~430 MB | 768 | Better retrieval quality, larger |
| `intfloat/e5-small-v2` | ~130 MB | 384 | Strong retrieval performance |
| `nomic-ai/nomic-embed-text-v1` | ~550 MB | 768 | Long context (8192 tokens), good for long docs |

**Tip:** Upgrading to `BAAI/bge-base-en-v1.5` gives a noticeable retrieval quality boost with modest overhead.

### NLI Model Alternatives (`NLI_MODEL`)

| Model | Size | Notes |
|---|---|---|
| `cross-encoder/nli-deberta-v3-base` ✅ **(current)** | ~500 MB | Best accuracy; runs on GPU; ~10 min for 5 prompts |
| `cross-encoder/nli-MiniLM2-L6-H768` | ~120 MB | 4× faster, slightly lower accuracy — good for testing |
| `cross-encoder/nli-deberta-v3-small` | ~180 MB | Good balance of speed and accuracy |
| `typeform/distilbart-mnli-12-1` | ~420 MB | Older model, use only if deberta is unavailable |
| `facebook/bart-large-mnli` | ~1.6 GB | High accuracy, large — only if VRAM allows |

**Tip for faster iteration:** Switch to `cross-encoder/nli-MiniLM2-L6-H768` during development. Switch back to `nli-deberta-v3-base` for final results.

---

## Quick Swap Examples

**Use a faster NLI during development:**
```python
# config.py
NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
```

**Use a stronger generator for full academic evaluation:**
```python
# config.py
GENERATOR_MODEL = "qwen2.5:14b"   # pull first: ollama pull qwen2.5:14b
```

**Minimal RAM/VRAM setup (all models ≤2 GB):**
```python
# config.py
GENERATOR_MODEL    = "llama3.2"          # 2.0 GB
CLAIM_EXTRACTOR_MODEL = "gemma3:4b"      # 3.3 GB (smallest with good JSON)
CONSISTENCY_MODEL  = "gemma3:1b"         # 0.8 GB
EMBEDDING_MODEL    = "all-MiniLM-L6-v2" # 90 MB
NLI_MODEL          = "cross-encoder/nli-MiniLM2-L6-H768"  # 120 MB
```

**Best quality setup (16 GB+ VRAM):**
```python
# config.py
GENERATOR_MODEL    = "qwen2.5:14b"
CLAIM_EXTRACTOR_MODEL = "phi4"
CONSISTENCY_MODEL  = "mistral"
EMBEDDING_MODEL    = "BAAI/bge-base-en-v1.5"
NLI_MODEL          = "cross-encoder/nli-deberta-v3-base"
```
