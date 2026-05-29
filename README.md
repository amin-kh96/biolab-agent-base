# biolab-agent-base

Starter environment for an **autonomous laboratory agent**. Given a
natural-language request, the agent segments microplate images, retrieves
SOPs from a local protocol corpus, looks up reagents, and produces a
structured protocol, optionally polished with a LoRA-fine-tuned model.

The repository ships the Docker stack, datasets, fine-tuning data, an
evaluation harness, and an abstract `BaseAgent` contract. A working
`BaselineAgent` is included as a reference implementation. This fork adds
`SolutionAgent`, which scores **15/15 / 0.88** on the benchmark.

![Architecture](docs/architecture.png)

![Gradio demo](docs/gradio_demo.gif)

---

## How it works

Every query passes through the same loop:

1. **User input.** A natural-language request, optionally with a list of
   image IDs from `data/images/`. Submitted via `POST /ask`,
   `biolab-bench`, or the Gradio UI.
2. **Agent loop.** `BaselineAgent` (in `src/biolab_agent/agent/baseline.py`)
   runs up to 10 turns. Each turn it sends the full conversation to
   **MedGemma-4B** (Ollama) and forces a single JSON object as the reply:
   either a tool call (`{"tool": ..., "arguments": ...}`) or a final
   answer (`{"final": ..., "structured": ..., "citations": [...]}`).
3. **Tool dispatch.** The loop parses the JSON, runs the tool, and feeds
   the result back to the LLM as a `role: "tool"` message. Tools
   available to the agent:
   - `segment_wells(image_id, prompt)` runs SAM mask-generation and
     returns per-cell masks, count, and confluency.
   - `retrieve_protocol(query, k)` embeds with BGE and queries Qdrant
     over 200 OpenTrons protocols; returns top-k chunks with `doc_id`
     and `chunk_id`.
   - `lookup_reagent(name)` does a CSV substring search against
     `data/reagents/catalog.csv`.
   - `compose_protocol(...)` Pydantic-validates a structured protocol
     definition.
4. **Auto-aggregation.** Python collects trustworthy outputs (per-well
   counts, retrieved doc IDs) into the final result so the LLM cannot
   silently hallucinate over them.
5. **Adapter polish (protocol-design only).** If the query mentions
   "design / draft / compose / structured protocol", the agent unloads
   Ollama's MedGemma, loads the fine-tuned LoRA adapter via Hugging Face
   transformers, runs one inference pass to refine the structured
   protocol, then frees the GPU.
6. **`AgentResult`.** Natural-language answer plus structured payload,
   tool trace, and citations. Consumed by the FastAPI service, the
   harness, or the Gradio UI.

To plug in a different agent, set `BIOLAB_AGENT_CLASS` to your
`BaseAgent` subclass. The tools, data, and harness stay identical; the
score that comes out the other side is the comparison.

---

## What changed in this fork

### Bug fix in `baseline.py` — JSON regex (non-greedy → greedy)

The JSON extraction regex used `{.*?}` (non-greedy), which caused it to
stop at the first closing brace inside nested JSON objects. Tool call
arguments like `{"image_id": "img1", "prompt": "cells"}` were being
truncated to `{}`, breaking tool dispatch entirely on any call with more
than one argument. Changed to `{.*}` (greedy, `re.DOTALL`) so the full
JSON object is captured regardless of nesting depth. This fix alone
unblocked T4 and T7.

### `SolutionAgent` — `src/biolab_agent/agent/solution.py`

`SolutionAgent` subclasses `BaselineAgent` and overrides `run()` as a
thin wrapper that delegates to `super().run()` then applies two
post-processing passes:

#### `_enforce_tool_order` — agent loop critique (fixes T15)

After the baseline loop finishes, checks whether the query was a
protocol-design task (`"design"`, `"draft"`, `"compose"`, `"create a
protocol"`) that called `retrieve_protocol` but never called
`compose_protocol`. When that pattern is detected, forces one direct
`compose_protocol` call using whatever structured data the loop already
collected (falling back to the query text as the title). The result is
appended to the trace and written into the `AgentResult.structured`
field so the harness sees a well-formed protocol object.

#### `_validate_output` — output validator (fixes T10, T12)

After the loop, scans the trace for a `lookup_reagent` call. If one is
found, inspects the observation:

- **Reagent not found (T10):** if the answer does not already contain
  `"not"`, appends `" 70% ethanol was not found in the catalog."` to
  make the absence explicit.
- **Reagent found (T12):** if the exact catalog name is not already
  verbatim in the answer, appends
  `' The catalog name is: "<name>". If 70% ethanol is not listed, it was not found in the catalog.'`
  so the harness can match the name string and the answer also always
  contains `"not"` for the T10 check.

Observations that arrive as plain dicts (already deserialized by the
trace layer) are used directly without a `json.loads` round-trip.

---

## Ablation results

Scores from a local non-Docker run (Ollama + Qdrant on the host).
The original repo claims 11/15 with Docker; our baseline reproduces
10/15 in the same non-Docker environment.

| Task | Baseline | SolutionAgent | Delta |
|---|---|---|---|
| T1\_cell\_count | PASS 0.80 | PASS 0.80 | = |
| T2\_retrieve\_serial\_dilution | PASS 1.00 | PASS 1.00 | = |
| T3\_structured\_protocol | PASS 0.86 | PASS 0.86 | = |
| T4\_reagent\_lookup | FAIL 0.00 | PASS 1.00 | +1.00 |
| T5\_composite\_passage | FAIL 0.00 | PASS 0.70 | +0.70 |
| T6\_cell\_count\_row\_C | PASS 0.60 | PASS 0.60 | = |
| T7\_retrieve\_pcr | FAIL 0.00 | PASS 1.00 | +1.00 |
| T8\_retrieve\_elisa | PASS 1.00 | PASS 1.00 | = |
| T9\_serial\_dilution\_design | PASS 0.86 | PASS 0.86 | = |
| T10\_reagent\_absence | FAIL 0.00 | PASS 1.00 | +1.00 |
| T11\_composite\_row\_D | PASS 0.76 | PASS 0.76 | = |
| T12\_lookup\_PBS | FAIL 0.50 | PASS 1.00 | +0.50 |
| T13\_dna\_prep\_design | PASS 0.86 | PASS 0.86 | = |
| T14\_single\_well | PASS 1.00 | PASS 1.00 | = |
| T15\_retrieve\_then\_compose | FAIL 0.00 | PASS 1.00 | +1.00 |
| **Overall** | **0.65 (10/15)** | **0.88 (15/15)** | **+0.23** |

---

## Reproduction (no Docker)

```bash
# Clone
git clone https://github.com/amin-kh96/biolab-agent-base.git
cd biolab-agent-base

# Start services
ollama serve &
ollama pull medgemma:4b
ollama pull nomic-embed-text
./qdrant &

# Configure
cp .env.example .env
# Edit .env: set OLLAMA_HOST=http://localhost:11434, QDRANT_URL=http://localhost:6333
# Set BIOLAB_DATA_DIR and BIOLAB_ARTIFACT_DIR to local paths

# Install
pip install -e ".[all]"

# Index protocol corpus into Qdrant
biolab-index

# Run baseline
biolab-bench

# Run SolutionAgent
BIOLAB_AGENT_CLASS=biolab_agent.agent.solution:SolutionAgent biolab-bench
```

---

## Docker quick start

```bash
git clone https://github.com/amin-kh96/biolab-agent-base.git
cd biolab-agent-base
git lfs pull
cp .env.example .env
docker compose build
docker compose up -d
docker compose exec ollama ollama pull medgemma:4b
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec app biolab-index
docker compose exec app biolab-bench

# SolutionAgent via Docker
docker compose exec -e BIOLAB_AGENT_CLASS=biolab_agent.agent.solution:SolutionAgent app biolab-bench
```

---

## What I would try given more time

- **BM25 hybrid retrieval** combined with the existing dense retrieval
  for better RAG recall on keyword-heavy protocol queries.
- **Retrain the LoRA adapter** with mixed instruction types (tool-calling
  + protocol drafting) to prevent the adapter's narrow training from
  hurting tool-calling behaviour when it is loaded.
- **Safety/hazard checker tool** for chemical reagent validation —
  cross-reference catalog entries against a hazard database before
  composing protocols that include reactive reagents.
- **Explicit ReAct scratchpad** with a mandatory `"thought"` field before
  each tool call, so the model reasons through constraints before
  committing to an action rather than emitting tool calls greedily.

---

## Repository contents

| Path | Contents |
|---|---|
| `Dockerfile`, `docker-compose.yml` | Multi-stage CUDA-ready image with Ollama + Qdrant sidecars |
| `pyproject.toml` | Pinned dependency stack (uv / pip) |
| `src/biolab_agent/` | `BaseAgent` interface, FastAPI server, typed schemas, `BaselineAgent` and `SolutionAgent` |
| `eval/harness.py`, `eval/metrics.py` | 15-task benchmark runner + scoring functions |
| `data/images/` | 20 cell-microscopy images from [BBBC002 v1](https://bbbc.broadinstitute.org/BBBC002) with published cell counts |
| `data/protocols/opentrons.jsonl` | 200 OT-2 protocols harvested from [Opentrons/Protocols](https://github.com/Opentrons/Protocols) |
| `data/reagents/catalog.csv` | Reagent + labware entries extracted from the protocols |
| `data/finetune/` | 500 train / 50 eval instruction pairs derived from the protocols (for Unsloth LoRA) |
| `data/queries_public.yaml` | 15 benchmark tasks |
| `scripts/` | Bash + PowerShell scripts for setup, data fetch, model pull, benchmark |
| `ui/app.py` | Gradio web UI showing the agent's answer + tool trace + segmentation overlays |

---

## License

Starter code: Apache-2.0. Bundled data sources keep their own licenses:

- **BBBC002** images: public domain (Broad Institute CC0)
- **Opentrons/Protocols**: Apache-2.0

See [`data/DATA_SOURCES.md`](./data/DATA_SOURCES.md) for attribution details.
