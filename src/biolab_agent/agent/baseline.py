"""Reference agent.

Uses prompt-driven JSON tool-calling so it works across Ollama models
regardless of whether native tool-calling is wired up for a given tag.
Each LLM turn emits one JSON object: either a tool call or a final answer.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from biolab_agent.agent.base import BaseAgent
from biolab_agent.llm import get_client
from biolab_agent.schemas import AgentResult, ToolTrace
from biolab_agent.tools import TOOL_IMPLS, TOOL_SPECS

_MAX_ITERATIONS = 10
_MAX_CITATIONS = 6

# Heuristic: when the user is asking us to "design/compose/draft/create ...
# protocol", scope the fine-tuned adapter to polishing the structured output.
# Everywhere else we use the base model so tool-calling JSON stays intact.
_PROTOCOL_DESIGN_KEYWORDS = (
    "design an",
    "design a",
    "compose a",
    "compose an",
    "draft a protocol",
    "draft an",
    "create a protocol",
    "design a protocol",
    "write a protocol",
    "generate a protocol",
    "structured protocol",
    "produce the protocol",
    "convert this experiment",
)


_SYSTEM_PROMPT_TEMPLATE = """You are an autonomous laboratory agent. You help plan and run experiments \
using the tools listed below. Think step by step and call tools to gather evidence before answering.

AVAILABLE TOOLS:
{tool_summaries}

PROTOCOL
Every turn you output EXACTLY one JSON object  -  nothing else, no prose, no markdown fences.
Exactly one of these two shapes:

  {{"tool": "<tool_name>", "arguments": {{ ... }}}}
      Call a tool. Use the argument names from the tool's schema.

  {{"final": "<natural-language answer>",
    "structured": {{ ... }},
    "citations": [["<doc_id>", "<chunk_id>"], ...]}}
      Return the final answer. ALWAYS include `structured` (may be empty) and `citations` (may be empty).

CRITICAL RULES
- Never include prose outside the JSON. No ```json fences.
- When the user provides `Available image_ids: [...]`, you MUST call segment_wells ONCE PER image_id,
  then return `{{"final": "...", "structured": {{"cell_count": {{"<image_id>": <n>}}, "confluency": {{"<image_id>": <pct>}}}}}}`.
  Use the tool's `cell_count` and `confluency_pct` fields VERBATIM  -  do not round or invent values.
- When the user asks you to design / compose / draft a protocol, you MUST call `compose_protocol`
  exactly once (no need to retrieve beforehand) with at least: `title`, `labware`, `pipettes`,
  `reagents`. Then return `{{"final": "...", "structured": <compose_protocol output>}}`.
  compose_protocol is REQUIRED for any protocol-design task  -  do not end without calling it.
- For retrieval-style tasks (the user asks you to find/retrieve a protocol), call retrieve_protocol
  ONCE and return `{{"final": "...", "citations": [[doc_id, chunk_id], ...]}}` using the hits.
- Do NOT call retrieve_protocol more than twice for a single task.
- Do NOT invent data. If lookup_reagent returns null, say so explicitly.
- Stop after at most {max_iter} tool calls.
"""


def _format_tool_summary(specs: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for spec in specs:
        fn = spec["function"]
        props = fn["parameters"].get("properties", {})
        args_desc = ", ".join(f"{k}:{props[k].get('type', 'any')}" for k in props)
        lines.append(f"- {fn['name']}({args_desc})  -  {fn['description']}")
    return "\n".join(lines)


# Accept the model wrapping its JSON in ```json fences or prose prefixes.
_JSON_PATTERNS = (
    re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL),
    re.compile(r"(\{.*\})", re.DOTALL),
)


def _extract_json(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    for pat in _JSON_PATTERNS:
        m = pat.search(raw)
        if not m:
            continue
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
    return None


def _serialize(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, list):
        return [_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


def _strip_heavy(obj: Any) -> Any:
    """Drop large byproducts (RLE masks, long text) before echoing observation to the LLM."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "rle":
                continue
            if k == "text" and isinstance(v, str) and len(v) > 400:
                out[k] = v[:400] + "…"
            else:
                out[k] = _strip_heavy(v)
        return out
    if isinstance(obj, list):
        return [_strip_heavy(x) for x in obj]
    return obj


class BaselineAgent(BaseAgent):
    def __init__(self, config) -> None:  # type: ignore[no-untyped-def]
        super().__init__(config)
        self._llm = get_client()
        self._adapter_llm = None  # lazy: only constructed on protocol-design tasks
        self._system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            tool_summaries=_format_tool_summary(TOOL_SPECS),
            max_iter=_MAX_ITERATIONS,
        )

    def _is_protocol_design(self, query: str) -> bool:
        low = query.lower()
        return any(kw in low for kw in _PROTOCOL_DESIGN_KEYWORDS)

    def _polish_with_adapter(self, query: str) -> dict[str, Any] | None:
        """Generate a clean structured protocol with the LoRA-adapted model.

        Called once at the end of a protocol-design task. The adapter was
        trained on (description → structured JSON) pairs; using it here is
        the one place where its over-specialisation is actually wanted.
        """
        if not self.config.lora_adapter:
            return None
        if self._adapter_llm is None:
            # Free Ollama's base-model VRAM before loading the HF+adapter
            # instance  -  laptop GPUs (8 GB) can't hold both at once.
            try:
                import httpx

                httpx.post(
                    f"{self.config.ollama_host}/api/generate",
                    json={"model": self.config.llm_model, "keep_alive": 0},
                    timeout=3.0,
                )
            except Exception:
                pass
            from biolab_agent.llm.hf_client import HFChatClient

            self._adapter_llm = HFChatClient(adapter_path=self.config.lora_adapter)

        instruction = (
            "Please convert this experiment brief into a structured protocol "
            "definition with keys `title`, `categories`, `labware`, `pipettes`, "
            "`reagents`. Return only the JSON object."
        )
        try:
            resp = self._adapter_llm.chat(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": f"{instruction}\n\n{query}"}],
                options={"temperature": 0.0, "num_predict": 400, "top_p": 0.9},
            )
            parsed = _extract_json(resp["message"]["content"])
        finally:
            # 4 GB of VRAM  -  free it now so later tasks (segmentation,
            # Ollama reload) have room on the 8 GB laptop GPU.
            self._unload_adapter_llm()
        if not isinstance(parsed, dict):
            return None
        # Only keep the protocol-shape fields  -  don't leak model artefacts.
        return {
            k: parsed[k]
            for k in ("title", "categories", "labware", "pipettes", "reagents", "notes")
            if k in parsed
        }

    def _unload_adapter_llm(self) -> None:
        if self._adapter_llm is None:
            return
        # Drop our own reference first so gc has a chance.
        self._adapter_llm = None
        try:
            import gc

            import torch

            # The HFChatClient's cached (model, tokenizer, device) tuple is
            # pinned by an lru_cache on the module-level `_load_model`. Clear
            # it so Python + CUDA drop the weights.
            from biolab_agent.llm.hf_client import _load_model

            _load_model.cache_clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def run(
        self,
        query: str,
        image_ids: list[str] | None = None,
    ) -> AgentResult:
        start = time.perf_counter()
        trace: list[ToolTrace] = []
        confluency: dict[str, float] = {}
        cell_counts: dict[str, int] = {}
        citations: list[tuple[str, str]] = []
        structured: dict[str, Any] | None = None
        final_answer = ""

        user_content = query
        if image_ids:
            user_content += f"\n\nAvailable image_ids: {list(image_ids)}"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]

        for step in range(_MAX_ITERATIONS):
            resp = self._llm.chat(
                model=self.config.llm_model,
                messages=messages,
                options={"temperature": 0.1, "num_predict": 600, "top_p": 0.9},
            )
            assistant_raw = resp["message"]["content"]
            messages.append({"role": "assistant", "content": assistant_raw})
            parsed = _extract_json(assistant_raw)
            if parsed is None:
                # Model failed to produce JSON  -  stop and keep whatever we have.
                final_answer = assistant_raw.strip() or "Agent produced no parseable output."
                break

            if "final" in parsed:
                final_answer = str(parsed.get("final", "")).strip()
                if isinstance(parsed.get("structured"), dict):
                    structured = dict(parsed["structured"])
                raw_cites = parsed.get("citations") or []
                for c in raw_cites:
                    if (isinstance(c, list | tuple)) and len(c) >= 2:
                        citations.append((str(c[0]), str(c[1])))
                break

            tool = parsed.get("tool")
            args = parsed.get("arguments") or parsed.get("args") or {}
            if not tool or tool not in TOOL_IMPLS:
                trace.append(
                    ToolTrace(
                        step=step,
                        tool=str(tool or "<unknown>"),
                        args=args,
                        ok=False,
                        error=f"Unknown tool {tool!r}",
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(
                            {
                                "error": f"Unknown tool {tool!r}. Valid tools: {list(TOOL_IMPLS)}.",
                            }
                        ),
                    }
                )
                continue

            t0 = time.perf_counter()
            try:
                result = TOOL_IMPLS[tool](**args)
                observation = _serialize(result)
                trace.append(
                    ToolTrace(
                        step=step,
                        tool=tool,
                        args=args,
                        ok=True,
                        observation=observation,
                        elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                    )
                )
            except Exception as exc:
                trace.append(
                    ToolTrace(
                        step=step,
                        tool=tool,
                        args=args,
                        ok=False,
                        error=f"{type(exc).__name__}: {exc}",
                        elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps({"tool": tool, "error": str(exc)}),
                    }
                )
                continue

            # Auto-aggregate well-known results so the final structured output
            # is close to what the harness expects even if the model forgets.
            if tool == "segment_wells":
                for mask in observation.get("masks", []):
                    # Prefer the requested image_id over the per-mask well_id
                    # (they coincide today but may diverge for multi-well plates).
                    wid = args.get("image_id") or mask.get("well_id")
                    if wid:
                        confluency[str(wid)] = float(mask.get("confluency_pct", 0.0))
                        cc = mask.get("cell_count")
                        if cc is not None:
                            cell_counts[str(wid)] = int(cc)
            elif tool == "retrieve_protocol":
                for hit in observation[:_MAX_CITATIONS]:
                    citations.append((str(hit.get("doc_id", "")), str(hit.get("chunk_id", ""))))
            elif tool == "compose_protocol":
                structured = {k: v for k, v in observation.items() if v is not None}

            # Strip heavy byproducts (RLE strings) before feeding back to the LLM  -
            # they are not useful for planning and blow up the context.
            light_observation = _strip_heavy(observation)
            messages.append(
                {
                    "role": "tool",
                    "content": json.dumps({"tool": tool, "observation": light_observation})[:6000],
                }
            )
        else:
            final_answer = (
                final_answer or "Max iterations reached before the agent emitted a final answer."
            )

        if confluency or cell_counts:
            # Real tool outputs beat whatever the LLM put in structured  -
            # small models tend to hallucinate plausible-looking numbers
            # and forget to use the measured values.
            structured = dict(structured or {})
            if confluency:
                structured["confluency"] = {**structured.get("confluency", {}), **confluency}
            if cell_counts:
                structured["cell_count"] = {**structured.get("cell_count", {}), **cell_counts}

        # Adapter scoping: polish the structured protocol only when the task
        # actually asks for a designed protocol. Avoids the adapter's narrow
        # training hurting tool-calling behaviour on the other task kinds.
        if self.config.lora_adapter and self._is_protocol_design(query):
            polished = self._polish_with_adapter(query)
            if polished:
                structured = {**(structured or {}), **polished}

        # Deduplicate citations while preserving order.
        seen: set[tuple[str, str]] = set()
        unique_cites: list[tuple[str, str]] = []
        for c in citations:
            if c not in seen:
                seen.add(c)
                unique_cites.append(c)

        return AgentResult(
            query=query,
            answer=final_answer or "(empty)",
            structured=structured,
            trace=trace,
            model=self.config.llm_model,
            adapter=self.config.lora_adapter,
            elapsed_ms=round((time.perf_counter() - start) * 1000.0, 2),
            citations=unique_cites[:_MAX_CITATIONS],
        )
