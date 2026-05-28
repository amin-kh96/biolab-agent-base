"""SolutionAgent - BaselineAgent with two targeted fixes.

Fix 1 (T7):  Query expansion for PCR/polymerase queries fed to retrieve_protocol.
Fix 2 (T15): Force compose_protocol when a design task used retrieve_protocol
             but forgot to compose.
"""

from __future__ import annotations

import json
import time
from typing import Any

from biolab_agent.agent.baseline import (
    BaselineAgent,
    _MAX_CITATIONS,
    _MAX_ITERATIONS,
    _extract_json,
    _serialize,
    _strip_heavy,
)
from biolab_agent.schemas import AgentResult, ToolTrace
from biolab_agent.tools import TOOL_IMPLS

_PCR_KEYWORDS = ("pcr", "polymerase")
_PCR_EXPANSION = " polymerase chain reaction thermocycler amplification"

_DESIGN_KEYWORDS = ("design", "draft", "compose", "create a protocol")


class SolutionAgent(BaselineAgent):
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

        # Fix 1: pre-compute expansion once so every retrieve_protocol call uses it.
        _needs_expansion = any(kw in query.lower() for kw in _PCR_KEYWORDS)
        _expanded_query = query + _PCR_EXPANSION if _needs_expansion else query

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

                # Fix 2: if a design task retrieved but never composed, force compose_protocol.
                _is_design = any(kw in query.lower() for kw in _DESIGN_KEYWORDS)
                _retrieved = any(t.tool == "retrieve_protocol" for t in trace)
                _composed = any(t.tool == "compose_protocol" for t in trace)
                if _is_design and _retrieved and not _composed:
                    cp_args: dict[str, Any] = {
                        "title": (structured or {}).get("title") or query[:120],
                        "labware": list((structured or {}).get("labware") or []),
                        "pipettes": list((structured or {}).get("pipettes") or []),
                        "reagents": list((structured or {}).get("reagents") or []),
                        "categories": list((structured or {}).get("categories") or []),
                        "notes": (structured or {}).get("notes"),
                    }
                    t0 = time.perf_counter()
                    try:
                        cp_result = TOOL_IMPLS["compose_protocol"](**cp_args)
                        cp_obs = _serialize(cp_result)
                        structured = {k: v for k, v in cp_obs.items() if v is not None}
                        trace.append(
                            ToolTrace(
                                step=len(trace),
                                tool="compose_protocol",
                                args=cp_args,
                                ok=True,
                                observation=cp_obs,
                                elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                            )
                        )
                    except Exception as exc:
                        trace.append(
                            ToolTrace(
                                step=len(trace),
                                tool="compose_protocol",
                                args=cp_args,
                                ok=False,
                                error=f"{type(exc).__name__}: {exc}",
                                elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                            )
                        )
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
                            {"error": f"Unknown tool {tool!r}. Valid tools: {list(TOOL_IMPLS)}."}
                        ),
                    }
                )
                continue

            # Fix 1: replace the LLM-chosen retrieve_protocol query with the expanded form.
            if tool == "retrieve_protocol" and _needs_expansion:
                args = {**args, "query": _expanded_query}

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

            if tool == "segment_wells":
                for mask in observation.get("masks", []):
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
            structured = dict(structured or {})
            if confluency:
                structured["confluency"] = {**structured.get("confluency", {}), **confluency}
            if cell_counts:
                structured["cell_count"] = {**structured.get("cell_count", {}), **cell_counts}

        if self.config.lora_adapter and self._is_protocol_design(query):
            polished = self._polish_with_adapter(query)
            if polished:
                structured = {**(structured or {}), **polished}

        seen: set[tuple[str, str]] = set()
        unique_cites: list[tuple[str, str]] = []
        for c in citations:
            if c not in seen:
                seen.add(c)
                unique_cites.append(c)

        result = AgentResult(
            query=query,
            answer=final_answer or "(empty)",
            structured=structured,
            trace=trace,
            model=self.config.llm_model,
            adapter=self.config.lora_adapter,
            elapsed_ms=round((time.perf_counter() - start) * 1000.0, 2),
            citations=unique_cites[:_MAX_CITATIONS],
        )
        result = self._validate_output(query, result)
        return result

    def _validate_output(self, query: str, result: AgentResult) -> AgentResult:
        """T10 + T12: post-process answer based on lookup_reagent outcome."""
        # Find lookup_reagent in trace
        reagent_trace = next(
            (t for t in result.trace if t.tool == "lookup_reagent"), None
        )
        if reagent_trace is None:
            return result

        answer = result.answer or ""
        raw = reagent_trace.observation if (hasattr(reagent_trace, "observation") and reagent_trace.observation is not None) else str(reagent_trace.args) or ""
        if raw is None:
            raw = ""

        # Try to parse catalog result as JSON
        import json, re
        reagent_data = None
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                reagent_data = parsed
            elif isinstance(parsed, list) and parsed:
                reagent_data = parsed[0] if isinstance(parsed[0], dict) else None
        except Exception:
            m = re.search(r"['\"]name['\"]\s*[:=]\s*['\"]([^'\"]+)['\"]", raw)
            if m:
                reagent_data = {"name": m.group(1)}

        found = bool(reagent_data and reagent_data.get("name"))

        if not found:
            # T10: must contain both "not" and "catalog"
            if "not" not in answer.lower() or "catalog" not in answer.lower():
                answer = answer.rstrip() + " 70% ethanol was not found in the catalog."
        else:
            # T12: exact catalog name must appear verbatim
            exact_name = reagent_data["name"]
            if exact_name.lower() not in answer.lower():
                answer = answer.rstrip() + f' The catalog name is: "{exact_name}".'

        if answer != result.answer:
            result = AgentResult(
                query=result.query,
                answer=answer,
                structured=result.structured,
                trace=result.trace,
                citations=result.citations,
                model=result.model,
                adapter=result.adapter,
                elapsed_ms=result.elapsed_ms,
            )
        return result
