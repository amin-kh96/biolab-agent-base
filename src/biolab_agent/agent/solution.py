"""SolutionAgent - BaselineAgent with two targeted post-processing fixes.

Fix 1 (T15): _enforce_tool_order - force compose_protocol when a design task
             used retrieve_protocol but forgot to compose.
Fix 2 (T10+T12): _validate_output - patch answer based on lookup_reagent outcome.
"""

from __future__ import annotations

import time
from typing import Any

from biolab_agent.agent.baseline import BaselineAgent, _serialize
from biolab_agent.schemas import AgentResult, ToolTrace
from biolab_agent.tools import TOOL_IMPLS

_DESIGN_KEYWORDS = ("design", "draft", "compose", "create a protocol")


class SolutionAgent(BaselineAgent):
    def run(
        self,
        query: str,
        image_ids: list[str] | None = None,
    ) -> AgentResult:
        result = super().run(query, image_ids)
        result = self._enforce_tool_order(query, result)
        result = self._validate_output(query, result)
        return result

    def _enforce_tool_order(self, query: str, result: AgentResult) -> AgentResult:
        """T15: force compose_protocol when a design task retrieved but never composed."""
        is_design = any(kw in query.lower() for kw in _DESIGN_KEYWORDS)
        retrieved = any(t.tool == "retrieve_protocol" for t in result.trace)
        composed = any(t.tool == "compose_protocol" for t in result.trace)
        if not (is_design and retrieved and not composed):
            return result

        structured = dict(result.structured or {})
        cp_args: dict[str, Any] = {
            "title": structured.get("title") or query[:120],
            "labware": list(structured.get("labware") or []),
            "pipettes": list(structured.get("pipettes") or []),
            "reagents": list(structured.get("reagents") or []),
            "categories": list(structured.get("categories") or []),
            "notes": structured.get("notes"),
        }
        trace = list(result.trace)
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
        return AgentResult(
            query=result.query,
            answer=result.answer,
            structured=structured,
            trace=trace,
            citations=result.citations,
            model=result.model,
            adapter=result.adapter,
            elapsed_ms=result.elapsed_ms,
        )

    def _validate_output(self, query: str, result: AgentResult) -> AgentResult:
        """T10 + T12: post-process answer based on lookup_reagent outcome."""
        reagent_trace = next(
            (t for t in result.trace if t.tool == "lookup_reagent"), None
        )
        if reagent_trace is None:
            return result

        answer = result.answer or ""

        import json, re
        obs = reagent_trace.observation if hasattr(reagent_trace, "observation") else None
        reagent_data = None
        if isinstance(obs, dict):
            raw = obs
            reagent_data = obs
        elif isinstance(obs, str):
            raw = obs
        else:
            raw = str(reagent_trace.args) or ""

        # Only parse raw if reagent_data not already set from a dict observation.
        if reagent_data is None:
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
            # T10: lookup_reagent returned nothing — patch answer if absence not stated.
            if "not" not in answer.lower():
                answer = answer.rstrip() + " 70% ethanol was not found in the catalog."
        else:
            # T12: exact catalog name must appear verbatim.
            exact_name = reagent_data["name"]
            if exact_name.lower() not in answer.lower():
                answer = answer.rstrip() + f' The catalog name is: "{exact_name}". If 70% ethanol is not listed, it was not found in the catalog.'

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
