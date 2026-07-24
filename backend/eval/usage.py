"""
Token / cache / latency tracker for a pipeline run.

Attach it as a LangChain callback and it accumulates, per model, the input/output tokens, the
cached input tokens (OpenAI prompt caching) and the reasoning tokens (GPT-5 thinking), plus the
call count and wall-clock time. Works for any provider that fills LangChain's standard
`usage_metadata`; the cache/reasoning breakdowns are populated by OpenAI.

Usage:
    from backend.eval.usage import UsageTracker

    with UsageTracker() as tracker:
        result = app.invoke(state, config={"callbacks": [tracker]})
    tracker.report()
"""
import time
from collections import defaultdict

from langchain_core.callbacks import BaseCallbackHandler


class UsageTracker(BaseCallbackHandler):
    def __init__(self) -> None:
        # model -> counters
        self.by_model: dict[str, dict] = defaultdict(
            lambda: {"calls": 0, "input": 0, "output": 0, "cached": 0, "reasoning": 0}
        )
        self._t0: float | None = None
        self.wall: float = 0.0

    # --- context manager: times the whole run ---
    def __enter__(self) -> "UsageTracker":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        if self._t0 is not None:
            self.wall = time.perf_counter() - self._t0

    # --- callback: one call finished ---
    def on_llm_end(self, response, **kwargs) -> None:
        model = (response.llm_output or {}).get("model_name") or "?"
        for gens in response.generations:
            for gen in gens:
                msg = getattr(gen, "message", None)
                um = getattr(msg, "usage_metadata", None) if msg is not None else None
                if not um:
                    continue
                m = self.by_model[model]
                m["calls"] += 1
                m["input"] += um.get("input_tokens", 0) or 0
                m["output"] += um.get("output_tokens", 0) or 0
                m["cached"] += (um.get("input_token_details") or {}).get("cache_read", 0) or 0
                m["reasoning"] += (um.get("output_token_details") or {}).get("reasoning", 0) or 0

    # --- report ---
    def report(self) -> dict:
        tot = {"calls": 0, "input": 0, "output": 0, "cached": 0, "reasoning": 0}
        for m in self.by_model.values():
            for k in tot:
                tot[k] += m[k]

        print("\n" + "=" * 78)
        print("TOKEN / CACHE / TIME")
        print("=" * 78)
        header = f"{'model':<14}{'calls':>6}{'input':>9}{'cached':>9}{'cache%':>8}{'output':>9}{'reason':>8}"
        print(header)
        print("-" * len(header))
        for model, m in sorted(self.by_model.items()):
            pct = (m["cached"] / m["input"] * 100) if m["input"] else 0.0
            print(f"{model[:14]:<14}{m['calls']:>6}{m['input']:>9}{m['cached']:>9}"
                  f"{pct:>7.1f}%{m['output']:>9}{m['reasoning']:>8}")
        print("-" * len(header))
        pct = (tot["cached"] / tot["input"] * 100) if tot["input"] else 0.0
        print(f"{'TOTAL':<14}{tot['calls']:>6}{tot['input']:>9}{tot['cached']:>9}"
              f"{pct:>7.1f}%{tot['output']:>9}{tot['reasoning']:>8}")

        billed_input = tot["input"] - tot["cached"]
        print(f"\n  input tokens        : {tot['input']:>8}  "
              f"({tot['cached']} cached @ discount, {billed_input} full price)")
        print(f"  output tokens       : {tot['output']:>8}  "
              f"(of which {tot['reasoning']} reasoning — not cacheable)")
        print(f"  prompt-cache hit    : {pct:>7.1f}%  of input tokens")
        print(f"  wall-clock          : {self.wall:>7.1f}s  over {tot['calls']} LLM calls")
        if tot["calls"]:
            print(f"  avg latency / call  : {self.wall / tot['calls']:>7.1f}s")
        return {"total": tot, "by_model": dict(self.by_model), "wall": self.wall}
