"""Goal-driven analysis agent — the fix for "the analysis is hardcoded".

v3.3.0 (Phase A). Until now the analysis page ran the SAME fixed computations no
matter what the user asked; the goal only flavored the prose. This module turns
analysis into an agent: an LLM **planner** reads the user's goal + the dataset's
real column profile and **chooses which tools to run, with what parameters** from a
grounded tool library. Different goal → different plan → different analysis. The
LLM never invents numbers — every value comes from a tool that actually computes on
the data (the LLM is the brain; the tools are the hands and eyes).

Pure/stdlib + reuse of sibling modules. `plan()` takes an injected `llm_call` so it
is model-agnostic and unit-testable (mock in tests, real gateway in the server).
Phase B (conversation loop) and a sandboxed code-writing tool come later.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# dataset context — parse sensor tables once; tools operate on this.
# ---------------------------------------------------------------------------
_TIME_NAMES = ("t", "time", "timestamp", "ts", "sec", "seconds", "millis", "ms")


def load_context(root, max_rows: int = 20000) -> dict:
    """Parse csv/tsv sensor files under `root` into a context tools can use:
    {files:[{name, cols:{col:[float]}, header, time_col, t_start_abs}], signals:[...]}"""
    root = Path(root)
    tables = [p for p in sorted(root.rglob("*"))
              if p.is_file() and p.suffix.lower() in (".csv", ".tsv")
              and "labels" not in {x.lower() for x in p.parts}]
    files, signals = [], []
    for p in tables[:24]:
        try:
            delim = "\t" if p.suffix.lower() == ".tsv" else ","
            with open(p, newline="", encoding="utf-8", errors="replace") as f:
                rd = csv.reader(f, delimiter=delim)
                header = [str(h).strip() for h in next(rd, [])]
                cols = {h: [] for h in header}
                for i, row in enumerate(rd):
                    if i >= max_rows:
                        break
                    for j, val in enumerate(row):
                        if j < len(header):
                            try:
                                cols[header[j]].append(float(val))
                            except (ValueError, TypeError):
                                pass
            cols = {k: v for k, v in cols.items() if v}
            if not cols:
                continue
            low = [h.lower() for h in header]
            tcol = next((header[i] for i, h in enumerate(low)
                         if h in _TIME_NAMES and header[i] in cols), None)
            t0 = cols[tcol][0] if tcol and cols[tcol] else None
            if tcol and cols[tcol] and cols[tcol][0] > 1e6:      # unix → s-from-start
                cols[tcol] = [t - cols[tcol][0] for t in cols[tcol]]
            files.append({"name": p.name, "cols": cols, "header": header,
                          "time_col": tcol, "t_start_abs": t0})
            for c in cols:
                if c != tcol:
                    signals.append(f"{p.name}:{c}")
        except Exception:
            continue
    return {"files": files, "signals": signals}


def profile_text(ctx: dict) -> str:
    """A compact, honest description of what's in the data — fed to the planner."""
    lines = []
    for f in ctx["files"]:
        n = max((len(v) for v in f["cols"].values()), default=0)
        sig = [c for c in f["cols"] if c != f["time_col"]]
        lines.append(f"- {f['name']}: {n} rows, time_col={f['time_col'] or 'none'}, "
                     f"signals={sig}")
    return "\n".join(lines) or "(no tabular sensor files)"


# ---------------------------------------------------------------------------
# tool library — each tool actually computes on ctx and returns grounded data.
# ---------------------------------------------------------------------------
def _all_signals(ctx, names=None):
    """Resolve requested signal names (bare or file:col) → [(file_dict, col)]."""
    out = []
    for f in ctx["files"]:
        for c in f["cols"]:
            if c == f["time_col"]:
                continue
            if names:
                want = any(c == n or f"{f['name']}:{c}" == n or n == c for n in names)
                if not want:
                    continue
            out.append((f, c))
    return out


def _t_tool_noise(ctx, signals=None, **_):
    from .dataset_eda import _signal_noise
    rows = []
    for f, c in _all_signals(ctx, signals):
        r = _signal_noise(f["cols"][c])
        if r:
            rows.append({"file": f["name"], "signal": c,
                         "noise_pct": r.get("noise_pct"), "snr_db": r.get("snr_db")})
    rows.sort(key=lambda x: -(x["noise_pct"] or 0))
    return {"kind": "noise", "title": "Signal noise / SNR",
            "rows": rows, "worst": rows[0] if rows else None}


def _t_tool_anomalies(ctx, types=None, **_):
    from .sensor_anomaly import detect_table
    files = []
    for f in ctx["files"]:
        r = detect_table(f["cols"], f["header"], f["time_col"], None)
        if r:
            evs = r["events"]
            if types:
                evs = [e for e in evs if e["type"] in types]
            files.append({"file": f["name"], "n": len(evs),
                          "by_type": r["by_type"], "events": evs[:12]})
    total = sum(x["n"] for x in files)
    return {"kind": "anomalies", "title": "Timestamped anomalies",
            "total": total, "files": files}


def _t_tool_cross(ctx, window_s: float = 1.0, **_):
    from .sensor_anomaly import detect_table
    from .sensor_align import build_alignment
    anom = []
    for f in ctx["files"]:
        r = detect_table(f["cols"], f["header"], f["time_col"], None)
        if r:
            r["file"] = f["name"]
            r["t_start_abs"] = f["t_start_abs"]
            anom.append(r)
    al = build_alignment(anom, window_s=float(window_s)) if anom else None
    if not al:
        return {"kind": "cross", "title": "Cross-sensor correlated moments",
                "n_correlated": 0, "note": "needs ≥2 sensor files to correlate"}
    return {"kind": "cross", "title": "Cross-sensor correlated moments",
            "n_correlated": al["n_correlated"], "basis": al["basis"],
            "correlated": al["correlated"]}


def _t_tool_segment_turns(ctx, heading_col=None, turn_rate=None, **_):
    """NEW: split the run into TURN vs STRAIGHT by heading-change rate, and report
    per-segment behaviour. Answers 'I care about the corners, not the straights.'"""
    import math
    target = None
    for f in ctx["files"]:
        for c in f["cols"]:
            if heading_col and c != heading_col:
                continue
            if any(k in c.lower() for k in ("heading", "yaw", "bearing", "course")):
                target = (f, c); break
        if target:
            break
    if not target:
        return {"kind": "segments", "title": "Turns vs straights",
                "note": "no heading/yaw column found to segment on"}
    f, c = target
    hd = f["cols"][c]
    n = len(hd)
    rate = []
    for i in range(1, n):
        d = ((hd[i] - hd[i - 1] + 180) % 360) - 180
        rate.append(abs(d))
    thr = float(turn_rate) if turn_rate else (sorted(rate)[int(len(rate) * 0.85)] if rate else 0)
    turn_idx = [i + 1 for i, r in enumerate(rate) if r > thr]
    frac = len(turn_idx) / max(n, 1)

    def _seg_stats(idxs):
        out = {}
        for f2 in ctx["files"]:
            for c2 in f2["cols"]:
                if c2 == f2["time_col"]:
                    continue
                vals = [f2["cols"][c2][i] for i in idxs if i < len(f2["cols"][c2])]
                if len(vals) > 2:
                    m = sum(vals) / len(vals)
                    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
                    out[f"{f2['name']}:{c2}"] = {"mean": round(m, 3), "std": round(sd, 3)}
        return out
    straight_idx = [i for i in range(n) if i not in set(turn_idx)]
    return {"kind": "segments", "title": "Turns vs straights",
            "segmented_on": f"{f['name']}:{c}", "turn_rate_thresh": round(thr, 2),
            "turn_fraction": round(frac, 3),
            "n_turn_samples": len(turn_idx), "n_straight_samples": len(straight_idx),
            "in_turns": _seg_stats(turn_idx), "in_straights": _seg_stats(straight_idx)}


def _t_tool_stats(ctx, columns=None, **_):
    import math
    rows = []
    for f, c in _all_signals(ctx, columns):
        v = f["cols"][c]
        m = sum(v) / len(v)
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / len(v))
        rows.append({"file": f["name"], "signal": c, "min": round(min(v), 4),
                     "mean": round(m, 4), "max": round(max(v), 4), "std": round(sd, 4)})
    return {"kind": "stats", "title": "Summary statistics", "rows": rows}


def _t_tool_plot_route(ctx, color_by=None, out_png=None, **_):
    from .sensor_viz import plot_dataset_sensors  # trajectory when lat/lon present
    if not out_png:
        return {"kind": "plot", "title": "Route", "note": "no output path"}
    r = None
    # plot the first GPS-like file directly
    from .sensor_viz import plot_sensor_table
    for f in ctx["files"]:
        low = {c.lower() for c in f["cols"]}
        if ({"lat"} & low or {"latitude"} & low) and ({"lon", "lng", "longitude"} & low):
            # write a temp CSV? simpler: reuse dataset-level plotter on the root later.
            pass
    return {"kind": "plot", "title": "Route shape", "note": "rendered on the page",
            "hint": "trajectory"}


# name -> (fn, description, params-doc)
TOOLS = {
    "signal_noise": (_t_tool_noise,
        "How clean each signal is: residual RMS after smoothing as % of range, plus SNR dB. Use for reliability / 'how noisy' questions.",
        {"signals": "list of signal names to check, or omit for all"}),
    "detect_anomalies": (_t_tool_anomalies,
        "Timestamped abnormal events: sudden changes, GPS teleports, sampling gaps, stuck-sensor flatlines. Use for 'what went wrong / when'.",
        {"types": "subset of [sudden_change,gps_jump,time_gap,flatline], or omit for all"}),
    "cross_sensor_correlation": (_t_tool_cross,
        "Moments where TWO OR MORE sensors flagged the same instant (real physical events vs single-sensor glitches). Needs ≥2 sensor files.",
        {"window_s": "coincidence window in seconds (default 1.0)"}),
    "segment_turns_vs_straight": (_t_tool_segment_turns,
        "Split the run into cornering vs straight-line segments by heading-change rate and compare per-segment behaviour. Use when the user cares about turns/corners specifically.",
        {"heading_col": "heading column name (auto if omitted)", "turn_rate": "deg/step threshold (auto = 85th pct)"}),
    "summary_stats": (_t_tool_stats,
        "min / mean / max / std per numeric signal. Use for a quick quantitative overview.",
        {"columns": "list of columns, or omit for all"}),
    "plot_route": (_t_tool_plot_route,
        "Draw the GPS route's shape (needs lat/lon). Use when spatial path matters.",
        {"color_by": "signal to color the path by, e.g. speed"}),
}


def tool_menu() -> list:
    return [{"name": k, "description": d, "params": p} for k, (_, d, p) in TOOLS.items()]


# ---------------------------------------------------------------------------
# planner — LLM picks tools+params from the goal. `llm_call(prompt,system)->str`.
# ---------------------------------------------------------------------------
_PLAN_SYS = (
    "You are a data-analysis planner. Given a dataset's real columns and the user's "
    "GOAL, choose which analysis tools to run and with what parameters. Pick only "
    "tools that serve THIS goal — do not run everything. Return STRICT JSON: "
    '{"plan":[{"tool":"<name>","params":{...},"why":"<one short reason>"}]}. '
    "Use 1-4 tools. Only use tool names from the provided list."
)


def plan(goal: str, ctx: dict, llm_call) -> dict:
    """Ask the planner LLM for a tool plan tailored to `goal`. Returns
    {ok, plan:[{tool,params,why}], raw, source}. Falls back to a heuristic plan
    if the model is unavailable or returns unusable output."""
    menu = tool_menu()
    prompt = (f"GOAL: {goal.strip() or '(none given — do a sensible general read)'}\n\n"
              f"DATASET COLUMNS:\n{profile_text(ctx)}\n\n"
              f"AVAILABLE TOOLS:\n{json.dumps(menu, indent=1)}\n\n"
              "Return the JSON plan now.")
    raw, source = "", "llm"
    try:
        raw = llm_call(prompt, _PLAN_SYS) or ""
    except Exception as e:
        raw, source = f"(llm error: {e})", "error"
    parsed = _extract_plan(raw)
    if not parsed:
        parsed, source = _heuristic_plan(goal, ctx), "heuristic"
    # keep only known tools
    parsed = [s for s in parsed if s.get("tool") in TOOLS][:4]
    if not parsed:
        parsed, source = _heuristic_plan(goal, ctx), "heuristic"
    return {"ok": True, "plan": parsed, "raw": raw[:1200], "source": source}


def _extract_plan(raw: str) -> list:
    import re
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return []
    steps = obj.get("plan") if isinstance(obj, dict) else obj
    out = []
    for s in (steps or []):
        if isinstance(s, dict) and s.get("tool"):
            out.append({"tool": str(s["tool"]).strip(),
                        "params": s.get("params") or {},
                        "why": str(s.get("why", ""))[:160]})
    return out


def _heuristic_plan(goal: str, ctx: dict) -> list:
    """Keyword fallback so the feature degrades gracefully without a model."""
    g = (goal or "").lower()
    steps = []
    if any(w in g for w in ("turn", "corner", "curve", "bend")):
        steps.append({"tool": "segment_turns_vs_straight", "params": {}, "why": "goal mentions turns"})
    if any(w in g for w in ("noise", "clean", "reliab", "quality", "snr")):
        steps.append({"tool": "signal_noise", "params": {}, "why": "goal mentions noise/quality"})
    if any(w in g for w in ("correlat", "together", "same", "agree", "cross", "both")):
        steps.append({"tool": "cross_sensor_correlation", "params": {}, "why": "goal mentions cross-sensor"})
    if any(w in g for w in ("anomal", "fault", "wrong", "error", "glitch", "spike", "event")):
        steps.append({"tool": "detect_anomalies", "params": {}, "why": "goal mentions anomalies"})
    if not steps:                                   # sensible default read
        steps = [{"tool": "summary_stats", "params": {}, "why": "general overview"},
                 {"tool": "detect_anomalies", "params": {}, "why": "default health check"}]
    return steps[:4]


def run_plan(plan_steps: list, ctx: dict) -> list:
    """Execute each step's tool on ctx; collect grounded results. Never raises."""
    results = []
    for s in plan_steps:
        fn = TOOLS.get(s["tool"], (None,))[0]
        if not fn:
            continue
        try:
            r = fn(ctx, **(s.get("params") or {}))
            r["tool"] = s["tool"]
            r["why"] = s.get("why", "")
            results.append(r)
        except Exception as e:
            results.append({"tool": s["tool"], "kind": "error", "error": str(e)[:200]})
    return results
