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
def _num(v, default=None):
    """Coerce a possibly-bad LLM param to float; fall back on anything non-numeric
    (the planner sometimes emits things like '85th pct' or 'auto')."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


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
    al = build_alignment(anom, window_s=(_num(window_s, 1.0) or 1.0)) if anom else None
    if not al:
        return {"kind": "cross", "title": "Cross-sensor correlated moments",
                "n_correlated": 0, "note": "needs ≥2 sensor files to correlate"}
    return {"kind": "cross", "title": "Cross-sensor correlated moments",
            "n_correlated": al["n_correlated"], "basis": al["basis"],
            "correlated": al["correlated"]}


def _t_tool_segment_turns(ctx, heading_col=None, turn_percentile=None, turn_rate=None, **_):
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
    # threshold = the Nth-percentile heading-change (top ~(100-N)% sharpest = turns).
    # Percentile speaks the planner's language; a raw deg/step turn_rate is honored
    # only if it yields a sane split, else we fall back to the percentile.
    if not rate:
        return {"kind": "segments", "title": "Turns vs straights", "note": "series too short"}
    pct = _num(turn_percentile, 85.0)
    pct = min(99.0, max(50.0, pct if pct is not None else 85.0))
    auto_thr = sorted(rate)[min(len(rate) - 1, int(len(rate) * pct / 100.0))]
    thr = _num(turn_rate)
    if thr is None or not (0 < sum(1 for r in rate if r > thr) < len(rate) * 0.5):
        thr = auto_thr                                      # bad/degenerate → auto
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


def _t_tool_focus_time(ctx, center=None, radius_s=3.0, time=None, t=None, **_):
    """Zoom into a moment: what each signal was doing in [t-rad, t+rad] (window mean
    vs its overall mean), plus any anomalies / cross-sensor moments inside it. This is
    the conversational drill-down after 'something happened at t=X'."""
    c = _num(center)
    if c is None:
        c = _num(time)
    if c is None:
        c = _num(t)
    rad = _num(radius_s, 3.0) or 3.0
    if c is None:
        return {"kind": "focus", "title": "Focus on a time",
                "note": "give a time in seconds (from the start of the run) to zoom into"}
    per = []
    for f in ctx["files"]:
        tcol = f["time_col"]
        if not tcol or tcol not in f["cols"]:
            continue
        times = f["cols"][tcol]
        idx = [i for i, tv in enumerate(times) if abs(tv - c) <= rad]
        if not idx:
            continue
        sig = {}
        for cn, vals in f["cols"].items():
            if cn == tcol:
                continue
            win = [vals[i] for i in idx if i < len(vals)]
            if not win:
                continue
            wm = sum(win) / len(win)
            om = sum(vals) / len(vals)
            sig[cn] = {"window_mean": round(wm, 3), "overall_mean": round(om, 3),
                       "delta": round(wm - om, 3)}
        per.append({"file": f["name"], "n_samples": len(idx), "signals": sig})
    from .sensor_anomaly import detect_table
    from .sensor_align import build_alignment
    anoms, anom_full = [], []
    for f in ctx["files"]:
        r = detect_table(f["cols"], f["header"], f["time_col"], None)
        if r:
            r["file"] = f["name"]; r["t_start_abs"] = f["t_start_abs"]; anom_full.append(r)
            for e in r["events"]:
                if e.get("time") is not None and abs(e["time"] - c) <= rad:
                    anoms.append({"file": f["name"],
                                  **{k: e[k] for k in ("type", "signal", "time") if k in e}})
    corr = []
    al = build_alignment(anom_full) if len(anom_full) >= 2 else None
    if al:
        corr = [cc for cc in al["correlated"] if abs(cc["t"] - c) <= rad]
    return {"kind": "focus", "title": "Focus around t=%.1fs" % c, "center": round(c, 2),
            "radius_s": rad, "per_file": per, "anomalies_here": anoms[:10],
            "correlated_here": corr}


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
        {"heading_col": "heading column name (auto if omitted)", "turn_percentile": "0-100; samples above this heading-change percentile count as turns (default 85 = sharpest 15%)"}),
    "focus_time": (_t_tool_focus_time,
        "Zoom into a specific moment: what every signal was doing around time t (window mean vs its overall mean), plus anomalies and cross-sensor moments inside that window. Use when the user names a time or asks 'what happened at / around T seconds'.",
        {"center": "time in seconds from the start of the run", "radius_s": "half-window in seconds (default 3)"}),
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
    "tools that serve THIS goal — do not run everything. Use 1-4 tools; only tool "
    "names from the provided list.\n"
    "BUT if the GOAL is empty, vague, or too broad to analyze well, do NOT guess — "
    "instead offer the user concrete directions to choose from, each phrased as a "
    "specific question tailored to THESE columns.\n"
    "Return STRICT JSON, exactly one of:\n"
    '  {"plan":[{"tool":"<name>","params":{...},"why":"<short reason>"}]}\n'
    '  {"clarify":["<specific question 1>","<specific question 2>","<specific question 3>"]}\n'
    "Routing guidance (analytical judgment):\n"
    "- If the user refers to a SPECIFIC TIME ('at 60s', 'around 2 min', 'that moment', "
    "'what happened at T') -> use focus_time with that time (in seconds) as center. "
    "This holds even if a previous turn used a different tool.\n"
    "- noisy / reliable / clean / quality -> signal_noise.\n"
    "- what went wrong / when / faults / glitches / spikes -> detect_anomalies.\n"
    "- turns / corners / curves vs straights -> segment_turns_vs_straight.\n"
    "- do the sensors agree / same event / correlate across sensors -> cross_sensor_correlation.\n"
    "- Plan for the CURRENT goal; earlier turns are context, not a template to repeat. "
    "Prefer 1-2 focused tools over many."
)


def _default_clarify(ctx: dict) -> list:
    """Concrete directions tailored to what's actually in the data (used when the
    goal is empty and the model doesn't offer its own)."""
    low = set()
    multi = len([f for f in ctx["files"] if f.get("cols")]) >= 2
    for f in ctx["files"]:
        low |= {c.lower() for c in f["cols"]}
    qs = []
    if {"lat", "latitude"} & low and {"lon", "lng", "longitude"} & low:
        qs.append("Show the route and where the GPS position jumps unrealistically")
    if any(k in c for c in low for k in ("heading", "yaw", "bearing")):
        qs.append("Compare the robot's behaviour in turns vs straight sections")
    if multi:
        qs.append("Find moments where two or more sensors reacted to the same event")
    qs.append("Which signals are noisiest / least reliable, and when did anything go wrong?")
    return qs[:4]


def plan(goal: str, ctx: dict, llm_call) -> dict:
    """Ask the planner LLM for a tool plan tailored to `goal`, OR — when the goal is
    vague/empty — clarifying questions to guide the user. Returns
    {ok, mode:'analyze'|'clarify', plan|questions, raw, source}."""
    _g = (goal or "").strip().lower().rstrip("?. ")
    # vague = nothing specific left after removing generic/analysis stop-words. This
    # catches "analyze", "analyze this data", "give me an overview", etc. — the exact
    # case the prof hit — without a brittle exact-match list.
    import re as _re
    _STOP = {"analyze", "analyse", "analysis", "analyzing", "this", "that", "these",
             "the", "data", "dataset", "please", "for", "me", "my", "our", "us",
             "review", "eda", "do", "a", "an", "of", "it", "its", "check", "look",
             "at", "run", "go", "help", "summary", "summarize", "summarise", "explore",
             "what", "about", "tell", "insights", "give", "some", "show", "and", "to",
             "on", "in", "is", "are", "perform", "see", "view", "get", "overview",
             "report", "here", "can", "you", "please", "everything", "all"}
    _mean = [t for t in _re.findall(r"[a-z]+", _g) if t not in _STOP]
    vague = len(_g) < 6 or not _mean
    menu = tool_menu()
    prompt = (f"GOAL: {goal.strip() or '(none given)'}\n\n"
              f"DATASET COLUMNS:\n{profile_text(ctx)}\n\n"
              f"AVAILABLE TOOLS:\n{json.dumps(menu, indent=1)}\n\n"
              "Return the JSON now (a plan, or clarify if the goal is unclear).")
    raw, source = "", "llm"
    try:
        raw = llm_call(prompt, _PLAN_SYS) or ""
    except Exception as e:
        raw, source = f"(llm error: {e})", "error"
    steps, questions = _extract(raw)
    # only trust model clarify if it's genuinely 2+ crisp options (3B often returns
    # a single run-on meta-question — worse than our concrete data-aware defaults).
    good_qs = [q for q in questions if 8 <= len(q) <= 140]
    if vague and not steps:                                 # empty/generic goal → guide
        qs = good_qs if len(good_qs) >= 2 else _default_clarify(ctx)
        return {"ok": True, "mode": "clarify", "questions": qs[:4],
                "raw": raw[:1200],
                "source": source if len(good_qs) >= 2 else "default"}
    if len(good_qs) >= 2 and not steps:                     # model chose to clarify
        return {"ok": True, "mode": "clarify", "questions": good_qs[:4],
                "raw": raw[:1200], "source": source}
    if not steps:
        steps, source = _heuristic_plan(goal, ctx), "heuristic"
    # keep only known tools, drop duplicate (tool,params) steps
    seen, deduped = set(), []
    for s in steps:
        if s.get("tool") not in TOOLS:
            continue
        key = (s["tool"], json.dumps(s.get("params") or {}, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    steps = deduped[:4] or _heuristic_plan(goal, ctx)
    return {"ok": True, "mode": "analyze", "plan": steps, "raw": raw[:1200], "source": source}


def _extract(raw: str):
    """Parse the planner output → (plan_steps, clarify_questions). Either may be []."""
    import re
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return [], []
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return [], []
    if not isinstance(obj, dict):
        obj = {"plan": obj}
    steps = []
    for s in (obj.get("plan") or []):
        if isinstance(s, dict) and s.get("tool"):
            steps.append({"tool": str(s["tool"]).strip(),
                          "params": s.get("params") or {},
                          "why": str(s.get("why", ""))[:160]})
    questions = [str(q)[:180] for q in (obj.get("clarify") or []) if str(q).strip()]
    return steps, questions


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


# ---------------------------------------------------------------------------
# narrator — answer the goal using ONLY the grounded results (no invented numbers).
# ---------------------------------------------------------------------------
_NARRATE_SYS = (
    "You are a data analyst answering the user's question about their sensor dataset. "
    "You are given ONLY computed facts (real numbers from tools that ran on the data). "
    "Answer the question directly and specifically, citing the actual numbers/timestamps "
    "from the facts. Do NOT invent any number that isn't in the facts. If the facts don't "
    "cover the question, say what you'd need to run next. 2-5 sentences, plain and concrete."
)


def _facts_digest(results: list) -> str:
    """Compact, model-friendly rendering of the grounded results."""
    out = []
    for r in results:
        k = r.get("kind")
        if k == "noise":
            rows = ", ".join(f"{x['file']}:{x['signal']} {x['noise_pct']}% (SNR {x['snr_db']}dB)"
                             for x in r.get("rows", [])[:8])
            out.append(f"[noise] {rows}")
        elif k == "anomalies":
            fs = "; ".join(f"{x['file']}: {x['by_type']}" for x in r.get("files", []))
            out.append(f"[anomalies] total={r.get('total')}; {fs}")
        elif k == "cross":
            cs = "; ".join(f"t={c['t']}s ({c['n_files']} sensors)" for c in r.get("correlated", [])[:6])
            out.append(f"[cross-sensor] {r.get('n_correlated')} correlated moment(s): {cs or '—'} "
                       f"(basis: {r.get('basis','?')})")
        elif k == "segments":
            out.append(f"[turns vs straights] segmented on {r.get('segmented_on')}, "
                       f"turn_fraction={r.get('turn_fraction')}, "
                       f"in_turns={json.dumps(r.get('in_turns', {}))[:300]}, "
                       f"in_straights={json.dumps(r.get('in_straights', {}))[:300]}")
        elif k == "focus":
            if r.get("note"):
                out.append(f"[focus] {r['note']}")
            else:
                sigs = "; ".join(f"{p['file']}: " + ", ".join(
                    f"{s} window_mean={v['window_mean']} (delta {v['delta']} vs overall)"
                    for s, v in list(p["signals"].items())[:5]) for p in r.get("per_file", []))
                out.append(f"[focus t={r.get('center')}s +/-{r.get('radius_s')}s] {sigs} | "
                           f"anomalies_here={r.get('anomalies_here')} | "
                           f"correlated_here={r.get('correlated_here')}")
        elif k == "stats":
            rows = ", ".join(f"{x['signal']}[{x['min']}..{x['max']}] mean {x['mean']} std {x['std']}"
                             for x in r.get("rows", [])[:10])
            out.append(f"[stats] {rows}")
        elif k == "error":
            out.append(f"[{r.get('tool')} error] {r.get('error')}")
        else:
            out.append(f"[{k}] {json.dumps({kk: vv for kk, vv in r.items() if kk not in ('tool','why')})[:300]}")
    return "\n".join(out)


def narrate(goal: str, results: list, llm_call) -> str:
    """LLM answers `goal` grounded ONLY in `results`. Falls back to the digest."""
    facts = _facts_digest(results)
    prompt = (f"USER QUESTION / GOAL: {goal.strip() or '(general read)'}\n\n"
              f"COMPUTED FACTS (the only numbers you may use):\n{facts}\n\n"
              "Answer now.")
    try:
        txt = (llm_call(prompt, _NARRATE_SYS) or "").strip()
        return txt or facts
    except Exception:
        return facts


def analyze_goal(root, goal: str, llm_call) -> dict:
    """One-shot: plan for `goal`, run the plan, narrate. Returns everything the
    page needs. `llm_call(prompt, system) -> str`."""
    ctx = load_context(root)
    if not ctx["files"]:
        return {"ok": False, "error": "no tabular sensor files to analyze",
                "plan": [], "results": [], "answer": ""}
    pl = plan(goal, ctx, llm_call)
    if pl.get("mode") == "clarify":                         # guide the user's intent
        return {"ok": True, "mode": "clarify", "goal": goal,
                "questions": pl.get("questions", []), "plan_source": pl["source"],
                "profile": profile_text(ctx)}
    results = run_plan(pl["plan"], ctx)
    answer = narrate(goal, results, llm_call)
    return {"ok": True, "mode": "analyze", "goal": goal, "plan": pl["plan"],
            "plan_source": pl["source"], "results": results, "answer": answer,
            "profile": profile_text(ctx)}
