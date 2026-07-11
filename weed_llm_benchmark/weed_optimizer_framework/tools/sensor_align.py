"""Cross-modal temporal alignment — pure, stdlib only.

v3.2.0: the multi-sensor differentiator. Given per-file anomaly events (each with a
time relative to that file's own start, plus the file's absolute start timestamp),
place every event on ONE shared time axis and find CORRELATED MOMENTS — instants
where two or more different sensors flagged something within a small window (a
pothole shows up as an IMU spike AND a GPS wobble at the same second). Optionally
maps each correlated moment to the nearest video frame.

Honesty: alignment uses the files' absolute timestamps when present (true shared
clock). When some file lacks absolute time we fall back to each-file-relative time
and SAY SO — that assumes the logs were started together. Video has no absolute
clock, so its frame mapping is an explicit "assumes the clip starts with the
sensors" estimate, not a guarantee.
"""
from __future__ import annotations


def build_alignment(files: list, video: dict | None = None, window_s: float = 1.0) -> dict | None:
    """`files`: list of per-file anomaly dicts (each: file, events:[{type,time,signal}],
    t_start_abs). Returns the aligned timeline + correlated moments, or None when
    there's nothing to align across (needs ≥2 sensor files, or ≥1 sensor + video)."""
    have_video = bool(video and video.get("fps"))
    if len([f for f in files if f.get("events")]) < 2 and not (files and have_video):
        return None

    abs_starts = [f["t_start_abs"] for f in files if f.get("t_start_abs") is not None]
    use_abs = len(abs_starts) == len([f for f in files if f.get("events")]) and len(abs_starts) >= 2
    origin = min(abs_starts) if abs_starts else 0.0

    lanes, all_ev, span = [], [], 0.0
    for f in files:
        off = (f["t_start_abs"] - origin) if (use_abs and f.get("t_start_abs") is not None) else 0.0
        evs = []
        for e in f.get("events", []):
            if e.get("time") is None:
                continue
            tc = round(e["time"] + off, 2)
            span = max(span, tc)
            evs.append({"t": tc, "type": e["type"], "signal": e.get("signal")})
            all_ev.append((tc, f["file"], e["type"], e.get("signal")))
        if evs:
            lanes.append({"file": f["file"], "events": evs})

    # correlated moments: consecutive events within `window_s` involving ≥2 files
    all_ev.sort(key=lambda x: x[0])
    correlated, used, n = [], [False] * len(all_ev), len(all_ev)
    for i in range(n):
        if used[i]:
            continue
        grp = [all_ev[i]]
        for j in range(i + 1, n):
            if all_ev[j][0] - all_ev[i][0] <= window_s:
                grp.append(all_ev[j])
            else:
                break
        if len({g[1] for g in grp}) >= 2:                 # ≥2 distinct files
            for k in range(i, i + len(grp)):
                used[k] = True
            correlated.append({
                "t": round(sum(g[0] for g in grp) / len(grp), 2),
                "n_files": len({g[1] for g in grp}),
                "involved": [{"file": g[1], "type": g[2], "signal": g[3]} for g in grp][:6],
            })

    out = {
        "basis": ("absolute timestamps (true shared clock)" if use_abs
                  else "each file's own start (assumes the logs were started together)"),
        "span_s": round(span, 1),
        "lanes": lanes,
        "correlated": correlated[:12],
        "n_correlated": len(correlated),
    }
    if have_video:
        fps = video["fps"]
        out["video"] = {"file": video.get("file"), "fps": fps,
                        "frames": video.get("frames"),
                        "note": f"frame ≈ time × {fps} fps, assuming the clip starts with the sensors",
                        "frame_at": {str(c["t"]): int(round(c["t"] * fps)) for c in correlated[:12]}}
    return out


def plot_alignment(align: dict, out_png) -> bool:
    """Horizontal timeline: one lane per source, event markers by type, dashed
    vertical lines at correlated moments. Returns True on success."""
    lanes = align.get("lanes") or []
    if not lanes:
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"sudden_change": "#2563eb", "gps_jump": "#dc2626",
              "time_gap": "#d97706", "flatline": "#7c3aed"}
    fig, ax = plt.subplots(figsize=(8.4, 0.7 * len(lanes) + 1.6), dpi=110)
    names = [lane["file"] for lane in lanes]
    for yi, lane in enumerate(lanes):
        ax.axhline(yi, color="#e5e7eb", lw=1, zorder=0)
        for e in lane["events"]:
            ax.scatter(e["t"], yi, s=70, marker="|", linewidths=2.5,
                       color=colors.get(e["type"], "#111827"), zorder=3)
    for c in align.get("correlated", []):
        ax.axvline(c["t"], color="#dc2626", ls="--", lw=1.2, alpha=.7, zorder=1)
        ax.text(c["t"], len(lanes) - 0.55, f"t={c['t']}s\n{c['n_files']} sensors",
                fontsize=8, color="#dc2626", ha="center", va="bottom")
    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_ylim(-0.6, len(lanes) - 0.2)
    ax.set_xlabel("time (s, shared axis)")
    ax.set_title("Cross-sensor timeline — red dashed = moments multiple sensors flagged together")
    handles = [plt.Line2D([0], [0], marker="|", color=c, lw=0, mew=2.5, ms=10, label=t.replace("_", " "))
               for t, c in colors.items()]
    ax.legend(handles=handles, fontsize=8, loc="upper right", ncol=2)
    ax.grid(axis="x", alpha=.3)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return True
