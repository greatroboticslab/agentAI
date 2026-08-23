#!/usr/bin/env python3
"""Regenerate every presented figure/table from checked-in, anchored data.

One command, no hand edits, deterministic output (no timestamps inside artifacts):

    python make_figures.py            # writes results/figures/

Data source: results/framework/figures_data.json — every point carries the anchor
where it was recorded and whether the holdout content-hash guard applies to it.
When the M1 re-measurement artifacts (results/framework/m1_<tier>_seed*.json)
exist, their mean±std fills the pending points automatically.
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results" / "framework" / "figures_data.json"
OUT = ROOT / "results" / "figures"


def load():
    d = json.loads(DATA.read_text())
    # fill pending quality-vs-scale points from M1 artifacts when they exist
    for p in d["quality_vs_scale"]:
        if not p.get("pending"):
            continue
        tier = "raw" if "raw" in p["label"] else "curated"
        vals = []
        for f in sorted((ROOT / "results" / "framework").glob(f"m1_{tier}_seed*.json")):
            try:
                j = json.loads(f.read_text())
            except Exception:
                continue
            s = j.get("summary") or {}
            m = s.get("map50_95") or (s.get("metrics") or {}).get("map50_95") \
                if isinstance(s.get("metrics"), dict) else s.get("map50_95")
            if j.get("ok") and isinstance(m, (int, float)):
                vals.append(float(m))
        if len(vals) >= 2:
            p["map50_95"] = round(statistics.mean(vals), 4)
            p["std"] = round(statistics.stdev(vals), 4)
            p["n_seeds"] = len(vals)
            p["pending"] = False
            p["filled_from"] = "m1_%s_seed*.json" % tier
    return d


def fig_quality_vs_scale(d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = d["quality_vs_scale"]
    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=150)
    # stagger annotation offsets so co-located cwd12-only points don't overprint
    offsets = [(10, -13), (12, 2), (-160, 12), (-150, 10), (8, -16), (8, 8), (8, 8)]
    for i, p in enumerate(pts):
        x = p["train_images"]
        if p.get("pending"):
            ax.axvline(x, color="#bbbbbb", ls=":", lw=1)
            # anchor the note to axis coordinates so it is always visible
            ax.annotate(p["label"] + " — running", (x, 0.03),
                        xycoords=ax.get_xaxis_transform(), rotation=90,
                        fontsize=7, color="#777777", ha="right", va="bottom")
            continue
        sealed = "PRE-GUARD" not in p["guard"]
        color = "#1f77b4" if sealed else "#d62728"
        marker = "o" if sealed else "s"          # shape == guard state (matches legend)
        yerr = p.get("std")
        n = p.get("n_seeds", 1)
        ax.errorbar(x, p["map50_95"], yerr=yerr, fmt=marker, color=color,
                    capsize=4, ms=7, lw=1.5)
        label = p["label"] + (" (n=%d)" % n if n > 1 else "")
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(label, (x, p["map50_95"]),
                    textcoords="offset points", xytext=(dx, dy), fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("training images (log scale)")
    ax.set_ylabel("cwd12 holdout mAP50-95 (pycocotools)")
    ax.set_title("Quality beats scale: uncurated pseudo-labeled volume hurts")
    ax.grid(True, alpha=0.25)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color="#1f77b4",
               label="guard-clean (sealed or single-dataset)"),
        Line2D([], [], marker="s", ls="", color="#d62728",
               label="pre-guard (leak channel open at the time)"),
    ], fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "quality_vs_scale.png")
    plt.close(fig)


def md_table(rows, cols, headers):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()

    fig_quality_vs_scale(d)

    b = d["benchmark_cwd12_map50"]
    (OUT / "benchmark_table.md").write_text(
        "# CottonWeedDet12 benchmark (mAP@0.5)\n\n*%s*\n\n" % d["benchmark_anchor"]
        + md_table(b, ["rank", "model", "params", "map50", "map50_95", "note"],
                   ["#", "Model", "Params", "mAP@0.5", "mAP50-95", "Note"]))

    ps = d["per_species_yolo11n_val"]
    (OUT / "per_species.md").write_text(
        "# Per-species validation — YOLO11n baseline\n\n*%s*\n\n" % ps["anchor"]
        + md_table(ps["rows"], ["cls", "map50", "map50_95"],
                   ["Species", "mAP@0.5", "mAP50-95"]))

    sd = d["rfdetr_seeds"]
    (OUT / "rfdetr_seeds.md").write_text(
        "# RF-DETR Large on cwd12-only — seed spread\n\n*%s*\n\n" % sd["anchor"]
        + md_table(sd["rows"], ["run", "seed", "epochs", "map50_95", "map50", "map75"],
                   ["Run", "Seed", "Epochs", "mAP50-95", "mAP@0.5", "mAP@0.75"])
        + "\n**mean ± std = %.4f ± %.4f (n=%d)** — quote this, never the best run alone.\n"
          % (sd["mean"], sd["std"], len(sd["rows"])))

    qs = d["quality_vs_scale"]
    lines = ["# Headline numbers with their evidentiary state\n"]
    for p in qs:
        state = "RUNNING" if p.get("pending") else (
            "%.4f%s (n=%d)" % (p["map50_95"],
                               " ± %.4f" % p["std"] if p.get("std") else "",
                               p.get("n_seeds", 1)))
        lines.append("- **%s** — %s · guard: %s · %s" %
                     (p["label"], state, p["guard"], p["anchor"]))
    ls = d["license_sweep_2026_08_23"]
    lines.append("\n## Data governance (license sweep 2026-08-23)")
    lines.append("- %d registry datasets, **%d with a recorded license**; sources: %s" %
                 (ls["datasets"], ls["license_recorded"], ls["sources"]))
    lines.append("- Consequence: %s" % ls["consequence"])
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")

    for f in sorted(OUT.iterdir()):
        print("%8d  %s" % (f.stat().st_size, f.name))


if __name__ == "__main__":
    main()
