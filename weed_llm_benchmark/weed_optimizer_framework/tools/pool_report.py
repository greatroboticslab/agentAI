"""Pool report + per-source scorecard (SUPERWEED_PLAN S1/S2 gates).

Two mechanisms the plan specified and execution had been substituting human
judgement for:

  * **scorecard** — one record per source: images, labelled, label ratio, classes,
    DINOv2 quality score, license, status. Written back into the registry entry so
    it travels with the data instead of living in a supervisor's head.
  * **pool report** — the aggregate the S2 gate asks for: size, class balance,
    quality histogram, license mix, and what is quarantined and why.

Usage (from the repo root, cluster or lab):
    python -m weed_optimizer_framework.tools.pool_report report
    python -m weed_optimizer_framework.tools.pool_report scorecards   # writes back
    python -m weed_optimizer_framework.tools.pool_report report --json pool.json

Reads the registry and `dinov2_curator/slug_scores.json`; never touches image data,
so it is cheap enough to run at every supervision tick.
"""
import argparse
import json
import os
import time
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REG = os.path.join(REPO, "results", "framework", "dataset_registry.json")
SCORES = os.path.join(REPO, "results", "framework", "dinov2_curator", "slug_scores.json")

# a source only counts toward the audited pool if it is actually usable
MIN_LABEL_RATIO = 0.5
MIN_LABELS = 50


def _load():
    reg = json.load(open(REG))
    ds = {k: v for k, v in (reg.get("datasets") or {}).items() if isinstance(v, dict)}
    try:
        raw = json.load(open(SCORES))
        scores = {k: float(v["score"]) for k, v in raw.items()
                  if isinstance(v, dict) and v.get("score") is not None}
    except Exception:
        scores = {}
    return reg, ds, scores


def _count_labels_on_disk(root, cap=200000):
    """Bounded count of YOLO label files under a dataset root.

    `local_labeled` is a field later harvests populate and legacy entries never
    did — reading it alone reports 0 for datasets the merge trains on happily
    (cottonweed_sp8, the in-domain core, is the clearest case). So look at disk,
    but only inside `labels*` directories and with a hard cap, because walking an
    image-filled tree on Lustre is exactly the mistake this project has made
    before.
    """
    if not root or not os.path.isdir(root):
        return None
    n = 0
    for dirpath, dirnames, filenames in os.walk(root):
        base = os.path.basename(dirpath).lower()
        if base.startswith("label"):
            for f in filenames:
                if f.endswith(".txt"):
                    n += 1
                    if n >= cap:
                        return n
        # never descend into image directories
        dirnames[:] = [d for d in dirnames
                       if not d.lower().startswith(("image", "img", "frames"))]
    return n


def _card(slug, info, scores):
    imgs = int(info.get("local_images") or info.get("n_images") or 0)
    lab = int(info.get("local_labeled") or info.get("n_local_labels") or 0)
    lab_src = "registry"
    if lab == 0:
        disk = _count_labels_on_disk(info.get("local_path"))
        if disk:
            lab, lab_src = disk, "disk"
    ratio = (lab / imgs) if imgs else 0.0
    prov = info.get("provenance") or {}
    usable = (lab >= MIN_LABELS and ratio >= MIN_LABEL_RATIO
              and info.get("status") != "quarantined")
    return {
        "images": imgs,
        "labeled": lab,
        "label_ratio": round(ratio, 3),
        "labeled_source": lab_src,
        "classes": len(info.get("class_names") or []),
        "dino_score": round(scores[slug], 3) if slug in scores else None,
        "license": prov.get("license") or "MISSING",
        "source": info.get("source"),
        "status": info.get("status"),
        "quarantine_reason": info.get("quarantine_reason"),
        "in_audited_pool": usable,
        "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def scorecards(write=True):
    reg, ds, scores = _load()
    cards = {slug: _card(slug, info, scores) for slug, info in ds.items()}
    if write:
        from .registry_lock import update_registry

        def _apply(r):
            for slug, card in cards.items():
                info = r.get("datasets", {}).get(slug)
                if isinstance(info, dict):
                    info["scorecard"] = card
            return r
        update_registry(REG, _apply)
        print("wrote %d scorecards into the registry (locked)" % len(cards))
    return cards


def report(as_json=None):
    reg, ds, scores = _load()
    cards = {slug: _card(slug, info, scores) for slug, info in ds.items()}
    active = {k: c for k, c in cards.items() if c["status"] != "quarantined"}
    pool = {k: c for k, c in cards.items() if c["in_audited_pool"]}

    def _sum(d, key):
        return sum(c[key] for c in d.values())

    classes = Counter()
    for slug, info in ds.items():
        if cards[slug]["in_audited_pool"]:
            for c in (info.get("class_names") or []):
                classes[str(c).strip().lower()] += 1

    qbuckets = Counter()
    for c in active.values():
        s = c["dino_score"]
        qbuckets["unscored" if s is None else
                 ("<0.30" if s < .30 else "0.30-0.45" if s < .45 else
                  "0.45-0.60" if s < .60 else ">=0.60")] += 1

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "datasets": {"total": len(cards), "active": len(active),
                     "quarantined": len(cards) - len(active),
                     "in_audited_pool": len(pool)},
        "images": {"active": _sum(active, "images"), "audited_pool": _sum(pool, "images")},
        "labeled": {"active": _sum(active, "labeled"),
                    "audited_pool": _sum(pool, "labeled")},
        "audited_pool_criteria": {"min_labels": MIN_LABELS,
                                  "min_label_ratio": MIN_LABEL_RATIO,
                                  "not_quarantined": True},
        "class_balance_top20": dict(classes.most_common(20)),
        "distinct_classes": len(classes),
        "quality_histogram_dino": dict(qbuckets),
        "license_mix": dict(Counter(c["license"] for c in active.values()).most_common()),
        "quarantined": {k: c["quarantine_reason"] for k, c in cards.items()
                        if c["status"] == "quarantined"},
        "excluded_from_pool_but_active": {
            k: ("%d labeled / %d images (ratio %.2f)"
                % (c["labeled"], c["images"], c["label_ratio"]))
            for k, c in active.items() if not c["in_audited_pool"]},
    }
    if as_json:
        with open(as_json, "w") as f:
            json.dump(out, f, indent=1)
        print("wrote", as_json)
    d = out["datasets"]
    print("POOL REPORT  %s" % out["generated_at"])
    print("  datasets: %d total · %d active · %d quarantined · %d in audited pool"
          % (d["total"], d["active"], d["quarantined"], d["in_audited_pool"]))
    print("  images:   %d active · %d audited-pool" % (out["images"]["active"],
                                                       out["images"]["audited_pool"]))
    print("  labeled:  %d active · %d audited-pool  <- the number that trains"
          % (out["labeled"]["active"], out["labeled"]["audited_pool"]))
    print("  classes:  %d distinct; top: %s"
          % (out["distinct_classes"],
             list(out["class_balance_top20"].items())[:6]))
    print("  DINO quality: %s" % out["quality_histogram_dino"])
    print("  licenses: %s" % dict(list(out["license_mix"].items())[:6]))
    print("  quarantined: %s" % list(out["quarantined"].keys()))
    if out["excluded_from_pool_but_active"]:
        print("  active but NOT audited-pool (too few/thin labels):")
        for k, why in list(out["excluded_from_pool_but_active"].items())[:10]:
            print("     %-46s %s" % (k[:46], why))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["report", "scorecards"])
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.cmd == "scorecards":
        scorecards()
    else:
        report(as_json=a.json)
