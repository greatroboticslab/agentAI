"""Rule-based dataset quality checks — pure, no FastAPI / app state.

v3.1 (P5 de-monolith, step 2): extracted verbatim from dashboard_server.py's
`_detect_dataset_issues` (the first extraction was dataset_eda.py in v3.0.193).
Behaviour is byte-identical; only the location changed. Its one dependency is the
per-domain threshold defaults in `db.DEFAULT_DOMAIN_CONFIG`.

Given a dataset-analysis dict `a` (from `_analyze_dataset`) and optional per-domain
`thr` overrides, returns a list of grounded issues: {severity, title, detail}.
"""


def detect_dataset_issues(a: dict, thr: dict = None) -> list:
    """Rule-based, grounded data-quality issues. Each: {severity, title, detail}.
    Thresholds come from the project's domain config (thr) so a new field can tune
    them; defaults = today's constants (no regression)."""
    from . import db as _dbthr
    T = dict(_dbthr.DEFAULT_DOMAIN_CONFIG["thresholds"])
    if thr:
        T.update({k: v for k, v in thr.items() if v is not None})
    issues = []
    if not a or not a.get("ok"):
        return issues
    n = a.get("n_images") or 0
    ann = a.get("annotations") or {}
    per = ann.get("per_class") or {}
    im = a.get("images") or {}

    if 0 < n < 20:
        issues.append({"severity": "high", "title": "Very small dataset",
                       "detail": f"Only {n} images — too few to train a reliable model. Aim for at least a few hundred."})
    elif 20 <= n < T["small_dataset"]:
        issues.append({"severity": "medium", "title": "Small dataset",
                       "detail": f"{n} images is enough to smoke-test the pipeline but likely too few for good accuracy."})

    if ann.get("type") == "none" and a.get("modality", {}).get("image"):
        issues.append({"severity": "high", "title": "No labels detected",
                       "detail": "Supervised training needs labels — add YOLO labels/ + data.yaml, COCO/VOC "
                                 "annotations, or class subfolders (train/<class>/) for classification."})

    if per and len(per) >= 2:
        vals = list(per.values())
        ratio = (max(vals) / min(vals)) if min(vals) else 0
        if ratio >= T["imbalance_high"]:
            issues.append({"severity": "high", "title": "Severe class imbalance",
                           "detail": f"Largest class has {ratio:.0f}× the samples of the smallest — the model will "
                                     f"be biased. Collect more of the rare classes or rebalance."})
        elif ratio >= T["imbalance_med"]:
            issues.append({"severity": "medium", "title": "Class imbalance",
                           "detail": f"Class sizes differ by {ratio:.0f}× — consider balancing."})
        few = [k for k, v in per.items() if v < T["min_per_class"]]
        if few:
            issues.append({"severity": "medium", "title": "Classes with too few samples",
                           "detail": f"Under {T['min_per_class']} samples: " + ", ".join(sorted(few)[:8]) + "."})

    sampled = im.get("sampled") or 0
    dup = a.get("near_duplicates") or 0
    if sampled and dup / sampled > T["dup_frac"]:
        issues.append({"severity": "medium", "title": "Many near-duplicate images",
                       "detail": f"~{dup}/{sampled} sampled images look near-identical — duplicates inflate size "
                                 f"without adding information and can leak between train/val."})

    w = (im.get("width") or {}); h = (im.get("height") or {})
    if w.get("median") and h.get("median") and min(w["median"], h["median"]) < T["tiny_px"]:
        issues.append({"severity": "medium", "title": "Tiny images",
                       "detail": f"Median size ~{w['median']}×{h['median']}px — very small images limit achievable accuracy."})
    if w.get("max") and w["max"] > 5000:
        issues.append({"severity": "low", "title": "Very large images",
                       "detail": f"Up to {w['max']}px wide — training will down-scale them (imgsz); fine but slower to load."})

    splits = a.get("splits") or {}
    if per and not any(k in splits for k in ("val", "valid", "test")):
        issues.append({"severity": "medium", "title": "No validation split",
                       "detail": "No val/ or test/ split found — without a held-out set you can't measure real accuracy."})

    if ann.get("type") == "yolo":
        lbl = ann.get("labeled_images") or 0
        if n and lbl < n:
            issues.append({"severity": "medium", "title": "Some images unlabeled",
                           "detail": f"{n - lbl} of {n} images have no YOLO label file."})
    return issues
