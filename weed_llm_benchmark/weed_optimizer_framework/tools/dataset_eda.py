"""Dataset EDA — pure, filesystem-only analysis helpers (no FastAPI / app state).

Extracted from dashboard_server.py (v3.0.193) as the first step of the Phase-5
monolith modularization. `analyze_nonimage(root)` inspects NON-image data (video /
audio / sensor / pointcloud / text) under a directory and returns a per-modality
summary. Behaviour is byte-identical to the in-dashboard version it replaced.
"""
from __future__ import annotations


def analyze_nonimage(root) -> dict:
    """v3.0.162 — best-effort per-modality analysis for NON-image data (video /
    audio / sensor / pointcloud / text). Bounded samples; every branch degrades
    to a plain count on error. Excludes annotation sidecars (labels/, data.yaml)."""
    out = {}
    SIDECAR_NAMES = {"data.yaml", "data.yml", "classes.txt", "notes.json"}
    # Analysis-specific DISJOINT extension map (each ext → exactly one modality) so
    # shared extensions never double-count. Distinct from _MODALITY_EXT (which is
    # about upload acceptance and deliberately overlaps).
    BUCKET_EXT = (
        ("video", {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}),
        ("audio", {".wav", ".flac", ".mp3", ".ogg", ".m4a"}),
        ("pointcloud", {".pcd", ".ply", ".las", ".laz", ".npy", ".npz"}),
        ("sensor", {".csv", ".tsv", ".parquet", ".bag", ".mcap", ".gpx", ".nmea", ".log"}),
        ("text", {".txt", ".md", ".json", ".jsonl"}),
    )
    _ext2mod = {e: m for m, exts in BUCKET_EXT for e in exts}
    buckets = {m: [] for m, _ in BUCKET_EXT}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        parts = {x.lower() for x in p.parts}
        if "labels" in parts or p.name.lower() in SIDECAR_NAMES:
            continue                            # skip annotation sidecars
        m = _ext2mod.get(p.suffix.lower())
        if m:
            buckets[m].append(p)

    # ---- VIDEO (cv2) ----
    vids = buckets["video"]
    if vids:
        info = {"n": len(vids), "per": [], "total_duration_s": 0.0, "sampled": 0}
        try:
            import cv2
            for p in vids[:6]:
                try:
                    cap = cv2.VideoCapture(str(p))
                    fps = cap.get(cv2.CAP_PROP_FPS) or 0
                    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                    cap.release()
                    dur = round(frames / fps, 1) if fps else 0
                    info["per"].append({"file": p.name, "dur_s": dur, "fps": round(fps, 1),
                                        "frames": int(frames), "w": w, "h": hh})
                    info["total_duration_s"] += dur
                except Exception:
                    continue
            info["sampled"] = len(info["per"])
            info["total_duration_s"] = round(info["total_duration_s"], 1)
        except Exception:
            info["note"] = "cv2 unavailable"
        out["video"] = info

    # ---- AUDIO (.wav via stdlib; others counted) ----
    auds = buckets["audio"]
    if auds:
        info = {"n": len(auds), "sampled": 0, "wav_dur_s": 0.0}
        import wave as _wave
        got = 0
        for p in auds[:8]:
            if p.suffix.lower() != ".wav":
                continue
            try:
                with _wave.open(str(p)) as wf:
                    fr = wf.getframerate()
                    info["wav_dur_s"] += round(wf.getnframes() / fr, 1) if fr else 0
                    info["rate"] = fr
                    info["channels"] = wf.getnchannels()
                    got += 1
            except Exception:
                pass
        info["sampled"] = got
        info["wav_dur_s"] = round(info["wav_dur_s"], 1)
        if got == 0:
            info["note"] = "duration needs a codec lib (only .wav is read natively)"
        out["audio"] = info

    # ---- SENSOR (csv/tsv columns + numeric stats; json/jsonl counted) ----
    sens = buckets["sensor"]
    if sens:
        info = {"n": len(sens), "tables": []}
        import csv as _csv
        tabular = [x for x in sens if x.suffix.lower() in (".csv", ".tsv")]
        # v3.0.187: understand tabular/time-series sensor data like sensor data —
        # detect a LABEL column (categorical) + class balance, and estimate the
        # sampling rate from a time column — so downstream analysis knows this data
        # is (often) already labeled and time-sampled, not "unlabeled images".
        _label_names = ("label", "class", "activity", "target", "y", "annotation", "state")
        _time_names = ("t", "time", "timestamp", "ts", "sec", "seconds", "millis", "ms")
        label_counts_all = {}       # {value: count} across sampled files
        label_col_name = None
        rates = []                  # estimated Hz per file
        total_rows = 0
        _SAMPLE_N = 24
        for p in tabular[:_SAMPLE_N]:
            try:
                delim = "\t" if p.suffix.lower() == ".tsv" else ","
                with open(p, newline="", encoding="utf-8", errors="replace") as f:
                    rd = _csv.reader(f, delimiter=delim)
                    header = next(rd, [])
                    low = [str(h).strip().lower() for h in header]
                    nrow = 0
                    numcols = {}
                    catcols = {}           # {col_idx: {value: count}} for non-numeric
                    tcol = next((i for i, h in enumerate(low) if h in _time_names), None)
                    tmin = tmax = None
                    for row in rd:
                        nrow += 1
                        if nrow <= 20000:
                            for i, val in enumerate(row):
                                try:
                                    fv = float(val)
                                    numcols.setdefault(i, []).append(fv)
                                    if i == tcol:
                                        tmin = fv if tmin is None else min(tmin, fv)
                                        tmax = fv if tmax is None else max(tmax, fv)
                                except (ValueError, TypeError):
                                    if val != "" and i < len(header):
                                        c = catcols.setdefault(i, {})
                                        c[val] = c.get(val, 0) + 1
                        if nrow >= 500000:
                            break
                    total_rows += nrow
                    stats = {}
                    for i, vals in numcols.items():
                        if vals and i < len(header):
                            s = sorted(vals)
                            stats[header[i] or f"col{i}"] = {
                                "min": round(s[0], 3), "max": round(s[-1], 3),
                                "mean": round(sum(s) / len(s), 3)}
                    # pick the label column: a named categorical col, else any categorical
                    lc_idx = next((i for i, h in enumerate(low)
                                   if h in _label_names and i in catcols), None)
                    if lc_idx is None and catcols:
                        lc_idx = max(catcols, key=lambda i: len(catcols[i]))
                    tbl = {"file": p.name, "rows": nrow, "cols": header[:24], "numeric": stats}
                    if lc_idx is not None:
                        label_col_name = header[lc_idx] if lc_idx < len(header) else f"col{lc_idx}"
                        tbl["label_col"] = label_col_name
                        tbl["label_values"] = catcols[lc_idx]
                        for v, c in catcols[lc_idx].items():
                            label_counts_all[v] = label_counts_all.get(v, 0) + c
                    if tcol is not None and tmin is not None and tmax is not None and tmax > tmin:
                        hz = round((nrow - 1) / (tmax - tmin), 1)
                        tbl["sampling_hz"] = hz
                        rates.append(hz)
                    info["tables"].append(tbl)
            except Exception:
                continue
        # dataset-level rollup the analysis + model can reason over
        info["n_files"] = len(sens)
        info["sampled_files"] = min(len(tabular), _SAMPLE_N)
        info["total_rows"] = total_rows   # over sampled files
        if label_col_name:
            info["label_column"] = label_col_name
            info["class_balance"] = dict(sorted(label_counts_all.items(),
                                                key=lambda kv: -kv[1]))
            info["n_classes"] = len(label_counts_all)
            if info["sampled_files"] < len(sens):
                info["class_balance_note"] = (
                    f"counts over {info['sampled_files']} of {len(sens)} files (sampled)")
        if rates:
            info["sampling_hz_est"] = round(sum(rates) / len(rates), 1)
        out["sensor"] = info

    # ---- POINTCLOUD (npy shape / ply,pcd header point count) ----
    pcs = buckets["pointcloud"]
    if pcs:
        info = {"n": len(pcs), "files": []}
        for p in pcs[:6]:
            pts = None
            try:
                e = p.suffix.lower()
                if e == ".npy":
                    import numpy as _np
                    pts = int(_np.load(str(p), mmap_mode="r").shape[0])
                elif e in (".ply", ".pcd"):
                    with open(p, "rb") as f:
                        for _ in range(80):
                            line = f.readline().decode("latin-1", "replace").strip().lower()
                            if not line:
                                break
                            if e == ".ply" and line.startswith("element vertex"):
                                pts = int(line.split()[-1]); break
                            if e == ".pcd" and line.startswith("points"):
                                pts = int(line.split()[-1]); break
                            if line in ("end_header",):
                                break
            except Exception:
                pass
            info["files"].append({"file": p.name, "points": pts})
        out["pointcloud"] = info

    # ---- TEXT (doc count + avg length) ----
    txts = buckets["text"]
    if txts:
        info = {"n": len(txts), "sampled": 0, "total_kb": 0.0, "avg_chars": 0}
        chars = got = 0
        for p in txts[:60]:
            try:
                info["total_kb"] += p.stat().st_size / 1024
                if got < 25:
                    chars += len(open(p, encoding="utf-8", errors="replace").read(200000))
                    got += 1
            except Exception:
                pass
        info["sampled"] = got
        info["total_kb"] = round(info["total_kb"], 1)
        info["avg_chars"] = int(chars / got) if got else 0
        out["text"] = info

    return out
