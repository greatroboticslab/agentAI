"""Sensor anomaly detection — pure, dependency-light (stdlib only).

v3.1.9: goes past "how noisy is the data" to "WHERE are the abnormal events".
Grounded, explainable detectors on time-ordered numeric signals:

- sudden-change events : robust first-difference outliers (median + MAD z-score,
  so a signal's normal spread doesn't trigger it); angle columns are unwrapped so
  a 359°→1° heading wrap isn't mistaken for a jump.
- GPS teleports        : implied speed between consecutive lat/lon fixes above a
  physically implausible bound (bad fix / multipath).
- time gaps            : sampling-interval dropouts (dt >> the median dt).
- flatlines            : a normally-varying signal stuck on one value (dead sensor).

Every detector returns concrete events with the timestamp/index so the UI can mark
them on the plot and the review can say exactly when/where. Bounded work.
"""
from __future__ import annotations

import math

_ANGLE_NAMES = ("heading", "yaw", "bearing", "course", "azimuth", "roll", "pitch")
_LAT_NAMES = ("lat", "latitude", "gps_lat")
_LON_NAMES = ("lon", "lng", "longitude", "gps_lon")


def _median(xs: list) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _mad(xs: list, med: float) -> float:
    """Median absolute deviation (robust spread)."""
    return _median([abs(x - med) for x in xs]) if xs else 0.0


def robust_jumps(vals: list, times: list | None = None, is_angle: bool = False,
                 k: float = 6.0, max_events: int = 10) -> list:
    """Sudden-change events via robust z-score on first differences. Returns
    [{index, time, delta, sigma}] for the k-sigma outliers (largest first)."""
    n = len(vals)
    if n < 12:
        return []
    diffs = []
    for i in range(1, n):
        d = vals[i] - vals[i - 1]
        if is_angle:                                   # unwrap ±360
            d = ((d + 180.0) % 360.0) - 180.0
        diffs.append(d)
    med = _median(diffs)
    mad = _mad(diffs, med)
    sigma = 1.4826 * mad
    if sigma <= 1e-12:                                 # near-constant diffs → use std
        mean = sum(diffs) / len(diffs)
        sigma = math.sqrt(sum((d - mean) ** 2 for d in diffs) / len(diffs))
    if sigma <= 1e-12:
        return []
    ev = []
    for i, d in enumerate(diffs):
        z = abs(d - med) / sigma
        if z > k:
            idx = i + 1
            ev.append({"index": idx,
                       "time": round(times[idx], 2) if times and idx < len(times) else None,
                       "delta": round(d, 4), "sigma": round(z, 1)})
    ev.sort(key=lambda e: -e["sigma"])
    return ev[:max_events]


def _haversine_m(la1, lo1, la2, lo2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(la1), math.radians(la2)
    dp = math.radians(la2 - la1)
    dl = math.radians(lo2 - lo1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def gps_teleports(lat: list, lon: list, times: list | None,
                  max_speed_mps: float = 55.0, max_events: int = 10) -> list:
    """Consecutive fixes whose implied speed exceeds a plausible bound → bad GPS."""
    n = min(len(lat), len(lon))
    if n < 3:
        return []
    ev = []
    for i in range(1, n):
        dt = (times[i] - times[i - 1]) if (times and i < len(times)) else 1.0
        if dt <= 0:
            continue
        dist = _haversine_m(lat[i - 1], lon[i - 1], lat[i], lon[i])
        spd = dist / dt
        if spd > max_speed_mps:
            ev.append({"index": i, "time": round(times[i], 2) if times and i < len(times) else None,
                       "jump_m": round(dist, 1), "implied_mps": round(spd, 1)})
    ev.sort(key=lambda e: -e["implied_mps"])
    return ev[:max_events]


def time_gaps(times: list, max_events: int = 8) -> list:
    """Sampling dropouts: dt far above the median interval."""
    if len(times) < 12:
        return []
    dts = [times[i] - times[i - 1] for i in range(1, len(times)) if times[i] > times[i - 1]]
    if len(dts) < 8:
        return []
    med = _median(dts)
    if med <= 0:
        return []
    thr = max(3.0 * med, med + 5.0 * (1.4826 * _mad(dts, med)))
    ev = []
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        if dt > thr:
            ev.append({"index": i, "time": round(times[i], 2), "gap_s": round(dt, 3),
                       "x_median": round(dt / med, 1)})
    ev.sort(key=lambda e: -e["gap_s"])
    return ev[:max_events]


def flatlines(vals: list, times: list | None = None, min_run: int | None = None) -> list:
    """Longest constant run on a normally-varying signal → stuck/dead sensor."""
    n = len(vals)
    if n < 30 or (max(vals) - min(vals)) <= 0:
        return []
    if min_run is None:
        min_run = max(25, n // 20)
    ev = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[j + 1] == vals[i]:
            j += 1
        run = j - i + 1
        if run >= min_run:
            ev.append({"index": i, "time": round(times[i], 2) if times and i < len(times) else None,
                       "value": round(vals[i], 4), "length": run})
        i = j + 1
    ev.sort(key=lambda e: -e["length"])
    return ev[:5]


def detect_table(cols: dict, header: list, time_col: str | None,
                 label_col: str | None) -> dict:
    """Run all detectors on one parsed table. `cols` = {name: [floats]} (time-ordered).
    Returns {n_events, by_type, events[], worst_signal} or {} if nothing found."""
    low = {h.lower(): h for h in header}
    times = None
    if time_col and time_col in cols and len(cols[time_col]) > 2:
        times = cols[time_col]
    events, by_type = [], {}

    def _add(t, **kw):
        by_type[t] = by_type.get(t, 0) + 1
        events.append({"type": t, **kw})

    # per-signal sudden changes + flatlines
    per_sig_count = {}
    for name, vals in cols.items():
        if name in (time_col, label_col) or len(vals) < 12:
            continue
        is_ang = any(a in name.lower() for a in _ANGLE_NAMES)
        for e in robust_jumps(vals, times, is_angle=is_ang):
            _add("sudden_change", signal=name, **e)
            per_sig_count[name] = per_sig_count.get(name, 0) + 1
        for e in flatlines(vals, times):
            _add("flatline", signal=name, **e)

    # GPS teleports (needs lat + lon)
    lat_c = next((low[n] for n in _LAT_NAMES if n in low), None)
    lon_c = next((low[n] for n in _LON_NAMES if n in low), None)
    if lat_c and lon_c:
        for e in gps_teleports(cols.get(lat_c, []), cols.get(lon_c, []), times):
            _add("gps_jump", **e)

    # time gaps
    if times:
        for e in time_gaps(times):
            _add("time_gap", **e)

    if not events:
        return {}
    events.sort(key=lambda e: -(e.get("sigma") or e.get("implied_mps") or
                                e.get("gap_s") or e.get("length") or 0))
    worst = max(per_sig_count, key=per_sig_count.get) if per_sig_count else None
    # keep a generous set (not just the top-14): a strong dominant anomaly (e.g. a GPS
    # teleport) otherwise crowds out weaker-but-real events like a speed dip that must
    # pair with an IMU spike for cross-sensor correlation, or the lone time_gap. The UI
    # still shows only the top few; downstream (cross-modal, focus) needs the fuller set.
    return {"n_events": len(events), "by_type": by_type,
            "events": events[:60], "worst_signal": worst}


def level_label(total_events: int, total_rows: int) -> str:
    """Overall clean / minor / significant from event density."""
    if total_events == 0:
        return "clean"
    rate = total_events / max(total_rows, 1)
    return "minor" if rate < 0.01 else "significant"


_TIME_NAMES = ("t", "time", "timestamp", "ts", "sec", "seconds", "millis", "ms")
_LABEL_NAMES = ("label", "class", "activity", "target", "y", "annotation", "state")


def analyze_dataset_anomalies(root) -> dict | None:
    """Walk csv/tsv sensor files under `root`, detect anomalies per file, roll up.
    Returns {overall_level, total_events, files:[{file, n_events, by_type, events,
    worst_signal}]} or None when there are no tabular sensor files."""
    from pathlib import Path
    import csv as _csv
    root = Path(root)
    tables = [p for p in sorted(root.rglob("*"))
              if p.is_file() and p.suffix.lower() in (".csv", ".tsv")
              and "labels" not in {x.lower() for x in p.parts}]
    if not tables:
        return None
    per_file, total_events, total_rows = [], 0, 0
    for p in tables[:24]:
        try:
            with open(p, newline="", encoding="utf-8", errors="replace") as f:
                _first = f.readline(); f.seek(0)
                delim = ("\t" if p.suffix.lower() == ".tsv"
                         else max([",", ";", "\t", "|"], key=_first.count))
                rd = _csv.reader(f, delimiter=delim)
                header = [str(h).replace("\ufeff", "").strip() for h in next(rd, [])]
                cols = {h: [] for h in header}
                for i, row in enumerate(rd):
                    if i >= 20000:
                        break
                    for j, val in enumerate(row):
                        if j < len(header):
                            try:
                                _fv = float(val)
                                if _fv == _fv and _fv not in (float("inf"), float("-inf")):
                                    cols[header[j]].append(_fv)
                            except (ValueError, TypeError):
                                pass
            cols = {k: v for k, v in cols.items() if v}
            if not cols:
                continue
            low = [h.lower() for h in header]
            tcol = next((header[i] for i, h in enumerate(low)
                         if h in _TIME_NAMES and header[i] in cols), None)
            t_start_abs = None                                # absolute start → cross-file alignment
            if tcol and cols[tcol]:
                t_start_abs = cols[tcol][0]
                if cols[tcol][0] > 1e6:                        # unix ts → seconds from start
                    _t0 = cols[tcol][0]
                    cols[tcol] = [t - _t0 for t in cols[tcol]]
            lcol = next((header[i] for i, h in enumerate(low) if h in _LABEL_NAMES), None)
            total_rows += max((len(v) for v in cols.values()), default=0)
            r = detect_table(cols, header, tcol, lcol)
            if r:
                r["file"] = p.name
                r["t_start_abs"] = t_start_abs
                per_file.append(r)
                total_events += r["n_events"]
        except Exception:
            continue
    return {"overall_level": level_label(total_events, total_rows),
            "total_events": total_events, "files": per_file}
