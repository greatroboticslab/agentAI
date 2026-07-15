"""Sensor-data visualization — pure, no FastAPI / app state (de-monolith pattern).

v3.1.4: renders a PNG for tabular sensor data so the analysis page can SHOW the
data, not just numeric stats. Modality-aware:

- GPS-like table (has lat/lon columns)  -> trajectory plot: the route's actual
  shape, equal-aspect, colored by speed when a speed column exists, start/end
  markers. This is what a reviewer wants to see first for driving data.
- Other numeric tables (IMU, generic)   -> time-series plot of up to 4 numeric
  signals against the time column (or row index if no time column).

Matplotlib Agg only (headless server safe). Returns a small dict describing what
was plotted, or None when nothing plottable was found. Bounded: reads at most
`max_rows` rows so a huge log can't stall analysis.
"""
from __future__ import annotations

import csv
from pathlib import Path

_LAT_NAMES = ("lat", "latitude", "gps_lat")
_LON_NAMES = ("lon", "lng", "longitude", "gps_lon")
_TIME_NAMES = ("t", "time", "timestamp", "ts", "sec", "seconds", "millis", "ms")
_SPEED_NAMES = ("speed", "speed_mps", "speed_kmh", "velocity", "v")


def _read_table(path: Path, max_rows: int = 100000):
    """Read a csv/tsv into {column_name: [floats]}, skipping non-numeric cells."""
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        _first = f.readline(); f.seek(0)
        delim = ("\t" if path.suffix.lower() == ".tsv"
                 else max([",", ";", "\t", "|"], key=_first.count))
        rd = csv.reader(f, delimiter=delim)
        header = [str(h).replace("\ufeff", "").strip() for h in next(rd, [])]
        cols = {h: [] for h in header}
        for i, row in enumerate(rd):
            if i >= max_rows:
                break
            for j, val in enumerate(row):
                if j >= len(header):
                    continue
                try:
                    _fv = float(val)
                    if _fv == _fv and _fv not in (float("inf"), float("-inf")):
                        cols[header[j]].append(_fv)
                except (ValueError, TypeError):
                    pass
    return {k: v for k, v in cols.items() if v}


def _find(cols: dict, names) -> str | None:
    low = {k.lower(): k for k in cols}
    for n in names:
        if n in low:
            return low[n]
    return None


def plot_sensor_table(csv_path, out_png) -> dict | None:
    """Render the most useful plot for one tabular sensor file. Returns
    {"kind": "trajectory"|"timeseries", "file": ..., "n_points": ...} or None."""
    csv_path, out_png = Path(csv_path), Path(out_png)
    try:
        cols = _read_table(csv_path)
    except Exception:
        return None
    if not cols:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lat_c, lon_c = _find(cols, _LAT_NAMES), _find(cols, _LON_NAMES)
    if lat_c and lon_c and len(cols[lat_c]) > 1 and len(cols[lat_c]) == len(cols[lon_c]):
        # ---- GPS trajectory: the route's shape ----
        lat, lon = cols[lat_c], cols[lon_c]
        speed_c = _find(cols, _SPEED_NAMES)
        fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=110)
        if speed_c and len(cols[speed_c]) == len(lat):
            sc = ax.scatter(lon, lat, c=cols[speed_c], cmap="viridis", s=6)
            fig.colorbar(sc, ax=ax, label=f"{speed_c}")
        else:
            ax.plot(lon, lat, "-", lw=1.2, color="#2563eb")
        ax.plot(lon[0], lat[0], "o", ms=10, color="#059669", label="start")
        ax.plot(lon[-1], lat[-1], "s", ms=9, color="#dc2626", label="end")
        # v3.1.9: mark GPS anomalies (implausible jumps) on the route
        try:
            from .sensor_anomaly import gps_teleports
            times_c = _find(cols, _TIME_NAMES)
            _tv = cols.get(times_c) if times_c else None
            jt = gps_teleports(lat, lon, _tv if _tv and len(_tv) == len(lat) else None)
            if jt:
                ax.scatter([lon[e["index"]] for e in jt if e["index"] < len(lon)],
                           [lat[e["index"]] for e in jt if e["index"] < len(lat)],
                           marker="x", s=90, c="#dc2626", linewidths=2.5, zorder=5,
                           label="anomaly (%d)" % len(jt))
        except Exception:
            pass
        ax.set_xlabel(lon_c); ax.set_ylabel(lat_c)
        ax.set_title(f"Trajectory — {csv_path.name} ({len(lat)} points)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=.3); ax.legend(loc="best", fontsize=9)
        fig.tight_layout(); fig.savefig(out_png); plt.close(fig)
        return {"kind": "trajectory", "file": csv_path.name, "n_points": len(lat)}

    # ---- generic sensor (IMU etc.): signals over time ----
    time_c = _find(cols, _TIME_NAMES)
    signals = [k for k in cols if k != time_c][:4]
    if not signals:
        return None
    n = min(len(cols[s]) for s in signals)
    if n < 2:
        return None
    x = cols[time_c][:n] if time_c and len(cols[time_c]) >= n else list(range(n))
    if time_c and x and x[0] > 1e9:                     # unix ts -> seconds from start
        x = [v - x[0] for v in x]
    fig, axes = plt.subplots(len(signals), 1, figsize=(7.8, 1.7 * len(signals) + 0.8),
                             dpi=110, sharex=True)
    if len(signals) == 1:
        axes = [axes]
    # v3.1.9: mark sudden-change anomalies on each signal
    try:
        from .sensor_anomaly import robust_jumps
        _ANG = ("heading", "yaw", "bearing", "course", "azimuth", "roll", "pitch")
    except Exception:
        robust_jumps = None
    for ax, s in zip(axes, signals):
        sv = cols[s][:n]
        ax.plot(x, sv, lw=0.9, color="#2563eb")
        if robust_jumps is not None:
            try:
                jj = robust_jumps(sv, x if len(x) == len(sv) else None,
                                  is_angle=any(a in s.lower() for a in _ANG))
                if jj:
                    xi = [x[e["index"]] for e in jj if e["index"] < len(x)]
                    yi = [sv[e["index"]] for e in jj if e["index"] < len(sv)]
                    ax.scatter(xi, yi, marker="x", s=60, c="#dc2626", linewidths=2, zorder=5)
            except Exception:
                pass
        ax.set_ylabel(s, fontsize=9); ax.grid(alpha=.3)
    axes[-1].set_xlabel(f"{time_c} (s from start)" if time_c else "sample #")
    axes[0].set_title(f"Sensor signals — {csv_path.name} ({n} samples)")
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)
    return {"kind": "timeseries", "file": csv_path.name, "n_points": n,
            "signals": signals}


def plot_dataset_sensors(root, out_png) -> dict | None:
    """Find the first plottable csv/tsv under `root` (skipping labels/) and plot
    it. Prefers a GPS-like file (lat/lon) over generic tables."""
    root = Path(root)
    tables = [p for p in sorted(root.rglob("*"))
              if p.is_file() and p.suffix.lower() in (".csv", ".tsv")
              and "labels" not in {x.lower() for x in p.parts}]
    if not tables:
        return None
    # first pass: prefer a file with lat+lon in its header
    def _has_gps(p):
        try:
            head = open(p, encoding="utf-8", errors="replace").readline().lower()
        except Exception:
            return False
        return any(n in head for n in _LAT_NAMES) and any(n in head for n in _LON_NAMES)
    ordered = sorted(tables, key=lambda p: (not _has_gps(p),))
    for p in ordered[:5]:
        r = plot_sensor_table(p, out_png)
        if r:
            return r
    return None
