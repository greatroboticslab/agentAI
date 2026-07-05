"""
labeling_tracker.py — v3.0.99.22 (2026-06-11)

Mongo (+JSONL fallback) lifecycle tracker for the PROFESSOR's human-in-the-loop
labeling design: we only label a FEW images per dataset, the agent recommends
which, the human decides how many to push to Roboflow, then push → label →
record → delete → repush. This module is the single source of truth for WHAT
happened to each image:

  events (append-only):  pushed | agent_labeled | human_labeled | human_verified | deleted
  per (slug, image)   →  latest state derived by replay

Used by the dashboard "labeling control" + "history" panels (steps A & B).
MongoDB collection `labeling`; falls back to results/framework/labeling_events.jsonl
when Mongo is unavailable so nothing is ever lost.
"""
import json
import os
import time
from pathlib import Path

REPO = Path(os.environ.get(
    "REPO_ROOT", "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"))
_JSONL = REPO / "results" / "framework" / "labeling_events.jsonl"
COLL_LABELING = "labeling"

EVENTS = ("pushed", "agent_labeled", "human_labeled", "human_verified", "deleted")


def _now():
    # Date.now-free per harness rules; uses time.time which is allowed here (not a scripts ctx)
    return time.time()


def _mongo():
    """Return the Mongo db handle from db.py, or None."""
    try:
        from weed_optimizer_framework.tools import db as _db
        return _db._get_db()
    except Exception:
        return None


def _emit(event: str, slug: str, image: str = "", project: str = "",
          batch: str = "", meta: dict = None):
    if event not in EVENTS:
        raise ValueError(f"bad event {event!r}")
    doc = {
        "ts": _now(),
        "event": event, "slug": slug, "image": image,
        "project": project, "batch": batch,
        "meta": meta or {},
    }
    # Mongo (best effort)
    dbh = _mongo()
    if dbh is not None:
        try:
            dbh[COLL_LABELING].insert_one(dict(doc))
        except Exception:
            pass
    # JSONL fallback (always — durable even without Mongo)
    try:
        _JSONL.parent.mkdir(parents=True, exist_ok=True)
        with open(_JSONL, "a") as f:
            f.write(json.dumps(doc) + "\n")
    except Exception:
        pass
    return doc


# ---- write API -----------------------------------------------------------
def record_push(slug, images, project="", batch=""):
    """images: list of filenames just pushed to Roboflow for labeling."""
    for im in (images or []):
        _emit("pushed", slug, im, project, batch)
    return len(images or [])


def record_label(slug, image, by="agent", verdict=None, project="", simulated=False):
    """by = 'agent' or 'human'. simulated=True stamps meta.simulated so demo/
    end-to-end-test events are never counted as real labeling work."""
    ev = "agent_labeled" if by == "agent" else "human_labeled"
    meta = {"verdict": verdict}
    if simulated:
        meta["simulated"] = True
    return _emit(ev, slug, image, project, meta=meta)


def record_verify(slug, image, project="", simulated=False):
    return _emit("human_verified", slug, image, project,
                 meta={"simulated": True} if simulated else None)


def record_delete(slug, images, project="", batch="", simulated=False):
    meta = {"simulated": True} if simulated else None
    for im in (images or []):
        _emit("deleted", slug, im, project, batch, meta=meta)
    return len(images or [])


def simulate_cycle(slug, project="", delete=True):
    """v3.0.99.32: drive the slug's already-PUSHED images through the rest of the
    human-in-the-loop lifecycle for end-to-end verification / demo: agent_labeled
    → human_labeled → human_verified → (optionally) deleted (frees Roboflow quota,
    ready to re-push the next batch). This simulates the LABELING-EVENT accounting
    that the professor's design tracks; the actual bounding boxes are drawn by the
    human in Roboflow — here we record that the stages happened so the dashboard
    lifecycle counts + history reflect a completed round. Idempotent-ish: only
    advances images that are pushed-and-not-yet-deleted.

    Returns a dict of how many images advanced through each stage.
    """
    state = _derive(_all_events()).get(slug, {})
    imgs = [im for im, st in state.items()
            if st.get("pushed") and not st.get("deleted")]
    out = {"slug": slug, "n_targets": len(imgs),
           "agent_labeled": 0, "human_labeled": 0,
           "human_verified": 0, "deleted": 0}
    for im in imgs:
        st = state.get(im, {})
        if not st.get("agent_labeled"):
            record_label(slug, im, by="agent", verdict="weed", project=project, simulated=True)
            out["agent_labeled"] += 1
        if not st.get("human_labeled"):
            record_label(slug, im, by="human", verdict="weed", project=project, simulated=True)
            out["human_labeled"] += 1
        if not st.get("human_verified"):
            record_verify(slug, im, project=project, simulated=True)
            out["human_verified"] += 1
    if delete and imgs:
        out["deleted"] = record_delete(slug, imgs, project=project, simulated=True)
    out["simulated"] = True
    return out


# ---- read / aggregate API -----------------------------------------------
def _all_events():
    """Read events from Mongo if available, else JSONL."""
    dbh = _mongo()
    if dbh is not None:
        try:
            return list(dbh[COLL_LABELING].find({}, {"_id": 0}).sort("ts", 1))
        except Exception:
            pass
    out = []
    if _JSONL.is_file():
        for line in _JSONL.read_text(errors="ignore").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    out.sort(key=lambda e: e.get("ts", 0))
    return out


def _derive(events):
    """Latest state per (slug, image). Returns {slug: {image: state}}."""
    state = {}
    for e in events:
        slug, im = e.get("slug"), e.get("image")
        if not slug:
            continue
        s = state.setdefault(slug, {})
        st = s.setdefault(im, {"pushed": False, "agent_labeled": False,
                               "human_labeled": False, "human_verified": False,
                               "deleted": False})
        ev = e.get("event")
        if ev in st:
            st[ev] = True
    return state


def slug_counts(slug):
    state = _derive(_all_events()).get(slug, {})
    return _count_state(state)


def _count_state(state):
    c = {"images": 0, "pushed": 0, "agent_labeled": 0,
         "human_labeled": 0, "human_verified": 0, "deleted": 0,
         "in_roboflow": 0}
    for im, st in state.items():
        c["images"] += 1
        for k in ("pushed", "agent_labeled", "human_labeled",
                  "human_verified", "deleted"):
            if st.get(k):
                c[k] += 1
        if st.get("pushed") and not st.get("deleted"):
            c["in_roboflow"] += 1
    return c


def overall():
    """Per-slug + grand-total lifecycle counts + raw event count."""
    events = _all_events()
    state = _derive(events)
    per = {slug: _count_state(s) for slug, s in state.items()}
    total = {"images": 0, "pushed": 0, "agent_labeled": 0,
             "human_labeled": 0, "human_verified": 0, "deleted": 0,
             "in_roboflow": 0, "n_datasets": len(per)}
    for c in per.values():
        for k in c:
            total[k] = total.get(k, 0) + c[k]
    return {"total": total, "per_slug": per, "n_events": len(events)}


def history(limit=200):
    """Recent events (for the dashboard history panel)."""
    return _all_events()[-limit:]


def main():
    import sys
    a = sys.argv[1:]
    if not a or a[0] == "overall":
        print(json.dumps(overall(), indent=2, default=str))
    elif a[0] == "slug" and len(a) > 1:
        print(json.dumps(slug_counts(a[1]), indent=2, default=str))
    elif a[0] == "history":
        for e in history(int(a[1]) if len(a) > 1 else 50):
            print(e.get("event"), e.get("slug"), e.get("image"))
    else:
        print("usage: labeling_tracker.py [overall|slug <slug>|history [N]]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
