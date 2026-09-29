"""The repository's funnel pre-registration as the synthetic test worlds use
it: before the draw.

results/framework/inc/funnel/prereg_v1.json holds the draw's sample lock
(A1, cluster job 47261044) and the copy-detector amendment (A2,
docs/FUNNEL_AUDIT.md 14). A world that draws its own sample needs the prereg
without that lock; everything else, A2 included, is the repository's, written
as funnel.domain.append_amendment writes it (json.dumps(indent=1)). The
world's own sample lock then takes the next free id, A3
(domain.next_amendment_id)."""
import json
import pathlib

REAL = pathlib.Path(__file__).resolve().parents[1] / "results" / "framework" / "inc" / "funnel" / "prereg_v1.json"


def pre_draw_raw(src=REAL):
    raw = json.loads(pathlib.Path(src).read_text())
    raw["amendments"] = [a for a in raw.get("amendments") or [] if a.get("kind") != "sample_lock"]
    return raw


def write_pre_draw(dst, src=REAL):
    """Write the pre-draw prereg to dst (its parent made); returns dst."""
    dst = pathlib.Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(pre_draw_raw(src), indent=1, ensure_ascii=False))
    return dst
