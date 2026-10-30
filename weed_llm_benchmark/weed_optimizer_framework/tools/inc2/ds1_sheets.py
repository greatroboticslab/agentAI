"""DS1's sheets for people (lever L31, the lab, 0 SU; and `record`, a person's or the delegate's step): the source
sheets, the owner's audit sheets, the LN sheet and the MT-W check sheets, each batch one self-contained HTML file
with every image drawn with every box and its class written, headings in Chinese and English, and one decision box
per source (per image or per box where the item asks for that); its save button writes a decisions file naming the
sheet's sha256, which `record` stores (docs/CONTINUOUS_LOOP.md, Amendment 2026-10-29, DS.0 items 4 and 20, DS.8.2,
DS.9.1, DS.10).

    python -m weed_optimizer_framework.tools.inc2.ds1_sheets render --kind source|audit|ln|mtw --batch N
                                                                     [--lab-inc DIR] [--no-pull]
    python -m weed_optimizer_framework.tools.inc2.ds1_sheets record --file DECISIONS.json --decided-by human:<id>|delegate
    python -m weed_optimizer_framework.tools.inc2.ds1_sheets status            (read-only: what waits for whom)
    python -m weed_optimizer_framework.tools.inc2.ds1_sheets supersede --sheet audit_002 --by audit_001 --why TEXT
                                                                     (the delegate's: an extra render of one batch)

Bundles (the cluster, inside L30's steps; ds1/v1/sheets/bundles/<kind>_<name>/): each sampled image downscaled to a
JPEG of long side 512 (its pixels as stored: an EXIF-tagged image is refused at admission, DS.5.1 item 8), its boxes
(normalised, with the class unit or the species hidden as the sheet asks), the reason a removed image was removed,
and bundle.json listing every file with its sha256. The lab pulls a bundle from the cluster's data-transfer node and
checks every file against bundle.json before it renders (no file is rendered unverified).

The samples (seeded; stable_int texts named below):
  source  per source or file-name family: 24 admitted and 12 removed images with their reasons, and 12 images per
          class unit (inc2/ds1/sheet/<source>/<key>); the decision (accept, accept for training only, reject),
          the setting (Field, Turf, Indoor, Aerial), the class roles (weed, crop, plant part, non-plant,
          unresolved), the capture-name pairs, the model audit's rates;
  audit   the owner's: the 5 decided sheets with the lowest stable_int("inc2/ds1/audit/<source>") among each phase's
          sources (every decided sheet once he reverses one), every source whose class names were resolved in the
          names round, SIU's species-role table; each shows the delegate's decision beside his own box;
  ln      550 images (200 of the SIU increments inc0001, inc0003, inc0004 pooled, 100 of inc0005, 250 of cwd12
          train_core), mixed in the order stable_int("inc2/ds1/ln/<sha256>"), source, file name and species hidden;
          marks per box (correct, not a plant, loose or partial, two plants in one box) and per image the plants
          left unboxed;
  mtw     per source min(60, all) seeded MT-W images and every image whose stored verifier verdicts flag a box;
          per image: in error or not; a correction drawn on the image (boxes removed or added), logged and frozen
          as MT-W v1.1 / v2 labels, never a training label.

Records (ds1/v1/sheets/): rendered/<kind>_<batch>.json (the spec's sha256, the HTML's sha256, the sources or items),
superseded.jsonl (append-only: a rendered sheet another render of the same batch stands for; kept, never recorded,
not counted for its set; Revision 2, item 3), audit_v2/ (the model audit taken again, which later renders show),
decisions.jsonl (append-only: one line per decision with decided_by human:<id> or 'delegate', the sheet's sha256,
the decider's name as the file gives it, the time). A reversal of the delegate's decision on an audit sheet is
recorded as the owner's and widens the audit to every decided sheet.

A refusal raises SheetRefused; the CLI prints "[inc2.ds1_sheets] ERROR: ..." and exits 2.
"""
from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import html
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from ..inc import common as C
from . import common as C2
from . import ds1 as D

FMT_BUNDLE = "inc2-ds1-sheet-bundle/1"
FMT_SPEC = "inc2-ds1-sheet/1"
FMT_DECISIONS = "inc2-ds1-sheet-decisions/1"
FMT_RENDER = "inc2-ds1-sheet-render/1"
FMT_DECISION_ROW = "inc2-ds1-decision/1"
FMT_SUPERSEDED = "inc2-ds1-sheet-superseded/1"
KINDS = ("source", "audit", "ln", "mtw")
BATCH_SOURCES = 4
SAMPLE_ADMITTED, SAMPLE_REMOVED, SAMPLE_UNIT = 24, 12, 12
AUDIT_SAMPLE = 5
THUMB_SIDE = 512
THUMB_QUALITY = 80
LN_GROUPS = D.LN_GROUPS
LN_STREAM = "weed_stream_v1_fork1364"
LN_INCREMENTS = {"siu_increments": ("inc0001", "inc0003", "inc0004"), "inc0005": ("inc0005",)}
MTW_SAMPLE = D.MTW_SAMPLE
FLAG_VERDICTS = ("conflict", "unknown", "rejected", "not_plant")
ROLE_LABELS = {"weed": "杂草 weed", "crop": "作物 crop", "part": "植物部位 plant part", "nonplant": "非植物 non-plant",
               "unresolved": "未定 unresolved"}
DECISION_LABELS = {"accept": "接受 accept", "accept_training_only": "仅用于训练 accept for training only",
                   "reject": "拒绝 reject"}
SETTING_LABELS = {"Field": "田间 Field", "Turf": "草坪 Turf", "Indoor": "室内 Indoor", "Aerial": "航拍 Aerial"}
LN_MARKS = {"correct": "正确 correct", "not_plant": "不是植物 not a plant", "loose": "松散或不全 loose or partial",
            "two_plants": "一框两株 two plants in one box"}
DECIDED_BY_RE = re.compile(r"(human:[A-Za-z0-9_.@+-]{1,80}|delegate)\Z")
# the owner's audit sheets are his (DS.0 item 4: "a reversal is recorded as his"): recorded under his id only
OWNER_HUMAN = "human:harry567566@gmail.com"
MTW_VERDICTS = ("correct", "error")
CLUSTER_DATA_SSH = "byler@data.bridges2.psc.edu"
CLUSTER_INC = "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/inc"


class SheetRefused(D.Ds1Refused):
    pass


def log(msg):
    print("[inc2.ds1_sheets] %s" % msg, flush=True)


# ------------------------------------------------------------------ paths
def sheets_dir(inc=None):
    return (Path(inc) if inc else C.INC_DIR) / "ds1" / D.NAME / "sheets"


def bundle_dir(name, inc=None):
    return sheets_dir(inc) / "bundles" / name


def decisions_path(inc=None):
    return sheets_dir(inc) / "decisions.jsonl"


def rendered_path(kind, batch, inc=None):
    return sheets_dir(inc) / "rendered" / ("%s_%03d.json" % (kind, int(batch)))


def html_path(kind, batch, inc=None):
    return sheets_dir(inc) / "html" / ("ds1_%s_%03d.html" % (kind, int(batch)))


def superseded_path(inc=None):
    return sheets_dir(inc) / "superseded.jsonl"


def audit_v2(inc=None):
    """{(phase, revision, source): the model audit taken again (Revision 2, item 4)} from ds1/v1/sheets/audit_v2/,
    which inc2.ds1_data.reaudit writes beside each screen revision's record (the lab pulls it with the sheets)."""
    from . import ds1_data as DD
    out = {}
    d = sheets_dir(inc) / "audit_v2"
    for p in sorted(d.glob("p[12]_r[0-9][0-9][0-9]_audit_v2.json")) if d.is_dir() else []:
        r = _read_json(p)
        if isinstance(r, dict) and r.get("format") == DD.FMT_REAUDIT:
            for src, v in (r.get("sources") or {}).items():
                out[(int(r.get("phase") or 0), int(r.get("revision") or 0), src)] = v
    return out


def _read_json(p):
    return D._read_json(p)


def _jsonl(p):
    out = []
    p = Path(p)
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def canon(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def sha_obj(obj):
    return hashlib.sha256(canon(obj)).hexdigest()


# ------------------------------------------------------------------ samples (seeded)
def seeded(rows, source, n):
    return sorted(rows, key=lambda r: (C.stable_int(D_SEED_SAMPLE % (source, r["key"])), r["key"]))[:int(n)]


D_SEED_SAMPLE = "inc2/ds1/sheet/%s/%s"


def source_sample(source, admitted, removed, unit_of):
    """The source sheet's images (module docstring): 24 admitted, 12 removed with their reasons, 12 per class unit
    (a unit's images among every candidate of the source)."""
    items = [("admitted", r) for r in seeded(admitted, source, SAMPLE_ADMITTED)]
    items += [("removed", r) for r in seeded(removed, source, SAMPLE_REMOVED)]
    units = collections.defaultdict(list)
    for r in admitted + removed:
        for i, b in enumerate(r.get("boxes") or ()):
            units[unit_of(r, b[0], i)].append(r)
    for u in sorted(units):
        uniq = {r["key"]: r for r in units[u]}
        items += [("unit:%s" % u, r) for r in seeded(list(uniq.values()), "%s/%s" % (source, u), SAMPLE_UNIT)]
    return items


def audit_sample(sources, n=AUDIT_SAMPLE):
    """The owner's seeded audit sample (DS.0 item 4): the n sources with the lowest stable_int("inc2/ds1/audit/
    <source>") (ties by name)."""
    return sorted(sources, key=lambda s: (C.stable_int("inc2/ds1/audit/%s" % s), s))[:int(n)]


def sticky_audit_sample(revisions, n=AUDIT_SAMPLE):
    """The owner's seeded audit sample of a phase over its screen revisions (Revision 1 of DS1, item 3): after each
    revision, the n sources with the lowest stable_int among the sources screened so far (revisions: [[sources of
    revision 1], [of revision 2], ...], role C and wholly quarantined sources left out by the caller); a source once
    drawn stays drawn, so a later revision never takes the audit away from a source it was drawn for."""
    seen, out = [], set()
    for srcs in revisions:
        seen += [s for s in srcs if s not in seen]
        out |= set(audit_sample(seen, n))
    return out


def ln_draw(groups):
    """The LN sheet's 550 images (DS.8.2): per group the seeded draw of its size (stable_int("inc2/ds1/ln/draw/
    <group>/<sha256>")), then all mixed in the order stable_int("inc2/ds1/ln/<sha256>")."""
    out = []
    for g, want in LN_GROUPS.items():
        rows = groups.get(g) or []
        pick = sorted(rows, key=lambda r: (C.stable_int("inc2/ds1/ln/draw/%s/%s" % (g, r["sha256"])), r["key"]))
        if len(pick) < want:
            raise SheetRefused("LN group %s holds %d images, the sheet takes %d" % (g, len(pick), want))
        out += [dict(r, ln_group=g) for r in pick[:want]]
    return sorted(out, key=lambda r: (C.stable_int("inc2/ds1/ln/%s" % r["sha256"]), r["key"]))


def mtw_sample(source, rows, flagged=()):
    """A source's MT-W check images (DS.10 item 4): min(60, all) seeded, plus every image a stored verifier
    verdict flags."""
    pick = sorted(rows, key=lambda r: (C.stable_int("inc2/ds1/mtw/check/%s/%s" % (source, r["key"])), r["key"]))
    take = {r["key"] for r in pick[:MTW_SAMPLE]} | set(flagged)
    return [r for r in pick if r["key"] in take], len(pick) <= MTW_SAMPLE


# ------------------------------------------------------------------ bundles (the cluster)
def thumb(path, side=THUMB_SIDE):
    """(JPEG bytes, (w, h) of the original) of an image downscaled to long side `side` (its stored pixels)."""
    from PIL import Image
    with Image.open(path) as im:
        im.load()
        w, h = im.size
        im = im.convert("RGB")
        s = float(side) / max(w, h)
        if s < 1:
            im = im.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=THUMB_QUALITY)
        return buf.getvalue(), (w, h)


def write_bundle(name, items, meta, inc=None):
    """A bundle (module docstring): items [{"id", "group", "image", "boxes": [[label, cx, cy, w, h]], ...}]; each
    image a JPEG under the bundle named by its sha256; bundle.json written last, listing every file's sha256."""
    d = bundle_dir(name, inc)
    if (d / "bundle.json").is_file():
        return _read_json(d / "bundle.json")
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for it in items:
        try:
            data, wh = thumb(it["image"])
        except Exception as e:  # noqa: BLE001 - an unreadable image is listed so, never dropped silently
            out.append(dict({k: v for k, v in it.items() if k != "image"}, file=None, error=str(e)[:200]))
            continue
        sha = hashlib.sha256(data).hexdigest()
        fn = "%s.jpg" % sha
        if not (d / fn).is_file():
            C2.write_bytes_atomic(d / fn, data)
        out.append(dict({k: v for k, v in it.items() if k != "image"}, file=fn, sha256=sha, wh=list(wh)))
    # seq: the bundles written before this one (each L30 step writes its bundles one after another), so LN and MT-W
    # batches keep their numbers when a later bundle is written (bundles)
    seq = sum(1 for p in d.parent.iterdir() if p.is_dir() and (p / "bundle.json").is_file())
    doc = {"format": FMT_BUNDLE, "name": name, "items": out, "meta": meta, "seq": seq, "written_utc": D._utc()}
    C2.write_json_atomic(d / "bundle.json", doc)
    return doc


def _boxes_units(row, unit_of):
    return [[unit_of(row, b[0], i)] + [float(x) for x in b[1:5]] for i, b in enumerate(row.get("boxes") or ())]


def bundle_sources(phase, revision, by_src, rec, out_rows=None, inc=None):
    """The source sheets' bundles of a screen revision, one per source (ds1_data.screen in production)."""
    from . import ds1_data as DD
    made = []
    for s, rows in sorted(by_src.items()):
        removed = [dict(r, reason=r.get("why")) for r in (out_rows or {}).get(s, [])]
        items = []
        for group, r in source_sample(s, rows, removed, DD.unit_of):
            items.append({"id": "%s|%s|%s" % (s, group, r["key"]), "group": group, "source": s, "key": r["key"],
                          "image": r["image"], "rel": r.get("rel"), "reason": r.get("reason"),
                          "boxes": _boxes_units(r, DD.unit_of)})
        name = "source_p%d_r%03d_%s" % (int(phase), int(revision), DD.slug(s))
        write_bundle(name, items, {"kind": "source", "phase": int(phase), "revision": int(revision), "source": s,
                                   "screen": rec.get(s)}, inc)
        made.append(name)
    return made


# ------------------------------------------------------------------ the batches
def bundles(kind, inc=None):
    """The bundles of a kind: source bundles by name (source_chunks keeps each screen revision's batches apart);
    LN and MT-W bundles in the order they were written (bundle.json's seq, then its time and name), so a bundle
    written later (a new source's MT-W v2) never renumbers an earlier batch."""
    d = sheets_dir(inc) / "bundles"
    pre = {"source": "source_", "audit": "source_", "ln": "ln_", "mtw": "mtw_"}[kind]
    names = sorted(p.name for p in d.iterdir() if p.is_dir() and p.name.startswith(pre)) if d.is_dir() else []
    if kind in ("ln", "mtw"):
        def order(n):
            b = _read_json(d / n / "bundle.json") or {}
            return (int(b.get("seq") if b.get("seq") is not None else 10 ** 9), str(b.get("written_utc") or "9999"), n)
        names = sorted(names, key=order)
    return names


def batch_bundles(kind, batch, inc=None, decisions=None, sources=None):
    """The bundles a batch renders: source sheets in chunks of BATCH_SOURCES in bundle order (batch 1 = the
    first chunk); the audit batch the sources the platform names (sources: the cluster's audit set, which the
    ticker passes) or else the audit set the decisions give; ln and mtw one bundle per batch."""
    names = bundles(kind, inc)
    if kind == "source":
        chunks = source_chunks(names)
        if not 1 <= int(batch) <= len(chunks):
            raise SheetRefused("source batch %s does not exist (%d batches)" % (batch, len(chunks)))
        return chunks[int(batch) - 1]
    if kind == "audit":
        want = set(sources) if sources is not None else audit_sources(
            decisions if decisions is not None else fold(inc))
        return [n for n in names if (_read_json(bundle_dir(n, inc) / "bundle.json") or {}).get("meta", {}).get(
            "source") in want]
    if not 1 <= int(batch) <= len(names):
        raise SheetRefused("%s batch %s does not exist (%d bundles)" % (kind, batch, len(names)))
    return [names[int(batch) - 1]]


def source_chunks(names):
    """The source batches: each screen revision's bundles (written together) in chunks of BATCH_SOURCES, revision by
    revision, so a later revision never changes an earlier batch."""
    by = {}
    for n in sorted(names):
        m = re.match(r"(source_p\d_r\d{3})_", n)
        by.setdefault(m.group(1) if m else "", []).append(n)
    out = []
    for rev in sorted(by):
        ns = by[rev]
        out += [ns[i:i + BATCH_SOURCES] for i in range(0, len(ns), BATCH_SOURCES)]
    return out


def verify_bundle(name, inc=None):
    """The bundle's items with every file checked against bundle.json's sha256 (refused otherwise)."""
    d = bundle_dir(name, inc)
    doc = _read_json(d / "bundle.json")
    if not isinstance(doc, dict) or doc.get("format") != FMT_BUNDLE:
        raise SheetRefused("%s holds no bundle.json" % d)
    for it in doc["items"]:
        if it.get("file") and C.sha256_file(d / it["file"]) != it.get("sha256"):
            raise SheetRefused("%s/%s does not hash as bundle.json records: the sheet is not rendered" % (d, it["file"]))
    return doc


def _rsync_env():
    env = dict(os.environ, SSH_ASKPASS=str(Path.home() / ".cluster_askpass.sh"), SSH_ASKPASS_REQUIRE="force",
               DISPLAY=":0")
    env.pop("SSH_AUTH_SOCK", None)
    return env


RSH = "ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o IdentityAgent=none -o NumberOfPasswordPrompts=1"


def push_render(kind, batch, lab_inc, runner=None):
    """The render record to the cluster (record runs there, against the sheets the lab rendered)."""
    run = runner or subprocess.run
    src = rendered_path(kind, batch, lab_inc)
    p = run(["setsid", "-w", "rsync", "-a", "-e", RSH, str(src), "%s:%s/ds1/%s/sheets/rendered/" % (
        CLUSTER_DATA_SSH, CLUSTER_INC, D.NAME)], capture_output=True, text=True, env=_rsync_env(), timeout=600)
    if p.returncode != 0:
        raise SheetRefused("the push of %s exited %d: %s" % (src.name, p.returncode, (p.stderr or "")[-300:]))
    return True


def pull_state(lab_inc, runner=None):
    """The lab's copy of the cluster's sheets state before a render: every bundle and the decisions recorded so far
    (rsync of ds1/v1/sheets/ from the data-transfer node, the lab's own renders and HTML left as they are). A render
    lists its batch's bundles only after this pull, so a batch the cluster wrote is never 'missing' on the lab."""
    run = runner or subprocess.run
    dst = sheets_dir(lab_inc)
    dst.mkdir(parents=True, exist_ok=True)
    p = run(["setsid", "-w", "rsync", "-a", "--exclude", "rendered/", "--exclude", "html/", "-e", RSH,
             "%s:%s/ds1/%s/sheets/" % (CLUSTER_DATA_SSH, CLUSTER_INC, D.NAME), str(dst) + "/"],
            capture_output=True, text=True, env=_rsync_env(), timeout=3600)
    if p.returncode != 0:
        raise SheetRefused("the pull of the sheets' state exited %d: %s" % (p.returncode, (p.stderr or "")[-300:]))
    return True


def pull_bundle(name, lab_inc, runner=None):
    """The lab's copy of a cluster bundle (rsync from the data-transfer node with the lab's askpass helper)."""
    run = runner or subprocess.run
    dst = bundle_dir(name, lab_inc)
    dst.mkdir(parents=True, exist_ok=True)
    p = run(["setsid", "-w", "rsync", "-a", "-e", RSH, "%s:%s/ds1/%s/sheets/bundles/%s/" % (
        CLUSTER_DATA_SSH, CLUSTER_INC, D.NAME, name), str(dst) + "/"], capture_output=True, text=True,
        env=_rsync_env(), timeout=3600)
    if p.returncode != 0:
        raise SheetRefused("the pull of bundle %s exited %d: %s" % (name, p.returncode, (p.stderr or "")[-300:]))
    return verify_bundle(name, lab_inc)


# ------------------------------------------------------------------ the HTML (one self-contained file per batch)
CSS = """
:root{--bg:#f7f6f2;--ink:#1d1d1b;--muted:#6b6a65;--line:#d9d6cc;--card:#fff;--accent:#2f6f4f;--warn:#a5532a;
--box:#e6007e;--box2:#00a2d4}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#171716;--ink:#ecebe6;--muted:#a3a198;
--line:#3a3935;--card:#21211f;--accent:#7cc39b;--warn:#e39a72}}
:root[data-theme="dark"]{--bg:#171716;--ink:#ecebe6;--muted:#a3a198;--line:#3a3935;--card:#21211f;--accent:#7cc39b;
--warn:#e39a72}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 -apple-system,
"PingFang SC","Helvetica Neue",Arial,sans-serif;padding:16px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:18px;margin:28px 0 8px;border-top:2px solid var(--line);padding-top:16px}
h3{font-size:15px;margin:16px 0 6px;color:var(--muted)}.meta{color:var(--muted);font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:10px}
.cell{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:6px;position:relative}
.cell svg{width:100%;height:auto;display:block;border-radius:4px}.cap{font-size:12px;color:var(--muted);
word-break:break-all;margin-top:4px}.box{fill:none;stroke:var(--box);stroke-width:2}.box.off{stroke:#888;
stroke-dasharray:4 3}.box.new{stroke:var(--box2)}.lab{fill:var(--box);font-size:13px;font-weight:600;
paint-order:stroke;stroke:#fff;stroke-width:3px}
.decide{background:var(--card);border:2px solid var(--accent);border-radius:8px;padding:12px;margin:10px 0}
.decide label{margin-right:14px;display:inline-block}.decide table{border-collapse:collapse;margin-top:8px}
.decide td,.decide th{border:1px solid var(--line);padding:4px 8px;font-size:13px;text-align:left}
.delegate{color:var(--warn);font-size:13px}.bar{position:sticky;top:0;background:var(--bg);padding:8px 0;
border-bottom:1px solid var(--line);z-index:2;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
button{background:var(--accent);color:#fff;border:0;border-radius:6px;padding:8px 14px;font-size:15px;cursor:pointer}
input[type=text]{padding:6px;border:1px solid var(--line);border-radius:4px;background:var(--card);color:var(--ink)}
select{padding:3px;background:var(--card);color:var(--ink);border:1px solid var(--line)}
.err{outline:3px solid var(--warn)}
"""

JS = r"""
const SPEC = JSON.parse(document.getElementById('spec').textContent);
function val(name){const e=document.querySelector('[name="'+CSS.escape(name)+'"]:checked')||
  document.querySelector('select[name="'+CSS.escape(name)+'"]')||document.querySelector('input[name="'+CSS.escape(name)+'"]');
  return e?(e.type==='checkbox'?e.checked:e.value):null;}
const corr = {};
function svgPt(svg, ev){const r=svg.getBoundingClientRect();return [(ev.clientX-r.left)/r.width,(ev.clientY-r.top)/r.height];}
document.querySelectorAll('svg[data-edit]').forEach(svg=>{
  const id=svg.dataset.edit; corr[id]={removed:[],added:[]}; let start=null, rect=null;
  svg.querySelectorAll('rect.box').forEach((b,i)=>b.addEventListener('click',e=>{e.stopPropagation();
    b.classList.toggle('off'); const k=corr[id].removed, j=k.indexOf(i); if(j<0)k.push(i); else k.splice(j,1);}));
  svg.addEventListener('mousedown',e=>{start=svgPt(svg,e);rect=document.createElementNS('http://www.w3.org/2000/svg','rect');
    rect.setAttribute('class','box new');svg.appendChild(rect);});
  svg.addEventListener('mousemove',e=>{if(!start)return;const p=svgPt(svg,e),W=svg.viewBox.baseVal.width,H=svg.viewBox.baseVal.height;
    rect.setAttribute('x',Math.min(p[0],start[0])*W);rect.setAttribute('y',Math.min(p[1],start[1])*H);
    rect.setAttribute('width',Math.abs(p[0]-start[0])*W);rect.setAttribute('height',Math.abs(p[1]-start[1])*H);});
  svg.addEventListener('mouseup',e=>{if(!start)return;const p=svgPt(svg,e);const w=Math.abs(p[0]-start[0]),h=Math.abs(p[1]-start[1]);
    if(w>0.005&&h>0.005){corr[id].added.push([(p[0]+start[0])/2,(p[1]+start[1])/2,w,h]);}else{rect.remove();}start=null;});
});
function collect(){
  const out={format:'inc2-ds1-sheet-decisions/1',sheet:{kind:SPEC.kind,batch:SPEC.batch,spec_sha256:SPEC.spec_sha256},
    decider:document.getElementById('decider').value.trim(),saved_utc:new Date().toISOString().replace(/\.\d+Z$/,'Z')};
  if(!out.decider){alert('请填写决定人 / Fill in who decides'); return null;}
  const missing=[];
  if(SPEC.kind==='source'||SPEC.kind==='audit'){out.sources={};
    for(const s of SPEC.sources){const d={decision:val('dec|'+s.source),setting:val('set|'+s.source),
      family:val('fam|'+s.source)||null,series:val('ser|'+s.source)===true,note:val('note|'+s.source)||'',roles:{}};
      for(const u of s.units){d.roles[u]=val('role|'+s.source+'|'+u); if(!d.roles[u])missing.push(s.source+' '+u);}
      if(!d.decision)missing.push(s.source+': decision'); if(!d.setting)missing.push(s.source+': setting');
      out.sources[s.source]=d;}}
  else {out.images={};
    for(const it of SPEC.items){const r={};
      if(SPEC.kind==='ln'){r.marks=it.boxes.map((b,i)=>val('mark|'+it.id+'|'+i)||'correct');
        r.missing=parseInt(val('miss|'+it.id)||'0',10)||0; r.checked=val('seen|'+it.id)===true;
        if(!r.checked)missing.push(it.id);}
      else {r.verdict=val('ok|'+it.id); r.error=r.verdict==='error'; r.removed=(corr[it.id]||{}).removed||[];
        r.added=(corr[it.id]||{}).added||[]; r.note=val('note|'+it.id)||''; if(!r.verdict)missing.push(it.id);}
      out.images[it.id]=r;}}
  if(missing.length){alert('尚未作答 / not answered yet ('+missing.length+'): '+missing.slice(0,8).join('; '));
    return null;}
  return out;}
function save(){const d=collect(); if(!d)return;
  const b=new Blob([JSON.stringify(d,null,1)],{type:'application/json'});const a=document.createElement('a');
  a.href=URL.createObjectURL(b);a.download='ds1_decisions_'+SPEC.kind+'_'+String(SPEC.batch).padStart(3,'0')+'_'+
  SPEC.spec_sha256.slice(0,12)+'.json';document.body.appendChild(a);a.click();a.remove();}
document.getElementById('save').addEventListener('click',save);
"""


def _svg(item, data_uri, label_mode="unit", edit=False):
    w, h = item.get("wh") or (1, 1)
    vb_w, vb_h = 1000, int(round(1000.0 * h / max(1, w)))
    parts = ['<svg viewBox="0 0 %d %d"%s xmlns="http://www.w3.org/2000/svg">' % (
        vb_w, vb_h, (' data-edit="%s"' % html.escape(item["id"])) if edit else ""),
        '<image href="%s" x="0" y="0" width="%d" height="%d"/>' % (data_uri, vb_w, vb_h)]
    for i, b in enumerate(item.get("boxes") or ()):
        lab, cx, cy, bw, bh = b[0], float(b[1]), float(b[2]), float(b[3]), float(b[4])
        x, y = (cx - bw / 2) * vb_w, (cy - bh / 2) * vb_h
        parts.append('<rect class="box" x="%.1f" y="%.1f" width="%.1f" height="%.1f"/>' % (x, y, bw * vb_w, bh * vb_h))
        text = {"unit": str(lab), "index": "#%d" % (i + 1), "weed": "weed #%d" % (i + 1)}[label_mode]
        parts.append('<text class="lab" x="%.1f" y="%.1f">%s</text>' % (x + 3, max(14.0, y - 4), html.escape(text)))
    parts.append("</svg>")
    return "".join(parts)


def _radio(name, opts, chosen=None):
    return "".join('<label><input type="radio" name="%s" value="%s"%s> %s</label>' % (
        html.escape(name), html.escape(k), " checked" if k == chosen else "", html.escape(v)) for k, v in opts.items())


def _select(name, opts, chosen=None, blank=False):
    """A select; blank: an empty first option chosen (the person must choose: an untouched sheet does not save)."""
    first = '<option value="" selected>— 请选择 choose —</option>' if blank else ""
    return '<select name="%s">%s%s</select>' % (html.escape(name), first, "".join(
        '<option value="%s"%s>%s</option>' % (html.escape(k), " selected" if (k == chosen and not blank) else "",
                                              html.escape(v)) for k, v in opts.items()))


def render_html(spec, files):
    """One self-contained HTML page of a batch: spec (the sheet's content, its sha256 inside), files {file name:
    JPEG bytes}."""
    from . import ds1_data as DD
    kind = spec["kind"]
    title = {"source": "来源审核表 Source sheets", "audit": "负责人抽查 Owner's audit",
             "ln": "逐框标注检查 LN: per-box label sheet", "mtw": "主测试集检查 MT-W check"}[kind]
    out = ['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width,initial-scale=1">',
           '<title>DS1 %s %03d</title><style>%s</style></head><body>' % (kind, int(spec["batch"]), CSS),
           '<h1>%s — DS1 batch %03d</h1>' % (title, int(spec["batch"])),
           '<div class="meta">sheet sha256 %s · %d images · rendered %s</div>' % (
               spec["spec_sha256"][:16], sum(len(s.get("items") or ()) for s in spec.get("sources") or []) +
               len(spec.get("items") or ()), spec.get("rendered_utc")),
           '<div class="bar"><label>决定人 Decider: <input type="text" id="decider" placeholder="name"></label>'
           '<button id="save">保存决定 Save decisions</button><span class="meta">The file names this sheet\'s sha256; '
           'record it with inc2.ds1_sheets record.</span></div>']

    def uri(fn):
        return "data:image/jpeg;base64," + base64.b64encode(files[fn]).decode("ascii") if fn in files else ""
    if kind in ("source", "audit"):
        for s in spec["sources"]:
            src = s["source"]
            out.append('<h2>来源 Source: %s</h2>' % html.escape(src))
            scr = s.get("screen") or {}
            av = scr.get("audit_verdict")
            out.append('<div class="meta">rows %s · live %s · out %s · cwd12 guard hits %s · capture-name pairs %s · '
                       'convention %s · %s %s</div>' % (
                           scr.get("rows"), scr.get("live"), html.escape(json.dumps(scr.get("out") or {})),
                           scr.get("cwd12_hits"), html.escape(json.dumps((scr.get("capture_names") or {}).get(
                               "examples") or [])[:600]), html.escape(json.dumps((scr.get("convention") or {}).get(
                                   "fails") or "passes")),
                           "model audit (Revision 2)" if DD.audit_valid(av) or av is None else
                           "model audit taken before Revision 2 (its rates read each box's class as a coordinate: "
                           "not valid; it is taken again before any pool)", html.escape(json.dumps(av)[:300])))
            prev = s.get("delegate") or {}
            if scr.get("role_c") or scr.get("quarantine_whole"):
                out.append('<div class="delegate"><b>%s</b></div>' % html.escape(
                    "角色C：文件名含 cwd12 / 3SeasonWeedDet10 拍摄名 role C: its file names carry cwd12 or 3SeasonWeedDet10 "
                    "capture names (DS.5.1 1(b)); the pairs are listed above" if scr.get("role_c") else
                    "整源隔离：cwd12 dev/test 副本 quarantined whole: a copy of a cwd12 dev or test image (D28-v2)") +
                    " — this source enters no DS1 manifest whatever is decided</div>")
            # every answer is given on the sheet: nothing is pre-selected on the owner's audit (the delegate's
            # decision is shown beside each box), so a sheet saved untouched cannot pass
            out.append('<div class="decide"><b>决定 Decision</b><br>%s' % _radio("dec|" + src, DECISION_LABELS, None))
            if prev:
                out.append('<div class="delegate">代理人的决定 The delegate decided: %s</div>' % html.escape(
                    json.dumps(prev, ensure_ascii=False)))
            blank = kind == "audit"
            out.append('<br><b>场景 Setting</b> %s' % _select("set|" + src, SETTING_LABELS, None if blank else
                                                              prev.get("setting"), blank=blank))
            out.append(' <b>族 Family</b> <input type="text" name="fam|%s" value="%s" size="10">' % (
                html.escape(src), html.escape(prev.get("family") or s.get("family") or "")))
            out.append(' <label><input type="checkbox" name="ser|%s"%s> 视频或托盘系列 video or tray series</label>' % (
                html.escape(src), " checked" if s.get("series") else ""))
            out.append('<table><tr><th>类别单元 Class unit</th><th>来源类名 Source names</th><th>框数 Boxes</th>'
                       '<th>角色 Role</th></tr>')
            for u in s["units"]:
                info = (s.get("unit_info") or {}).get(u) or {}
                dr = (prev.get("roles") or {}).get(u)
                out.append('<tr><td>%s</td><td>%s</td><td>%s</td><td>%s%s</td></tr>' % (
                    html.escape(u), html.escape(", ".join(info.get("names") or [])), info.get("boxes", ""),
                    _select("role|%s|%s" % (src, u), ROLE_LABELS, None if blank else (dr or info.get("suggested")),
                            blank=blank),
                    (' <span class="delegate">delegate: %s</span>' % html.escape(str(dr))) if blank and dr else ""))
            out.append('</table><br>备注 Note <input type="text" name="note|%s" size="60"></div>' % html.escape(src))
            groups = collections.OrderedDict()
            for it in s["items"]:
                groups.setdefault(it["group"], []).append(it)
            for g, its in groups.items():
                head = {"admitted": "入选样本 Admitted (24)", "removed": "被移除样本 Removed (12), with the reason"}.get(
                    g, "类别单元 Class unit %s (12)" % g[5:])
                out.append('<h3>%s</h3><div class="grid">' % html.escape(head))
                for it in its:
                    if not it.get("file"):
                        out.append('<div class="cell">unreadable: %s</div>' % html.escape(it.get("error") or ""))
                        continue
                    out.append('<div class="cell">%s<div class="cap">%s%s</div></div>' % (
                        _svg(it, uri(it["file"]), "unit"), html.escape(it.get("rel") or it.get("key") or ""),
                        (" — " + html.escape(str(it["reason"]))) if it.get("reason") else ""))
                out.append("</div>")
    else:
        out.append('<h2>%s</h2><div class="grid">' % ("每框选择；每图填写漏标株数 Mark every box; count the plants left "
                                                     "unboxed" if kind == "ln" else "错误：漏框、作物或空框、松散或不全；点框删除，"
                                                     "拖动添加 Error: a weed left unboxed, a box on a crop or on no plant, "
                                                     "a loose or partial box; click a box to remove it, drag to add one"))
        for it in spec["items"]:
            if not it.get("file"):
                continue
            cell = ['<div class="cell" id="c-%s">' % html.escape(it["id"]),
                    _svg(it, uri(it["file"]), "index" if kind == "ln" else "weed", edit=(kind == "mtw"))]
            if kind == "ln":
                for i, _b in enumerate(it.get("boxes") or ()):
                    cell.append('<div>#%d %s</div>' % (i + 1, _select("mark|%s|%d" % (it["id"], i), LN_MARKS,
                                                                       "correct")))
                cell.append('<div>漏标株数 Plants unboxed <input type="text" name="miss|%s" size="3" value="0"></div>'
                            % html.escape(it["id"]))
                cell.append('<div><label><input type="checkbox" name="seen|%s"> 已检查此图 I checked this image'
                            '</label></div>' % html.escape(it["id"]))
            else:
                cell.append('<div>%s <input type="text" name="note|%s" size="22" placeholder="note"></div>' % (
                    _radio("ok|" + it["id"], {"correct": "正确 correct", "error": "有错 in error"}, None),
                    html.escape(it["id"])))
                if it.get("flagged"):
                    cell.append('<div class="delegate">verifier flag</div>')
            cell.append('<div class="cap">%s</div></div>' % html.escape(it.get("caption") or ""))
            out.append("".join(cell))
        out.append("</div>")
    out.append('<script type="application/json" id="spec">%s</script>' % html.escape(
        json.dumps({k: v for k, v in spec.items() if k != "files"}, ensure_ascii=False), quote=False))
    out.append("<script>%s</script></body></html>" % JS)
    return "\n".join(out).encode("utf-8")


def sheet_spec(kind, batch, names, inc=None, decisions=None):
    """The sheet's content (what its sha256 covers): its bundles' items and, for source and audit sheets, each
    source's units, their source names and suggestions, and (audit) the delegate's decision."""
    decs = decisions if decisions is not None else fold(inc)
    av2 = audit_v2(inc) if kind in ("source", "audit") else {}
    spec = {"format": FMT_SPEC, "kind": kind, "batch": int(batch), "bundles": [], "sources": [], "items": []}
    files = {}
    for n in names:
        doc = verify_bundle(n, inc)
        spec["bundles"].append({"name": n, "sha256": C.sha256_file(bundle_dir(n, inc) / "bundle.json")})
        for it in doc["items"]:
            if it.get("file"):
                files[it["file"]] = (bundle_dir(n, inc) / it["file"]).read_bytes()
        meta = doc.get("meta") or {}
        if kind in ("source", "audit"):
            scr = meta.get("screen") or {}
            ra = av2.get((int(meta.get("phase") or 0), int(meta.get("revision") or 0), meta.get("source")))
            if ra:
                # the model audit taken again (Revision 2, item 4) in place of the screen's, which is kept beside it
                scr = dict(scr, audit_verdict=ra.get("audit_verdict"), audit_superseded=ra.get("superseded"))
            units = sorted((scr.get("units") or {}).keys())
            spec["sources"].append({
                "source": meta.get("source"), "units": units, "screen": scr, "family": scr.get("family"),
                "series": scr.get("series"),
                "unit_info": {u: dict(v, names=(scr.get("unit_names") or {}).get(u) or []) for u, v in
                              (scr.get("units") or {}).items()},
                "delegate": ((decs.get(meta.get("source")) or {}).get("delegate_decision") if kind == "audit"
                             else None), "items": doc["items"]})
        else:
            spec["items"] += doc["items"]
    spec["spec_sha256"] = sha_obj({k: v for k, v in spec.items() if k != "spec_sha256"})
    return spec, files


def render(kind, batch, inc=None, pull=False, runner=None, out_html=None, sources=None):
    """L31: render one batch (with pull: the cluster's sheets state copied to the lab first, pull_state, so the batch's
    bundles are listed from what the cluster wrote; every file checked against its bundle.json), write the HTML and
    its render record (the spec's and the file's sha256), and push the record back. sources: the audit batch's
    sources as the platform names them (the cluster's audit set)."""
    if kind not in KINDS:
        raise SheetRefused("--kind %s, not %r" % ("|".join(KINDS), kind))
    if kind == "audit" and pull and sources is None:
        raise SheetRefused("an audit batch rendered on the lab names its sources (--sources, the cluster's audit set)")
    if pull:
        pull_state(inc, runner)
    names = batch_bundles(kind, batch, inc, sources=sources)
    for n in names:
        verify_bundle(n, inc)
    if not names:
        raise SheetRefused("the %s batch %s has no bundle" % (kind, batch))
    spec, files = sheet_spec(kind, batch, names, inc)
    spec["rendered_utc"] = D._utc()
    data = render_html(spec, files)
    hp = Path(out_html) if out_html else html_path(kind, batch, inc)
    C2.write_bytes_atomic(hp, data)
    rec = {"format": FMT_RENDER, "kind": kind, "batch": int(batch), "spec_sha256": spec["spec_sha256"],
           "html": str(hp), "html_sha256": hashlib.sha256(data).hexdigest(), "bundles": spec["bundles"],
           "sources": [s["source"] for s in spec["sources"]], "items": [it["id"] for it in spec["items"]],
           "item_boxes": {it["id"]: it.get("boxes") for it in spec["items"]},
           "item_groups": {it["id"]: it.get("ln_group") or it.get("source") for it in spec["items"]},
           "item_totals": {it["id"]: it.get("total") for it in spec["items"] if it.get("total") is not None},
           "rendered_utc": spec["rendered_utc"]}
    C2.write_json_atomic(rendered_path(kind, batch, inc), rec)
    if pull:
        push_render(kind, batch, inc, runner)
    log("%s batch %d: %s (%d bytes, spec %s)" % (kind, int(batch), hp, len(data), spec["spec_sha256"][:12]))
    return rec


# ------------------------------------------------------------------ record (a person's or the delegate's)
def renders(inc=None):
    d = sheets_dir(inc) / "rendered"
    out = {}
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        r = _read_json(p)
        if isinstance(r, dict) and r.get("format") == FMT_RENDER:
            out[r["spec_sha256"]] = r
    return out


def superseded(inc=None):
    """{sheet name (kind_batch): its row} of the rendered sheets marked superseded (kept, never recorded)."""
    return {r.get("sheet"): r for r in _jsonl(superseded_path(inc)) if r.get("format") == FMT_SUPERSEDED}


def supersede(sheet, by, why, inc=None):
    """Mark a rendered sheet superseded by another of its kind with the same sources (2026-10-29: the owner's audit
    batch rendered again on every tick for one set; Revision 2, item 3): the sheet and its HTML stay, `record` refuses
    a decisions file naming it, the platform no longer counts it as that set's render. The delegate's, appended
    once per sheet to ds1/v1/sheets/superseded.jsonl."""
    rend = {"%s_%03d" % (r["kind"], r["batch"]): r for r in renders(inc).values()}
    if sheet not in rend or by not in rend:
        raise SheetRefused("supersede names rendered sheets: %s and %s are %s" % (
            sheet, by, ["rendered" if x in rend else "not rendered" for x in (sheet, by)]))
    if sheet == by:
        raise SheetRefused("a sheet is not superseded by itself")
    a, b = rend[sheet], rend[by]
    if a["kind"] != b["kind"] or sorted(a.get("sources") or []) != sorted(b.get("sources") or []) or \
            sorted(a.get("items") or []) != sorted(b.get("items") or []):
        raise SheetRefused("%s and %s are not the same batch (kind, sources, items)" % (sheet, by))
    sup = superseded(inc)
    if sheet in sup:
        raise SheetRefused("%s is superseded already (by %s)" % (sheet, sup[sheet].get("by")))
    if by in sup:
        raise SheetRefused("%s is itself superseded" % by)
    if any(r.get("sheet_sha256") == a["spec_sha256"] for r in _jsonl(decisions_path(inc))):
        raise SheetRefused("%s has recorded decisions: it is not superseded" % sheet)
    if not str(why or "").strip():
        raise SheetRefused("supersede needs --why")
    row = {"format": FMT_SUPERSEDED, "sheet": sheet, "spec_sha256": a["spec_sha256"], "html_sha256": a["html_sha256"],
           "by": by, "by_spec_sha256": b["spec_sha256"], "why": str(why), "decided_by": "delegate",
           "decider": D.DELEGATE, "recorded_utc": D._utc()}
    p = superseded_path(inc)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return row


def audit_sets(inc=None):
    """{sheet name: its sources (sorted)} of the owner's audit batches rendered and not superseded: the sets the
    platform does not render again (Revision 2, item 3)."""
    sup = superseded(inc)
    return {"%s_%03d" % (r["kind"], r["batch"]): sorted(r.get("sources") or []) for r in renders(inc).values()
            if r.get("kind") == "audit" and "%s_%03d" % (r["kind"], r["batch"]) not in sup}


def record(path, decided_by, inc=None):
    """Store a saved decisions file (DS.9.1): it must name a rendered sheet by its spec sha256 and decide what
    that sheet asks; each decision is one appended line with decided_by (human:<id> for a person, 'delegate' for
    the delegate under the grant), the sheet's sha256, the decider's name and the time. A file recorded once is
    never recorded again."""
    if not DECIDED_BY_RE.match(str(decided_by or "")):
        raise SheetRefused("--decided-by human:<id> or delegate, not %r" % decided_by)
    raw = Path(path).read_bytes()
    doc = json.loads(raw.decode("utf-8"))
    if not isinstance(doc, dict) or doc.get("format") != FMT_DECISIONS:
        raise SheetRefused("%s is not a DS1 decisions file" % path)
    sh = (doc.get("sheet") or {}).get("spec_sha256")
    rend = renders(inc).get(sh)
    if rend is None:
        raise SheetRefused("%s names sheet %s, which no rendered sheet is" % (path, str(sh)[:12]))
    kind = rend["kind"]
    sup = {r.get("spec_sha256"): r for r in superseded(inc).values()}
    if sh in sup:
        raise SheetRefused("%s names sheet %s, superseded by %s (%s): decide that sheet instead" % (
            path, sup[sh].get("sheet"), sup[sh].get("by"), sup[sh].get("why")))
    if kind == "audit" and decided_by != OWNER_HUMAN:
        raise SheetRefused("the audit sheets are the owner's (DS.0 item 4): decided_by must be %s, not %r"
                           % (OWNER_HUMAN, decided_by))
    if not str(doc.get("decider") or "").strip():
        raise SheetRefused("%s names no decider" % path)
    file_sha = hashlib.sha256(raw).hexdigest()
    have = _jsonl(decisions_path(inc))
    if any(r.get("file_sha256") == file_sha for r in have):
        raise SheetRefused("%s was recorded already" % path)
    rows = []
    base = {"format": FMT_DECISION_ROW, "kind": kind, "batch": rend["batch"], "sheet_sha256": sh,
            "html_sha256": rend["html_sha256"], "file_sha256": file_sha, "decided_by": decided_by,
            "decider": doc.get("decider"), "saved_utc": doc.get("saved_utc"), "recorded_utc": D._utc()}
    if kind in ("source", "audit"):
        want = set(rend["sources"])
        got = doc.get("sources") or {}
        if set(got) != want:
            raise SheetRefused("the file decides %s, the sheet asks %s" % (sorted(got), sorted(want)))
        for s, d in sorted(got.items()):
            if d.get("decision") not in ("accept", "accept_training_only", "reject"):
                raise SheetRefused("%s: decision %r" % (s, d.get("decision")))
            if d.get("setting") not in SETTING_LABELS:
                raise SheetRefused("%s: setting %r" % (s, d.get("setting")))
            bad = [u for u, r in (d.get("roles") or {}).items() if r not in ROLE_LABELS]
            if bad:
                raise SheetRefused("%s: roles outside %s for %s" % (s, sorted(ROLE_LABELS), bad))
            rows.append(dict(base, source=s, decision=d["decision"], setting=d["setting"], roles=d.get("roles") or {},
                             family=d.get("family") or None, series=bool(d.get("series")), note=d.get("note") or ""))
    else:
        want = set(rend["items"])
        got = doc.get("images") or {}
        if set(got) != want:
            raise SheetRefused("the file marks %d images, the sheet shows %d" % (len(got), len(want)))
        for i, d in sorted(got.items()):
            if kind == "ln":
                nb = len((rend.get("item_boxes") or {}).get(i) or [])
                marks = list(d.get("marks") or [])
                if len(marks) != nb or any(m not in LN_MARKS for m in marks) or int(d.get("missing") or 0) < 0:
                    raise SheetRefused("LN image %s: %d marks for %d boxes, or a mark outside %s" % (
                        i, len(marks), nb, sorted(LN_MARKS)))
                if d.get("checked") is not True:
                    raise SheetRefused("LN image %s is not marked checked: every image of the sheet is looked at "
                                       "(a sheet saved untouched is not a measurement)" % i)
                rows.append(dict(base, item=i, group=(rend.get("item_groups") or {}).get(i), marks=marks,
                                 missing=int(d.get("missing") or 0)))
            else:
                if d.get("verdict") not in MTW_VERDICTS or bool(d.get("error")) != (d.get("verdict") == "error"):
                    raise SheetRefused("MT-W image %s: its answer is %r, not one of %s (every image is answered)" % (
                        i, d.get("verdict"), list(MTW_VERDICTS)))
                shown = list((rend.get("item_boxes") or {}).get(i) or [])
                removed = sorted({int(j) for j in d.get("removed") or []})
                if any(not 0 <= j < len(shown) for j in removed):
                    raise SheetRefused("MT-W image %s: a removed box index outside its %d boxes" % (i, len(shown)))
                added = [[float(x) for x in b] for b in d.get("added") or []]
                if any(len(b) != 4 or not all(0.0 <= x <= 1.0 for x in b) for b in added):
                    raise SheetRefused("MT-W image %s: an added box is not (cx, cy, w, h) in [0, 1]" % i)
                row = dict(base, item=i, source=(rend.get("item_groups") or {}).get(i), verdict=d["verdict"],
                           error=d["verdict"] == "error", removed=removed, added=added, note=d.get("note") or "",
                           total=(rend.get("item_totals") or {}).get(i))
                if removed or added:
                    # the image's whole corrected label: the boxes shown without the ones removed, with the ones drawn
                    row["corrected_boxes"] = [[float(x) for x in b[-4:]] for j, b in enumerate(shown)
                                              if j not in removed] + added
                rows.append(row)
    p = decisions_path(inc)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
    return rows


# ------------------------------------------------------------------ the folds the builds and the gate read
def fold(inc=None):
    """{source: {decision, roles, setting, family, series, delegate_decision, owner_decision, decided_by,
    audit_required, audited, reversed}}: the last decision of each source by the delegate (or a person on a source
    sheet), the owner's audit decision over it (a reversal is recorded as his and governs), and whether the audit
    applies (DS.0 item 4: the seeded 5 of each phase, all once he reversed one, every names-round source, SIU's
    table)."""
    rows = _jsonl(decisions_path(inc))
    out = {}
    for r in rows:
        if r.get("kind") == "source":
            out.setdefault(r["source"], {})["delegate_decision"] = {k: r.get(k) for k in (
                "decision", "setting", "roles", "family", "series", "decided_by", "sheet_sha256")}
            out[r["source"]]["phase"] = _phase_of(r.get("sheet_sha256"), inc)
        elif r.get("kind") == "audit":
            out.setdefault(r["source"], {})["owner_decision"] = {k: r.get(k) for k in (
                "decision", "setting", "roles", "family", "series", "decided_by", "sheet_sha256")}
    names_round = _names_round_sources(inc)
    by_phase = collections.defaultdict(list)
    for s, v in out.items():
        if v.get("delegate_decision"):
            by_phase[v.get("phase") or 1].append(s)
    reversed_any = any(_reversed(v) for v in out.values())
    sample = set()
    for ph, ss in by_phase.items():
        revs = _phase_revisions(ph, inc)
        sample |= sticky_audit_sample(revs) if revs else set(audit_sample(ss))
    for s, v in out.items():
        own, dele = v.get("owner_decision"), v.get("delegate_decision")
        gov = own or dele or {}
        v.update({k: gov.get(k) for k in ("decision", "setting", "roles", "family", "series")})
        v["decided_by"] = gov.get("decided_by")
        v["reversed"] = _reversed(v)
        v["audit_required"] = bool(reversed_any or s in sample or s in names_round or s == _siu())
        v["audited"] = own is not None
        v["audit_reason"] = [x for x, c in (("reversal widened the audit", reversed_any), ("seeded 5", s in sample),
                                            ("names round", s in names_round), ("SIU's table", s == _siu())) if c]
    return out


def _siu():
    from . import ds1_data as DD
    return DD.SIU


def _reversed(v):
    own, dele = v.get("owner_decision"), v.get("delegate_decision")
    if not own or not dele:
        return False
    return any(own.get(k) != dele.get(k) for k in ("decision", "roles", "setting"))


def _phase_of(sheet_sha, inc=None):
    r = renders(inc).get(sheet_sha) or {}
    for b in r.get("bundles") or []:
        m = re.match(r"source_p(\d)_", b.get("name") or "")
        if m:
            return int(m.group(1))
    return 1


def _phase_revisions(phase, inc=None):
    """[[sources of revision 1], ...] of a phase's screen revisions (splits/ds1/screen/p<phase>_r<nnn>.json under inc):
    each revision's live sources, role C and wholly quarantined ones left out (they enter no manifest)."""
    d = (Path(inc) if inc else C.INC_DIR) / "splits" / "ds1" / "screen"
    out = []
    for p in sorted(d.glob("p%d_r[0-9][0-9][0-9].json" % int(phase))) if d.is_dir() else []:
        doc = _read_json(p) or {}
        out.append(sorted(s for s, v in (doc.get("sources") or {}).items() if v.get("live") and not (
            v.get("role_c") or int((v.get("capture_names") or {}).get("hits") or 0) or v.get("quarantine_whole"))))
    return out


def _names_round_sources(inc=None):
    """Sources a names round (L26) resolved class names of (the collector's names layer records them)."""
    p = (Path(inc) if inc else C.INC_DIR) / "intake" / "names"
    out = set()
    for f in sorted(p.glob("names_*.json")) if p.is_dir() else []:
        d = _read_json(f)
        if isinstance(d, dict) and d.get("source"):
            out.add(str(d["source"]))
    return out


def decided_sources(inc=None):
    """What ds1_data reads: {source: fold record} of every decided source."""
    return {s: v for s, v in fold(inc).items() if v.get("decision")}


def mtw_checks(inc=None):
    """{source: {verdict, errors, sample, total, corrected_labels, sha256}} of the MT-W check records (DS.10 item
    4), the rule of inc2.ds1.mtw_source_verdict."""
    rows = [r for r in _jsonl(decisions_path(inc)) if r.get("kind") == "mtw"]
    by = collections.defaultdict(list)
    for r in rows:
        by[r.get("source")].append(r)
    out = {}
    for s, rs in by.items():
        errors = sum(1 for r in rs if r.get("error"))
        corrected = {r["item"].split("|")[-1]: r for r in rs if r.get("corrected_boxes") is not None}
        uncorrected = [r["item"] for r in rs if r.get("error") and r.get("corrected_boxes") is None]
        total = int(max([r.get("total") or 0 for r in rs] + [len(rs)]))
        v = D.mtw_source_verdict(len(rs), total, errors, corrected=bool(corrected) and not uncorrected,
                                 whole=None)
        v.update(source=s, sha256=hashlib.sha256(canon(rs)).hexdigest(), corrections=len(corrected),
                 corrected_labels={k: r["corrected_boxes"] for k, r in corrected.items()})
        out[s] = v
    return out


def ln_marks(inc=None):
    """{group: [{"marks", "missing"}]} of the recorded LN sheet (DS.8.2)."""
    out = collections.defaultdict(list)
    for r in _jsonl(decisions_path(inc)):
        if r.get("kind") == "ln":
            out[r.get("group")].append({"marks": r.get("marks") or [], "missing": int(r.get("missing") or 0)})
    return dict(out)


def audit_sources(decisions):
    """The sources the owner's audit batch shows now: every decided source whose audit applies and is not done."""
    return sorted(s for s, v in decisions.items() if v.get("audit_required") and not v.get("audited")
                  and v.get("delegate_decision"))


def status(inc=None):
    """What waits for whom (read-only): the sheets decided, the audits pending, the MT-W checks and LN."""
    f = fold(inc)
    return {"decided": sorted(s for s, v in f.items() if v.get("decision")),
            "audit_pending": audit_sources(f), "mtw_checks": {s: v["verdict"] for s, v in mtw_checks(inc).items()},
            "ln_recorded": sum(len(v) for v in ln_marks(inc).values()),
            "rendered": sorted("%s_%03d" % (r["kind"], r["batch"]) for r in renders(inc).values()),
            "superseded": sorted(superseded(inc)), "audit_sets": audit_sets(inc)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="DS1's sheets (L31) and their decision records.")
    ap.add_argument("command", choices=("render", "record", "status", "supersede"))
    ap.add_argument("--kind", choices=KINDS, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lab-inc", default=None)
    ap.add_argument("--no-pull", action="store_true")
    ap.add_argument("--file", default=None)
    ap.add_argument("--decided-by", default=None)
    ap.add_argument("--sources", default=None, help="render --kind audit: the sources of the batch (comma list)")
    ap.add_argument("--sheet", default=None, help="supersede: the rendered sheet (kind_batch, e.g. audit_002)")
    ap.add_argument("--by", default=None, help="supersede: the sheet of the same batch that stands (e.g. audit_001)")
    ap.add_argument("--why", default=None, help="supersede: the reason, recorded")
    a = ap.parse_args(argv)
    try:
        if a.command == "render":
            if a.kind is None or a.batch is None:
                raise SheetRefused("render needs --kind and --batch")
            srcs = [x for x in a.sources.split(",") if x] if a.sources else None
            print(json.dumps(render(a.kind, a.batch, inc=a.lab_inc, pull=not a.no_pull, sources=srcs), indent=1))
        elif a.command == "record":
            rows = record(a.file, a.decided_by, inc=a.lab_inc)
            print(json.dumps({"recorded": len(rows)}))
        elif a.command == "supersede":
            print(json.dumps(supersede(a.sheet, a.by, a.why, inc=a.lab_inc), indent=1, sort_keys=True))
        else:
            print(json.dumps(status(a.lab_inc), indent=1, sort_keys=True))
    except (D.Ds1Refused, ValueError, OSError) as e:
        print("[inc2.ds1_sheets] ERROR: %s" % e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
