"""V3F: the family records a recorded deviation supersedes, moved aside byte for byte (deviation 5, 2026-10-24;
docs/prereg/V3F_dev5_2026-10-24.md; deviation 6, 2026-10-25, docs/prereg/V3F_dev6_2026-10-25.md).

    python -m weed_optimizer_framework.tools.inc2.v3fam_supersede status     (read-only)
    python -m weed_optimizer_framework.tools.inc2.v3fam_supersede apply      (the score jobs of v3fam_systems' and
                                                                            v3fam_trained's arms (group scores), D6's
                                                                            smoke (group d6) and the verdict's job
                                                                            (both) run it first)

ENTRIES names each record by its path under INC_DIR and its sha256. apply() moves a record whose file hashes so
into the entry's dated folder beside it (a hard link made there, then the original unlinked: never overwritten,
never deleted, the same bytes), and records the move in that folder's MANIFEST.json: the record's path, sha256
and size, when it was moved, by which job and host, why, and the deviation's file with the sha256 the deployed
tree holds for it. A file at the path with any other sha256 is not the record the deviation names: it is left in
place and reported. A record already moved (absent at its path, present in its folder with its sha256) is
reported moved. A move cut off after its link (the same bytes at both places) is completed by the next run.
Running it again changes nothing.

Why each record goes (deviation 5): deviation 4 (2026-10-20, deployed 2026-10-21) changed inc2/v3fam.py in its CLI
only, from sha256 3d776dae... to c5e923dc...; every reproduction record of 2026-10-12 pins 3d776dae... (code_shas),
so inc2.v3fam.require_equivalence refused all twelve score jobs of D1, D4F, D4N, D4R15, D5B and D8 ('the code
changed since dev_identity.json ran'). Deviation 5 puts inc2/v3fam.py back to 3d776dae... byte for byte. Then:
  - equivalence/identity_probe.json, the one record taken under c5e923dc... (D5B's job 49570114,
    2026-10-21T23:45:45Z), would refuse D5B's score in turn: it goes, and D5B's job takes it again
    (v3fam_systems.require_or_make_identity) under the restored code;
  - failed/score_<arm>.json for the six arms, written 2026-10-22T00:43:38Z by the continuation's job 49571261 from
    the ticker's --ticker-failed: those score jobs were refused by the platform's own stale records, not failed by
    the arms; they go, so no verdict reads them as the arms' failures (v3fam_verdict also refuses to count them by
    sha256 while they are still in place);
  - d6/smoke/smoke.json, D6's smoke of 2026-10-21 (job 49542653): it recorded the whole one-epoch fit (2,405.6 s,
    of which about 905 s were the one-time dataset construction and sanity check before the epoch) as the epoch
    time and projected 33.9 h against the 28.8 h rule; the registered rule takes the epoch's own time (V3F.5 D6:
    epoch time x 50 + 0.5 h), so the smoke is taken again with the epoch timed from its start (v3fam_detr); it goes
    first in that smoke's own job (group "d6"), so no reader takes it as D6's walltime outcome.

Why the record of deviation 6 goes: equivalence/identity_ood23.json was taken 2026-10-25T15:31:06Z by D1's score job
49820090 with Ultralytics' label cache recording the stored (landscape) size of the 289 ood23 photos PIL reads as MPO
tagged 6 or 8 (inc2.v3fam_loader); its family identity pass and its plain locked score were both taken so (it passed
with difference 0), and identity mode never runs E3's geometry check, so it does not cover the path under the loader
fix. It goes (group "scores"), and D1's job takes it again under the fix (inc2.v3fam_loader.identity).

Each entry belongs to a group: "scores" (the score jobs move these: GROUP_SCORES) and "d6" (D6's smoke job moves its
own record: GROUP_D6). The score jobs (MAINT) and D6's smoke (TRAIN) may run at the same time, so each moves only its
own group; the verdict's job, which runs after every arm, moves both.

Stdlib only (the snapshot, inc_autopilot.stream_remote, reads status() on the login node).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import socket
import sys
from pathlib import Path

FORMAT = "inc2-v3fam-superseded/1"
MANIFEST = "MANIFEST.json"
DEVIATION_5 = "docs/prereg/V3F_dev5_2026-10-24.md"
FOLDER_5 = "superseded_2026-10-24"
_WHY_PROBE = ("taken 2026-10-21T23:45:45Z by D5B's score job 49570114 under inc2/v3fam.py sha256 c5e923dc... "
              "(deviation 4's CLI change); deviation 5 restores inc2/v3fam.py to 3d776dae..., under which this record "
              "would refuse D5B's score; D5B's score job takes it again (require_or_make_identity)")
_WHY_FAILED = ("written 2026-10-22T00:43:38Z by the continuation's job 49571261 from the ticker's --ticker-failed: "
               "both score jobs of the arm were refused by inc2.v3fam.require_equivalence ('the code changed since "
               "dev_identity.json ran'), the platform's own stale reproduction records, not the arm's failure; the "
               "arm is scored again under deviation 5")
_WHY_SMOKE = ("written 2026-10-21T12:38:15Z by D6's smoke job 49542653: it recorded the whole one-epoch fit "
              "(2,405.6 s, including about 905 s of one-time dataset construction and sanity check before the epoch) "
              "as the epoch time and projected 33.9 h against the 28.8 h rule; V3F.5 D6's rule takes the epoch's own "
              "time, so the smoke is taken again with the epoch timed from its start (deviation 5)")
DEVIATION_6 = "docs/prereg/V3F_dev6_2026-10-25.md"
FOLDER_6 = "superseded_2026-10-25"
_WHY_OOD23 = ("taken 2026-10-25T15:31:06Z by D1's score job 49820090 while Ultralytics 8.4.37's label cache recorded "
              "the stored (landscape) size of the 289 ood23 photos PIL reads as MPO tagged 6 or 8 (exif_size swaps "
              "width and height for JPEG only), so both of its scores letterboxed those photos at 0.8 or 0.9 of the "
              "scale ratio_pad records; identity mode never runs E3's geometry check, which refused D1's first ood23 "
              "pass (WY_2, 0.109094); D1's job takes it again under deviation 6's loader fix (inc2.v3fam_loader)")
GROUP_SCORES = "scores"
GROUP_D6 = "d6"
GROUPS = (GROUP_SCORES, GROUP_D6)
# (path under INC_DIR, sha256 as read from the cluster on 2026-10-24, deviation file, folder beside it, why)
ENTRIES = (
    ("v3fam/v1/equivalence/identity_probe.json", "c99a0df0ab003c179d662ac394c93f2042a220fa79b812bb28295e281cc4b7d4",
     DEVIATION_5, FOLDER_5, _WHY_PROBE),
    ("v3fam/v1/failed/score_D1.json", "aaa6bbe1b3f27bb764c468c005ed34d48b60470108c20ab97dc59e8eacfa7f89",
     DEVIATION_5, FOLDER_5, _WHY_FAILED),
    ("v3fam/v1/failed/score_D4F.json", "e402b5f1691c0b37d149202e9dce1690f63603b0a7ae18c9258b9d8e74bd0149",
     DEVIATION_5, FOLDER_5, _WHY_FAILED),
    ("v3fam/v1/failed/score_D4N.json", "9581a06e81328d3006d58d5132737798e5c1cf06315d94bd10ba0ef322788083",
     DEVIATION_5, FOLDER_5, _WHY_FAILED),
    ("v3fam/v1/failed/score_D4R15.json", "84b50c9f92e68825009a27a89c115ffa8c5eff552d8e306e8ede506818ad0bf4",
     DEVIATION_5, FOLDER_5, _WHY_FAILED),
    ("v3fam/v1/failed/score_D5B.json", "f21f70895766dbf9c4962a2b818479763b5f9cf4356ad993b355825bcc6258e4",
     DEVIATION_5, FOLDER_5, _WHY_FAILED),
    ("v3fam/v1/failed/score_D8.json", "24994cc47c0b7786c903b665199e2447d7151ba05d138146f6e0f29562052ebf",
     DEVIATION_5, FOLDER_5, _WHY_FAILED),
    ("v3fam/v1/d6/smoke/smoke.json", "1d0691a852f1c4e3345e678e05cdade007b9d5c7de31b8fd52707f4ab597c2b8",
     DEVIATION_5, FOLDER_5, _WHY_SMOKE),
    # deviation 6 (2026-10-25; sha256 as read from the cluster on 2026-10-25)
    ("v3fam/v1/equivalence/identity_ood23.json", "3c29764783b8303b0659cfe75ee49b966dff4334394477fba24b2245045ad3ea",
     DEVIATION_6, FOLDER_6, _WHY_OOD23),
)
D6_SMOKE = "v3fam/v1/d6/smoke/smoke.json"


def group_of(rel):
    """The group an entry's record belongs to: D6's smoke record is "d6", every other "scores"."""
    return GROUP_D6 if rel == D6_SMOKE else GROUP_SCORES


class SupersedeRefused(RuntimeError):
    """A move that would overwrite or lose a file."""


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _sha_or_none(path):
    try:
        return _sha(path) if Path(path).is_file() else None
    except OSError:
        return None


def dest_of(inc, rel, folder):
    """Where an entry's record goes: its folder beside it, the same file name."""
    p = Path(inc) / rel
    return p.parent / folder / p.name


def superseded_sha256s(rel_prefix=None):
    """{path under INC_DIR: sha256} of every entry (those under rel_prefix when given)."""
    return {rel: sha for rel, sha, _d, _f, _w in ENTRIES if rel_prefix is None or rel.startswith(rel_prefix)}


def is_superseded(rel, sha):
    """True when the record at rel (under INC_DIR) with this sha256 is one a recorded deviation supersedes."""
    return bool(sha) and any(r == rel and s == sha for r, s, _d, _f, _w in ENTRIES)


def status(inc):
    """{entries: [{path, sha256, deviation, state, dest}], pending, moved, other}: each entry's state (pending: its
    record in place, to be moved; moved: in its folder with its sha256 and absent at its path; absent: in neither
    place; other: a file at the path or in the folder with another sha256, never touched). Read-only."""
    out = {"format": FORMAT, "entries": [], "pending": [], "moved": [], "other": [], "absent": []}
    for rel, sha, dev, folder, _why in ENTRIES:
        src, dst = Path(inc) / rel, dest_of(inc, rel, folder)
        s_src, s_dst = _sha_or_none(src), _sha_or_none(dst)
        if s_src == sha and s_dst in (None, sha):
            state = "pending"                    # (both: a move cut off after its link, completed by apply)
        elif s_src is None and s_dst == sha:
            state = "moved"
        elif s_src is None and s_dst is None:
            state = "absent"
        else:
            state = "other"
        out["entries"].append({"path": rel, "sha256": sha, "deviation": dev, "state": state,
                               "dest": str(dst.relative_to(Path(inc)))})
        out[state].append(rel)
    return out


def _write_manifest(folder, entry):
    """Add one moved record to the folder's MANIFEST.json (rewritten whole, atomically)."""
    mp = Path(folder) / MANIFEST
    try:
        doc = json.loads(mp.read_text()) if mp.is_file() else None
    except (OSError, ValueError):
        doc = None
    if not isinstance(doc, dict) or doc.get("format") != FORMAT:
        if mp.exists():
            raise SupersedeRefused("%s exists and is not a %s manifest: nothing is moved" % (mp, FORMAT))
        doc = {"format": FORMAT, "folder": Path(folder).name, "entries": {}}
    doc["entries"][entry["name"]] = entry
    tmp = mp.with_name(".%s.%d.tmp" % (MANIFEST, os.getpid()))
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    os.replace(str(tmp), str(mp))


def _link_or_copy(src, dst):
    """dst made from src without overwriting anything: a hard link (the same inode), or, where the file system
    refuses links, a copy written under a temporary name beside dst (exclusive create), checked against src's
    sha256 and only then renamed to dst, which must still be absent. A copy cut off (a full disk, an error) is
    removed (its own temporary file, never a record), so it never stands at dst; one a kill leaves keeps its
    temporary name and blocks nothing."""
    try:
        os.link(str(src), str(dst))                  # fails if dst appeared meanwhile: never overwritten
        return
    except FileExistsError:
        raise SupersedeRefused("%s exists: the superseded record %s is never moved over another file" % (dst, src))
    except OSError:
        pass
    tmp = Path(dst).with_name(".%s.%d.part" % (Path(dst).name, os.getpid()))
    try:
        with open(str(src), "rb") as fi, open(str(tmp), "xb") as fo:
            for b in iter(lambda: fi.read(1 << 20), b""):
                fo.write(b)
            fo.flush()
            os.fsync(fo.fileno())
        if _sha(tmp) != _sha(src):
            raise SupersedeRefused("the copy of %s does not hash as it: nothing is moved" % src)
        if os.path.lexists(str(dst)):
            raise SupersedeRefused("%s exists: the superseded record %s is never moved over another file" % (dst, src))
        os.rename(str(tmp), str(dst))
    finally:
        if os.path.lexists(str(tmp)):
            os.unlink(str(tmp))                      # this call's own partial copy, never a record


def apply(inc, repo=None, job=None, log=None, groups=GROUPS):
    """Move every pending entry's record of the given groups into its folder (status()), each recorded in the
    folder's MANIFEST.json. Returns {moved: [paths], status}. Refuses (SupersedeRefused) before touching a record
    whose destination already holds a file. repo: the deployed tree whose deviation files are hashed into the
    manifest. groups: GROUP_SCORES for the score jobs, GROUP_D6 for D6's smoke, both for the verdict's job."""
    inc = Path(inc)
    repo = Path(repo) if repo else None
    job = job if job is not None else (os.environ.get("SLURM_JOB_ID") or None)
    groups = tuple(groups or ())
    bad = [g for g in groups if g not in GROUPS]
    if bad:
        raise SupersedeRefused("unknown group(s) %s (not one of %s)" % (bad, list(GROUPS)))
    moved = []
    for rel, sha, dev, folder, why in ENTRIES:
        if group_of(rel) not in groups:
            continue
        src, dst = inc / rel, dest_of(inc, rel, folder)
        if _sha_or_none(src) != sha:
            continue
        if dst.is_symlink() or (dst.exists() and _sha_or_none(dst) != sha):
            raise SupersedeRefused("%s exists: the superseded record %s is never moved over another file" % (dst, src))
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            _link_or_copy(src, dst)
            if _sha(dst) != sha:
                os.unlink(str(dst))
                raise SupersedeRefused("%s does not hash as %s after the link: nothing is moved" % (dst, src))
        # else: a move cut off after its link (the same bytes at both places): completed here
        devp = (repo / dev) if repo else None
        _write_manifest(dst.parent, {"name": src.name, "from": rel, "to": str(dst.relative_to(inc)), "sha256": sha,
                                     "bytes": dst.stat().st_size, "moved_utc": _utc(), "job": job,
                                     "host": socket.gethostname(), "why": why, "deviation": dev,
                                     "deviation_sha256": _sha_or_none(devp) if devp else None})
        os.unlink(str(src))
        moved.append(rel)
        if log:
            log("superseded (%s): %s -> %s (sha256 %s)" % (dev, rel, dst.relative_to(inc), sha[:12]))
    return {"moved": moved, "status": status(inc)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="V3F: the records a recorded deviation supersedes, moved aside byte for "
                                             "byte into a dated folder with a manifest (deviations 5 and 6).")
    ap.add_argument("command", choices=("status", "apply"))
    ap.add_argument("--group", choices=GROUPS, action="append", default=None,
                    help="apply only this group's records (repeatable; default: every group)")
    a = ap.parse_args(argv)
    from ..inc import common as C
    try:
        if a.command == "status":
            print(json.dumps(status(C.INC_DIR), indent=1, sort_keys=True))
        else:
            out = apply(C.INC_DIR, repo=C.REPO, log=lambda m: print("[inc2.v3fam_supersede] %s" % m, flush=True),
                        groups=tuple(a.group) if a.group else GROUPS)
            print(json.dumps(out, indent=1, sort_keys=True))
    except (SupersedeRefused, OSError) as e:
        print("[inc2.v3fam_supersede] ERROR: %s" % e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
