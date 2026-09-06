#!/usr/bin/env python3
"""Live evidence bundle for the supervision layer (v3.27.0).

Why this exists
---------------
`corpus.py` freezes an incident that already happened. This module builds the
same object for a step that is happening now, so the supervisor reads one kind
of thing whether it is reviewing a live round or replaying 2026-08-29. The
bundle shape, the section names and the token estimate come from `corpus.py`
and are not redefined here; what is new is where the bytes come from.

The worked example this module is built against: on 2026-08-29 jobs 44727703
and 44767709 (both `--time=12:00:00`) hit the walltime at epoch 24 and 16 of 60
because the merged pool had grown to 8,583 iterations per epoch. `sacct` shows
State TIMEOUT with Elapsed 12:00:18 / 12:00:20 against a 12:00:00 Timelimit;
the 22 MB `.out` files still exist; and the same harvest log carries
`[net] WARN: SOCKS proxy ... SKIPPING github, using Kaggle/HF only`, which has
been present in every harvest log since June and is a designed control whose
correct verdict is "no incident". Every one of those facts has to survive into
the bundle, including the last one: a trimmer that drops the SKIPPING line
makes the no-incident case unscoreable.

Four constraints come from the machine, not from taste
------------------------------------------------------
* **One batched remote command per build.** The always-on server reaches the
  cluster over a throttled ssh channel where rapid sessions trigger a login
  throttle, so gathering thirteen sections cannot be thirteen commands.
  `remote_script()` is a single here-doc-free shell string that prints
  section-delimited output; `parse_output()` reads it back. Staging payloads
  ride in the same command when they fit under the per-command size limit.
* **Absolute line numbers.** Every excerpt is emitted as `%06d\\t<line>` with
  awk NR numbering of the ORIGINAL file (the same convention `corpus.py`
  stores), because a citation that resolves to a tail-relative index resolves
  to the wrong line as soon as the file grows. The tail is numbered from the
  file's own line count, not from 1.
* **Nothing is silently lost.** Per-section caps are applied with a priority
  order: `sacct`, the job-scoped strategy JSON and the computed signals are
  never trimmed; the `.out` tail is trimmed first and always keeps every WARN,
  ERROR, FAIL, TIMEOUT, CANCELLED and SKIPPING line plus the last N lines. Each
  trim writes what it removed and by how much into `export.trim`. A section
  whose command failed is `null` with the rc and its stderr; a section with
  nothing behind it is `null` with a reason; neither disappears.
* **The token estimate is checked before the call, not after.** `build()`
  refuses to hand back a bundle whose estimate exceeds the caller's `num_ctx`:
  it trims and says so in `export.build`. Silent truncation at the model
  boundary is the exact defect class this work exists to catch, so this module
  must not commit it itself.

Also: a compute node cannot reach the lab database, so the ledger tail, the
active corrections and the active plan are staged to /ocean as files by the
lab (`stage_commands`), gzip+base64 above the inline threshold and split when
still too large. The mirror's own sha256 travels with it, which is what the
ownership check compares against.

Nothing here raises out of a public entry point. `build()` always returns a
bundle: on total failure it is a bundle whose sections are all null and whose
`export.warnings` names what went wrong, which is a reviewable object, unlike
an exception in a scheduler thread. Pure stdlib, no network at import, no
subprocess anywhere in this module: the caller owns the channel and passes it
in as `ctx`.

Signals are computed by `signals.py` when it is importable, or by
`ctx["signals_fn"]`. When neither is available the section is `null` with the
reason recorded, never an empty list: "the detectors found nothing" and "there
were no detectors" must not look the same.

CLI (no cluster access of its own):
    python3 -m weed_optimizer_framework.tools.brain.evidence script \\
        --domain weed --round 4 --step train --jobid 44727703
    python3 -m weed_optimizer_framework.tools.brain.evidence build \\
        --domain weed --round 4 --step train --jobid 44727703 \\
        --from-file captured.txt [--out bundle.json] [--num-ctx 32768]
"""
import argparse
import base64
import gzip
import hashlib
import json
import os
import re
import sys
import time

try:                                     # package import (the normal path)
    from . import corpus
except ImportError:                      # direct execution from this directory
    import corpus                        # type: ignore

TOOL_VERSION = "wp3-evidence/1"

# The bundle shape is corpus.py's, not a second copy of it.
SECTIONS = corpus.SECTIONS

# Sections a trimmer may never touch. sacct is the ground truth about the job's
# fate, the strategy JSON is what the job was actually asked to do, and the
# signals are the deterministic facts the whole escalation policy rests on;
# trimming any of the three changes the verdict rather than the token count.
NEVER_TRIM = ("sacct", "strategy", "signals")

# The project directory whose free space the `disk_low` signal reads. It is the
# lab's shared quota, so exhausting it breaks every job of the group.
# VERIFY: `df -P -k /ocean/projects/cis240145p` on the cluster prints the
# filesystem this resolves to; the per-project quota is a different number and
# is reported separately (see _resources_value).
CLUSTER_PROJECT = os.environ.get("CLUSTER_PROJECT_DIR",
                                 "/ocean/projects/cis240145p")

# --- limits ---------------------------------------------------------------
# Every threshold this module has, in one place, each with the reason it holds
# that value. Overridable per call (`ctx["limits"]`) and per process
# (`EVIDENCE_<KEY>` in the environment). Signal thresholds are NOT here: they
# belong to signals.py / thresholds.json, and duplicating them would give the
# campaign two pre-registered values for the same rule.
DEFAULTS = {
    # Pre-registered bundle budget. Asserted against the model's num_ctx before
    # every call; 12,000 leaves room for the prompt and the schema inside a
    # 16,384-token window, which is the smallest window in the fast arm.
    "token_budget": 12000,
    # Reserved for prompt, schema and instructions when budgeting against
    # num_ctx. Measured against the deployed supervisor prompt, rounded up.
    "prompt_reserve_tokens": 1024,
    # corpus.CAPS parity: the same numbers the frozen cases were built with, so
    # a live bundle and an archived one show a model the same amount.
    "out_tail_lines": corpus.CAPS["out_tail_lines"],
    "out_tail_tail_lines": corpus.CAPS["out_tail_tail_lines"],
    "sacct_rows": corpus.CAPS["sacct_rows"],
    "trace_records": corpus.CAPS["trace_records"],
    # Ring size for matched lines on the remote side. out_tail_lines minus the
    # always-kept tail, so a full transfer cannot exceed the local cap.
    "out_tail_match_lines": max(
        1, corpus.CAPS["out_tail_lines"] - corpus.CAPS["out_tail_tail_lines"]),
    # A 22 MB .out reduces to these; the remote side never ships the whole file.
    "trace_lines": 200,          # JSONL lines pulled before the record cap
    "harvest_lines": 120,        # matched [net]/[src] lines from the harvest log
    "results_csv_lines": 400,    # header + last N epochs, contiguous
    "ledger_rounds": 5,          # "last 5 rounds" per the bundle spec
    # A JSON artifact larger than this is reported by identity only: reading it
    # costs the tick, and a truncated JSON is not parseable anyway.
    "json_max_bytes": 262144,
    # Hashing is one extra read of the file. Above this the read is the cost of
    # the tick, so identity is reported without a hash and the bundle says so.
    "sha256_max_bytes": 536870912,
    # Relay limit: one batched command carries at most 96 KB of payload
    # (inline base64, no scp on this channel).
    "stage_max_bytes": 98304,
    # Above this a staged payload is gzipped before base64. Below it the plain
    # form is easier to read on the cluster when something goes wrong.
    "gzip_above_bytes": 32768,
    # The gather reads a 22 MB log twice (hash + selection) on a shared
    # filesystem; 180 s is the scheduler tick's whole ssh budget.
    "remote_timeout_s": 180,
    # SU accounting rate per GPU-hour by GPU family (H100 = 2, V100 = 1).
    "su_h100_per_gpu_hour": 2.0,
    "su_v100_per_gpu_hour": 1.0,
    "su_default_per_gpu_hour": 1.0,
}

# Lines the out_tail trimmer keeps wherever they appear. corpus.py's regex plus
# SKIPPING: the chronic `SKIPPING github` line is the designed control in the
# 2026-08-29 bundle, and a bundle that trims it away cannot be scored on the
# verdict it exists to test.
OUT_TAIL_KEEP_RE = re.compile(
    r"(WARN|ERROR|FAIL|TIMEOUT|CANCELLED|Traceback|slurmstepd|SKIPPING|"
    r"out of memory|CUDA|Killed|exceeded)", re.I)

_LINE_RE = re.compile(r"^(\d{6})\t(.*)$", re.S)
_ID_RE = re.compile(r"[^0-9]")
_DOM_RE = re.compile(r"[^a-z0-9_]")
_STEP_RE = re.compile(r"[^a-z0-9_]")


def _limits(ctx=None):
    """Resolved limits: defaults, then environment, then the caller's ctx."""
    out = dict(DEFAULTS)
    for key, default in DEFAULTS.items():
        raw = os.environ.get("EVIDENCE_" + key.upper())
        if raw is None:
            continue
        try:
            out[key] = type(default)(raw)
        except (TypeError, ValueError):
            pass                     # an unreadable override keeps the default
    given = (ctx or {}).get("limits") if isinstance(ctx, dict) else None
    if isinstance(given, dict):
        for key, val in given.items():
            if key in out and isinstance(val, (int, float)):
                out[key] = val
    return out


# --- input hygiene ---------------------------------------------------------
def _job(job_id):
    """Digits only. This string is interpolated into a shell command that runs
    on a shared account, so it is filtered rather than quoted."""
    return _ID_RE.sub("", str(job_id or ""))[:24]


def _domain(domain):
    return _DOM_RE.sub("", str(domain or "").strip().lower())[:40] or "unknown"


def _step(step):
    return _STEP_RE.sub("", str(step or "").strip().lower())[:32] or "unknown"


def _round(round_num):
    try:
        return int(round_num)
    except (TypeError, ValueError):
        return None


def _num(x):
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f


def nonce(domain, round_num, step, job_id):
    """Delimiter nonce for one build: sha256 over the build's own identity.

    Deterministic on purpose — a test replays a recorded gather, and a random
    nonce would make the recording unusable. It is not a security boundary: the
    parser's defence against a log line that contains the delimiter is the byte
    count the shell declares on the END marker, not the unguessability of this
    string.
    """
    ident = "|".join([TOOL_VERSION, _domain(domain), str(_round(round_num)),
                      _step(step), _job(job_id)])
    return hashlib.sha256(ident.encode("utf-8")).hexdigest()[:16].upper()


# --- the one batched remote command ---------------------------------------
# Sections in SECTIONS order minus `signals`, which is computed here from the
# other twelve. Here-doc free: this string is quoted once by the caller's ssh
# path, and a here-doc inside it does not survive that quoting reliably.
_SCRIPT = r"""set +e
umask 077
EV=__EV__
J=__JOBID__
D=__DOMAIN__
FW=results/framework
BR=$FW/_brain/$D
T=${TMPDIR:-/tmp}/ev_$$
mkdir -p "$T" 2>/dev/null || T=.
O=$T/o; E=$T/e; N=$T/n; RAWF=$T/r
S() {
  awk 1 "$O" > "$N" 2>/dev/null
  if [ "$2" != "0" ]; then awk 'NR<=5 {print "[stderr] " $0}' "$E" >> "$N" 2>/dev/null; fi
  B=$(wc -c < "$N" 2>/dev/null | tr -d ' ')
  case "${B:-x}" in ''|*[!0-9]*) B=0 ;; esac
  echo "$EV BEGIN $1"
  cat "$N" 2>/dev/null
  echo "$EV END $1 rc=$2 bytes=$B"
}
H() {
  hb=$(wc -c < "$1" 2>/dev/null | tr -d ' ')
  case "${hb:-x}" in ''|*[!0-9]*) hb=0 ;; esac
  hm=$(stat -c %Y "$1" 2>/dev/null)
  [ -z "$hm" ] && hm=$(stat -f %m "$1" 2>/dev/null)
  case "${hm:-x}" in ''|*[!0-9]*) hm=0 ;; esac
  hs=""
  if [ "$hb" -le __SHAMAX__ ]; then hs=$(sha256sum "$1" 2>/dev/null | cut -d' ' -f1); fi
  echo "[file] path=$1 bytes=$hb mtime=$hm sha256=$hs"
}
NUMBER() { awk -v M=__JSONMAX__ '{ if (b<=M) { printf "%06d\t%s\n", NR, $0; b+=length($0)+1 } else { s++ } } END { printf "[lines] %d\n", NR; if (s) printf "[truncated] %d line(s) past %d bytes were not read\n", s, M }' "$1"; }
TAILNUM() { awk -v L="$2" '{ rb[NR%L]=$0; n=NR } END { s=n-L+1; if (s<1) s=1; for (i=s;i<=n;i++) printf "%06d\t%s\n", i, rb[i%L]; printf "[lines] %d\n", n }' "$1"; }
C() { if [ -f "$1" ]; then H "$1"; NUMBER "$1"; else echo "[none] $2"; fi; }
P() {
  for i in $IDS; do
    for p in $(ls -td $FW/$1$i$2 2>/dev/null); do
      if [ -f "$p" ]; then echo "$p"; return 0; fi
    done
  done
  return 1
}
RAW=$(sacct -j "$J" -X -n -P -o JobIDRaw 2>/dev/null | sed 's/[^0-9]//g' | tr '\n' ' ')
IDS="$J $RAW"

C "$BR/ledger.json" "no staged ledger at $BR/ledger.json (the lab stages it; a compute node cannot read the lab database)" > "$O" 2> "$E"; rc=$?
S ledger $rc

sacct -j "$J" -X -P -o JobID,JobIDRaw,JobName,State,Elapsed,Timelimit,Start,End,Submit,AllocTRES,ExitCode,NodeList > "$RAWF" 2> "$E"; rc=$?
awk '{printf "%06d\t%s\n", NR, $0}' "$RAWF" > "$O" 2>/dev/null
S sacct $rc

{ F=$(P '*' '*.out')
  if [ -n "$F" ]; then
    H "$F"
    awk -v K=__TAILN__ -v M=__MATCHN__ '
      { u=toupper($0)
        if (u ~ /WARN|ERROR|FAIL|TIMEOUT|CANCELLED|TRACEBACK|SLURMSTEPD|SKIPPING|CUDA|KILLED|EXCEEDED|OUT OF MEMORY/) { mi++; mn[mi%M]=NR; mt[mi%M]=$0 }
        tb[NR%K]=$0; n=NR }
      END { print "[block] matches"
            s=mi-M+1; if (s<1) s=1
            for (i=s;i<=mi;i++) printf "%06d\t%s\n", mn[i%M], mt[i%M]
            print "[block] tail"
            s=n-K+1; if (s<1) s=1
            for (i=s;i<=n;i++) printf "%06d\t%s\n", i, tb[i%K]
            printf "[lines] %d\n", n
            printf "[matches] %d kept %d\n", mi, (mi<M?mi:M) }' "$F"
  else
    echo "[none] no job .out under $FW for job id(s) $IDS"
  fi; } > "$O" 2> "$E"; rc=$?
S out_tail $rc

{ F=$(P 'mega_iter*' '*/*/results.csv')
  [ -z "$F" ] && F=$(P 'mega_iter*/job' '/results.csv')
  if [ -n "$F" ]; then
    H "$F"
    awk -v L=__CSVN__ 'NR==1 { hdr=$0; next } { rb[NR%L]=$0; n=NR } END { if (hdr != "") printf "%06d\t%s\n", 1, hdr; s=n-L+1; if (s<2) s=2; for (i=s;i<=n;i++) printf "%06d\t%s\n", i, rb[i%L]; printf "[lines] %d\n", n }' "$F"
  else
    echo "[none] no results.csv under $FW/mega_iter* for job id(s) $IDS"
  fi; } > "$O" 2> "$E"; rc=$?
S results_csv $rc

{ F=$(P 'm1_*_' '.json')
  if [ -n "$F" ]; then H "$F"; NUMBER "$F"; else echo "[none] no job-scoped strategy artifact $FW/m1_*_<jobid>.json"; fi; } > "$O" 2> "$E"; rc=$?
S strategy $rc

{ F=$(P "_brain/$D/trace/*_" '.jsonl')
  if [ -n "$F" ]; then H "$F"; TAILNUM "$F" __TRACEN__; else echo "[none] no trace at $BR/trace/*_<jobid>.jsonl"; fi; } > "$O" 2> "$E"; rc=$?
S trace $rc

C "$FW/dinov2_curator/slug_scores.json" "no slug_scores.json; a curated tier with no score file is itself the finding" > "$O" 2> "$E"; rc=$?
S slug_scores $rc

{ if [ -f "$FW/dataset_registry.json" ]; then H "$FW/dataset_registry.json"; else echo "[none] no dataset_registry.json under $FW"; fi
  echo "[block] prev"
  C "$BR/registry_prev.json" "no staged previous-registry summary at $BR/registry_prev.json"; } > "$O" 2> "$E"; rc=$?
S registry_diff $rc

{ HF=$(ls -td $FW/v3_0_43_brain_harvest_*.out 2>/dev/null | head -1)
  if [ -n "$HF" ]; then
    H "$HF"
    awk -v M=__HARVN__ '
      { u=toupper($0)
        if (u ~ /\[NET\]|\[SRC\]|SKIPPING|SOURCE|CANDIDATE|HARVEST/) { mi++; mn[mi%M]=NR; mt[mi%M]=$0 }
        n=NR }
      END { s=mi-M+1; if (s<1) s=1
            for (i=s;i<=mi;i++) printf "%06d\t%s\n", mn[i%M], mt[i%M]
            printf "[lines] %d\n", n
            printf "[matches] %d kept %d\n", mi, (mi<M?mi:M) }' "$HF"
  else
    echo "[none] no harvest .out under $FW"
  fi
  echo "[block] trace"
  TF=$(ls -td $BR/trace/*collect*.jsonl $BR/trace/*harvest*.jsonl 2>/dev/null | head -1)
  if [ -n "$TF" ]; then H "$TF"; TAILNUM "$TF" __TRACEN__; else echo "[none] no collect/harvest trace under $BR/trace"; fi; } > "$O" 2> "$E"; rc=$?
S harvest $rc

{ df -P -k __PROJ__ 2>/dev/null | awk '{printf "%06d\t%s\n", NR, $0}'
  echo "[block] squeue"
  echo "[value] squeue_depth=$(squeue -h -u "${USER:-nobody}" 2>/dev/null | wc -l | tr -d ' ')"
  echo "[block] quota"
  echo "[none] the per-project quota is not readable from df; VERIFY: run the site quota command for __PROJ__ and stage the answer as $BR/su.json"; } > "$O" 2> "$E"; rc=$?
S resources $rc

C "$BR/su.json" "no staged SU ledger at $BR/su.json" > "$O" 2> "$E"; rc=$?
S su $rc

C "$BR/corrections.json" "no correction mirror at $BR/corrections.json" > "$O" 2> "$E"; rc=$?
S corrections $rc

C "$BR/plan.json" "no active plan at $BR/plan.json" > "$O" 2> "$E"; rc=$?
S plan $rc

case "$T" in */ev_*) rm -rf "$T" 2>/dev/null ;; esac
exit 0
"""

# The order the script emits, and the order the parser expects.
REMOTE_SECTIONS = ("ledger", "sacct", "out_tail", "results_csv", "strategy",
                   "trace", "slug_scores", "registry_diff", "harvest",
                   "resources", "su", "corrections", "plan")


def remote_script(domain, round_num, step, job_id, limits=None, project=None):
    """The ONE shell command that gathers every section for this build.

    Prints section-delimited output `parse_output()` reads back. Takes the
    round and step for the nonce only: the round scheduler's own step is
    identified by its job id on the cluster, and a round number is not a path.
    """
    lim = dict(DEFAULTS)
    if isinstance(limits, dict):
        lim.update({k: v for k, v in limits.items() if k in lim})
    subs = {
        "__EV__": "EV" + nonce(domain, round_num, step, job_id),
        "__JOBID__": _job(job_id) or "0",
        "__DOMAIN__": _domain(domain),
        "__PROJ__": str(project or CLUSTER_PROJECT),
        "__SHAMAX__": str(int(lim["sha256_max_bytes"])),
        "__JSONMAX__": str(int(lim["json_max_bytes"])),
        "__TAILN__": str(max(1, int(lim["out_tail_tail_lines"]))),
        "__MATCHN__": str(max(1, int(lim["out_tail_match_lines"]))),
        "__CSVN__": str(max(1, int(lim["results_csv_lines"]))),
        "__TRACEN__": str(max(1, int(lim["trace_lines"]))),
        "__HARVN__": str(max(1, int(lim["harvest_lines"]))),
    }
    out = _SCRIPT
    for key, val in subs.items():
        out = out.replace(key, val)
    return out


# --- lab -> cluster staging ------------------------------------------------
def stage_commands(domain, payloads, limits=None):
    """Shell commands that write `payloads` under `_brain/<domain>/` on /ocean.

    A compute node cannot reach the lab database, so the ledger tail, the active
    corrections and the active plan travel as files. Each command carries at
    most `stage_max_bytes` of base64 (the relay's per-command ceiling); a
    payload above `gzip_above_bytes` is gzipped first, and one that is still too
    large is split across commands that append and are decoded once at the end.

    Returns a list of shell command strings, in the order they must run.
    """
    lim = _limits({"limits": limits} if isinstance(limits, dict) else None)
    dom = _domain(domain)
    base = "results/framework/_brain/%s" % dom
    cmds = []
    if not isinstance(payloads, dict):
        return cmds
    for name in sorted(payloads):
        safe = _DOM_RE.sub("", str(name).strip().lower())[:40]
        if not safe:
            continue
        try:
            body = corpus.canonical_json(payloads[name]).encode("utf-8")
        except Exception:
            continue                 # an unserialisable payload is not staged
        gz = len(body) > int(lim["gzip_above_bytes"])
        if gz:
            # mtime=0 so the same payload produces the same command bytes; a
            # staging command that changes every second is not diffable.
            body = gzip.compress(body, 9, mtime=0)
        blob = base64.b64encode(body).decode("ascii")
        # Room for the wrapper around the payload on each command.
        room = max(1024, int(lim["stage_max_bytes"]) - 512)
        chunks = [blob[i:i + room] for i in range(0, len(blob), room)] or [""]
        dst = "%s/%s.json" % (base, safe)
        tmp = "%s/.%s.b64" % (base, safe)
        for idx, chunk in enumerate(chunks):
            redirect = ">" if idx == 0 else ">>"
            cmd = ("mkdir -p %s && printf '%%s' %s %s %s"
                   % (base, chunk, redirect, tmp))
            if idx == len(chunks) - 1:
                # base64 -d is coreutils; VERIFY: `base64 --help` on the cluster
                # (BSD base64 spells it -D and would need the other flag).
                decode = "base64 -d < %s" % tmp
                if gz:
                    decode += " | gunzip"
                cmd += (" && %s > %s.tmp && mv -f %s.tmp %s && rm -f %s"
                        % (decode, dst, dst, dst, tmp))
            cmds.append(cmd)
    return cmds


def _pack(stage_cmds, gather, limits):
    """Fold staging into the gather command while it fits the relay ceiling.

    One command per build is the target; the ceiling is what actually forces a
    second one.
    """
    room = int(limits["stage_max_bytes"])
    packed, cur = [], ""
    for cmd in list(stage_cmds) + [gather]:
        if cur and len(cur) + len(cmd) + 2 > room:
            packed.append(cur)
            cur = cmd
        else:
            cur = cmd if not cur else (cur + "\n" + cmd)
    if cur:
        packed.append(cur)
    return packed


# --- parsing the gathered output ------------------------------------------
def _payload_bytes(lines):
    """Bytes the shell would have counted for these content lines.

    The remote side normalises every payload through `awk 1`, so each line ends
    with exactly one newline; that is what `wc -c` counted and what the END
    marker declares.
    """
    return sum(len(l.encode("utf-8")) + 1 for l in lines)


def _blank_block():
    return {"file": None, "pairs": [], "values": {}, "none": None,
            "lines_total": None, "matches": None, "truncated": None}


def _parse_kv(text):
    out = {}
    for part in text.split():
        if "=" in part:
            key, val = part.split("=", 1)
            out[key] = val
    return out


def _parse_payload(lines):
    """Payload lines -> {block_name: block}. Block "" is the section itself."""
    blocks = {"": _blank_block()}
    cur = ""
    stderr = []
    notes = []
    for raw in lines:
        m = _LINE_RE.match(raw)
        if m:
            blocks[cur]["pairs"].append([int(m.group(1)), m.group(2)])
            continue
        if raw.startswith("[block] "):
            cur = raw[8:].strip()
            blocks.setdefault(cur, _blank_block())
            continue
        if raw.startswith("[file] "):
            kv = _parse_kv(raw[7:])
            blocks[cur]["file"] = {
                "path": kv.get("path", ""),
                "bytes": int(_num(kv.get("bytes")) or 0),
                "mtime": _num(kv.get("mtime")) or 0.0,
                "sha256": kv.get("sha256") or None,
            }
            continue
        if raw.startswith("[none] "):
            blocks[cur]["none"] = raw[7:].strip()
            continue
        if raw.startswith("[lines] "):
            blocks[cur]["lines_total"] = int(_num(raw[8:].strip()) or 0)
            continue
        if raw.startswith("[matches] "):
            nums = re.findall(r"\d+", raw[10:])
            if len(nums) >= 2:
                blocks[cur]["matches"] = [int(nums[0]), int(nums[1])]
            continue
        if raw.startswith("[truncated] "):
            blocks[cur]["truncated"] = raw[12:].strip()
            continue
        if raw.startswith("[value] "):
            blocks[cur]["values"].update(_parse_kv(raw[8:]))
            continue
        if raw.startswith("[stderr] "):
            stderr.append(raw[9:])
            continue
        if raw.strip():
            notes.append(raw[:200])
    return blocks, stderr, notes


def parse_output(text, ev):
    """Section-delimited gather output -> {"sections": {...}, "notes": [...]}.

    Two hostile shapes are handled explicitly, because both have a plausible
    route into a job log:

    * **A line that looks like a delimiter.** A candidate END marker closes its
      section only when the payload accumulated so far is exactly the byte count
      the shell declared on that marker. Anything else is content, and the
      collision is recorded. A BEGIN inside an open section is always content.
    * **A section that never ends.** A cut channel leaves the last section open;
      it is returned with `complete` False and whatever arrived, so a truncated
      transfer is visible instead of being read as an empty section.
    """
    result = {"sections": {}, "notes": [], "unclaimed": 0}
    if not isinstance(text, str):
        result["notes"].append("gather produced no text (%s)" % type(text).__name__)
        return result
    begin_re = re.compile(r"^%s BEGIN (\S+)$" % re.escape(ev))
    end_re = re.compile(r"^%s END (\S+) rc=(-?\d+) bytes=(\d+)$" % re.escape(ev))
    cur, acc = None, []
    for raw in text.split("\n"):
        line = raw.rstrip("\r")
        if cur is None:
            m = begin_re.match(line)
            if m:
                cur, acc = m.group(1), []
            elif line.strip():
                result["unclaimed"] += 1
            continue
        m = end_re.match(line)
        if m and m.group(1) == cur:
            declared = int(m.group(3))
            seen = _payload_bytes(acc)
            if declared == seen:
                blocks, stderr, notes = _parse_payload(acc)
                result["sections"][cur] = {
                    "rc": int(m.group(2)), "bytes_declared": declared,
                    "bytes_seen": seen, "complete": True, "blocks": blocks,
                    "stderr": stderr, "notes": notes}
                cur, acc = None, []
                continue
            result["notes"].append(
                "section %s: a payload line matched its own end marker "
                "(declared %d bytes, %d seen at that point) and was kept as "
                "content" % (cur, declared, seen))
            acc.append(line)
            continue
        if begin_re.match(line) or end_re.match(line):
            result["notes"].append(
                "section %s: a payload line is delimiter-shaped and was kept "
                "as content" % cur)
        acc.append(line)
    if cur is not None:
        blocks, stderr, notes = _parse_payload(acc)
        result["sections"][cur] = {
            "rc": None, "bytes_declared": None, "bytes_seen": _payload_bytes(acc),
            "complete": False, "blocks": blocks, "stderr": stderr, "notes": notes}
        result["notes"].append(
            "section %s never terminated: the transfer was cut, so this "
            "section holds only what arrived" % cur)
    return result


# --- section values --------------------------------------------------------
def _trim_out_tail(pairs, cap, tail_n):
    """WARN/ERROR/TIMEOUT/SKIPPING lines plus the tail, capped.

    Same rule as corpus._trim_out_tail, with SKIPPING added to the keep set;
    the earliest matches go first because a job's failure is at its end.
    """
    if not pairs:
        return []
    tail_from = pairs[max(0, len(pairs) - int(tail_n))][0]
    sel = [p for p in pairs
           if p[0] >= tail_from or OUT_TAIL_KEEP_RE.search(p[1])]
    if cap and len(sel) > int(cap):
        sel = sel[-int(cap):]
    return sel


def _json_of(block):
    """(object, reason) from a block's numbered lines."""
    if not block["pairs"]:
        return None, (block["none"] or "the section carried no lines")
    obj, why = corpus._parse_json_artifact(block["pairs"])
    if why:
        if block.get("truncated"):
            why = "%s (%s)" % (why, block["truncated"])
        return None, why
    return obj, None


def _resources_value(sec, lim):
    """{df_project, quota_headroom_gb, fs_free_tb, squeue_depth} from df/squeue.

    Filesystem free and project quota headroom are different numbers and are
    reported separately; the quota one is `null` with a reason unless the lab
    staged it, because df cannot see a per-project allocation.
    """
    main = sec["blocks"].get("", _blank_block())
    out = {"df_project": None, "quota_headroom_gb": None, "fs_free_tb": None,
           "squeue_depth": None,
           "quota_reason": "not read: df reports the filesystem, not the "
                           "project allocation"}
    rows = corpus._table_rows([[n, t] for n, t in main["pairs"]], 20)
    if main["pairs"]:
        out["df_project"] = rows
        cells = main["pairs"][-1][1].split()
        if len(cells) >= 4:
            free_k = _num(cells[3])
            if free_k is not None:
                out["fs_free_tb"] = round(free_k / (1024.0 ** 3), 3)
    sq = sec["blocks"].get("squeue", _blank_block())["values"].get("squeue_depth")
    if sq is not None:
        depth = _num(sq)
        out["squeue_depth"] = int(depth) if depth is not None else None
    quota = sec["blocks"].get("quota", _blank_block())
    if quota.get("none"):
        out["quota_reason"] = quota["none"]
    return out


_TRES_GPU_RE = re.compile(r"gres/gpu(?::([a-z0-9\-]+))?=(\d+)", re.I)


def _elapsed_s(text):
    """SLURM Elapsed ("[DD-]HH:MM:SS") in seconds, or None."""
    s = str(text or "").strip()
    if not s:
        return None
    days = 0
    if "-" in s:
        head, s = s.split("-", 1)
        days = int(_num(head) or 0)
    parts = s.split(":")
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return None
    while len(parts) < 3:
        parts.insert(0, 0)
    return days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def _su_of_row(row, lim):
    """SU for one sacct allocation row: GPUs x elapsed hours x the family rate."""
    tres = str(row.get("AllocTRES") or row.get("ReqTRES") or "")
    m = _TRES_GPU_RE.search(tres)
    if not m:
        return None, None
    gpus = int(m.group(2))
    fam = (m.group(1) or "").lower()
    rate = lim["su_default_per_gpu_hour"]
    if "h100" in fam:
        rate = lim["su_h100_per_gpu_hour"]
    elif "v100" in fam:
        rate = lim["su_v100_per_gpu_hour"]
    elapsed = _elapsed_s(row.get("Elapsed"))
    if elapsed is None:
        return None, fam or None
    return round(gpus * (elapsed / 3600.0) * rate, 3), fam or None


def _su_value(sacct_rows, staged, lim):
    """{round, campaign, envelope} plus this job's own measured SU.

    The job number is measured from sacct here; the round and campaign totals
    and the envelope come from the lab's SU ledger when it was staged, because
    a compute node cannot add up a campaign it cannot read.
    """
    out = {"job": None, "round": None, "campaign": None, "envelope": None,
           "gpu_family": None,
           "rate_su_per_gpu_hour": {"h100": lim["su_h100_per_gpu_hour"],
                                    "v100": lim["su_v100_per_gpu_hour"],
                                    "default": lim["su_default_per_gpu_hour"]},
           "reason": None}
    for row in (sacct_rows or []):
        if not isinstance(row, dict):
            continue
        su, fam = _su_of_row(row, lim)
        if su is not None:
            out["job"], out["gpu_family"] = su, fam
            break
    if isinstance(staged, dict):
        for key in ("round", "campaign", "envelope"):
            if staged.get(key) is not None:
                out[key] = staged[key]
    else:
        out["reason"] = ("no SU ledger was staged; round and campaign totals "
                         "are unknown, not zero")
    return out


def _harvest_value(sec, lim):
    """{per_source, ...} from the harvest log lines and the collect trace.

    `per_source` is computed only from candidate records in the trace, whose
    fields are defined by the writer. When there is no trace it is `null` with a
    reason: counting sources out of free-text log lines would invent a
    vocabulary, and `source_degraded` would then fire on a parser's guess.
    """
    main = sec["blocks"].get("", _blank_block())
    tr = sec["blocks"].get("trace", _blank_block())
    value = {"per_source": None, "per_source_reason": None,
             "artifact_id": None, "path": None, "sha256": None, "lines": []}
    if main["file"]:
        value["path"] = main["file"]["path"]
        value["artifact_id"] = os.path.basename(main["file"]["path"])
        value["sha256"] = main["file"]["sha256"]
    value["lines"] = main["pairs"][-int(lim["harvest_lines"]):]
    if not main["pairs"] and main.get("none"):
        value["lines_reason"] = main["none"]
    recs = corpus._jsonl_records(tr["pairs"], int(lim["trace_records"])) \
        if tr["pairs"] else []
    cands = [r for r in recs if str(r.get("kind") or "") == "candidate"]
    if not tr["pairs"]:
        value["per_source_reason"] = tr.get("none") or "no collect trace was read"
    elif not cands:
        value["per_source_reason"] = ("the collect trace carries no candidate "
                                      "records in the window that was read")
    else:
        per = {}
        for rec in cands:
            src = str(rec.get("source") or "unknown")
            row = per.setdefault(src, {"candidates": 0, "verdicts": {},
                                       "images": 0})
            row["candidates"] += 1
            verdict = str(rec.get("verdict") or "unrecorded")
            row["verdicts"][verdict] = row["verdicts"].get(verdict, 0) + 1
            imgs = _num(rec.get("images"))
            if imgs is not None:
                row["images"] += int(imgs)
        value["per_source"] = per
        value["records_read"] = len(recs)
    return value


def _registry_value(sec):
    """Registry identity now, plus the staged previous snapshot when there is one."""
    main = sec["blocks"].get("", _blank_block())
    prev_block = sec["blocks"].get("prev", _blank_block())
    prev, why = _json_of(prev_block) if prev_block["pairs"] else (None, None)
    out = {"registry": main["file"], "prev": prev, "added_slugs": None,
           "reason": None}
    if main["file"] is None:
        out["reason"] = main.get("none") or "no registry was read"
        return out
    if prev is None:
        out["reason"] = (why or prev_block.get("none")
                         or "no previous snapshot was staged; only the current "
                            "registry's identity is reported")
        return out
    now_slugs = prev.get("_current_slugs")
    old = prev.get("slugs")
    if isinstance(old, list) and isinstance(now_slugs, list):
        out["added_slugs"] = sorted(set(now_slugs) - set(old))
    else:
        out["reason"] = ("the staged snapshot carries no slug list, so the "
                         "diff is unknown; identity and mtime are reported")
    return out


def _section_values(parsed, lim):
    """Every section value, plus the artifact manifest and the missing reasons."""
    sections, missing, source, artifacts, warnings = {}, {}, {}, [], []

    for name in REMOTE_SECTIONS:
        sec = parsed["sections"].get(name)
        if sec is None:
            sections[name] = None
            missing[name] = "the gather produced no %s section" % name
            continue
        main = sec["blocks"].get("", _blank_block())
        if main["file"]:
            artifacts.append({
                "artifact_id": os.path.basename(main["file"]["path"]),
                "path": main["file"]["path"], "sha256": main["file"]["sha256"],
                "bytes": main["file"]["bytes"], "mtime": main["file"]["mtime"],
                "lines": main["lines_total"], "section": name,
                "kept_lines": len(main["pairs"]), "present": True,
                "reason": None})
        if not sec["complete"]:
            warnings.append("section %s arrived truncated" % name)
        if sec["rc"] not in (0, None):
            sections[name] = None
            why = "command failed (rc=%s)" % sec["rc"]
            if sec["stderr"]:
                why += ": " + " | ".join(sec["stderr"])[:300]
            missing[name] = why
            continue
        # A section whose only block is a stated absence is null with that
        # reason. Sections that carry more than one block (harvest's log plus
        # its trace, registry_diff's identity plus the staged snapshot) are
        # still built: one empty block is not an empty section.
        only_main = set(sec["blocks"]) <= {""}
        if main.get("none") and not main["pairs"] and only_main:
            sections[name] = None
            missing[name] = main["none"]
            continue
        value, why = _one_section(name, sec, main, lim)
        if why:
            sections[name] = None
            missing[name] = why
        else:
            sections[name] = value
            if main["file"]:
                source[name] = os.path.basename(main["file"]["path"])
        if main.get("truncated"):
            warnings.append("section %s: %s" % (name, main["truncated"]))
        if main.get("matches") and main["matches"][0] > main["matches"][1]:
            warnings.append(
                "section %s: %d matching lines existed, the last %d were kept"
                % (name, main["matches"][0], main["matches"][1]))
        if (name == "results_csv" and isinstance(sections.get(name), dict)
                and main["lines_total"]):
            rows = max(0, int(main["lines_total"]) - 1)
            if rows > sections[name].get("rows", 0):
                warnings.append(
                    "section results_csv: the file has %d data rows, the last "
                    "%d were read" % (rows, sections[name]["rows"]))
                sections[name]["rows"] = rows
    return sections, missing, source, artifacts, warnings


def _one_section(name, sec, main, lim):
    """(value, reason) for one gathered section."""
    if name == "out_tail":
        if not main["file"]:
            return None, (main.get("none") or "no .out was read")
        pairs = {}
        for block in ("matches", "tail", ""):
            for pair in sec["blocks"].get(block, _blank_block())["pairs"]:
                pairs[pair[0]] = pair[1]        # the blocks overlap by design
        ordered = [[n, pairs[n]] for n in sorted(pairs)]
        return {"artifact_id": os.path.basename(main["file"]["path"]),
                "path": main["file"]["path"], "sha256": main["file"]["sha256"],
                "lines": _trim_out_tail(ordered, lim["out_tail_lines"],
                                        lim["out_tail_tail_lines"])}, None
    if name == "sacct":
        rows = corpus._table_rows(main["pairs"], int(lim["sacct_rows"]))
        if not rows:
            return None, "sacct returned no rows for this job"
        return rows, None
    if name == "results_csv":
        if not main["file"]:
            return None, (main.get("none") or "no results.csv was read")
        return corpus._results_csv_summary(main["pairs"],
                                           main["file"]["mtime"]), None
    if name == "trace":
        recs = corpus._jsonl_records(main["pairs"], int(lim["trace_records"]))
        if not recs:
            return None, (main.get("none")
                          or "the trace carried no parseable records")
        return recs, None
    if name == "slug_scores":
        obj, why = _json_of(main)
        if why:
            return None, why
        return corpus._slug_scores_summary(
            obj, main["file"]["mtime"] if main["file"] else None), None
    if name == "registry_diff":
        return _registry_value(sec), None
    if name == "harvest":
        return _harvest_value(sec, lim), None
    if name == "resources":
        return _resources_value(sec, lim), None
    if name == "su":
        obj, _ = _json_of(main)
        return obj if isinstance(obj, dict) else None, None
    # ledger, strategy, corrections, plan: staged or job-written JSON.
    obj, why = _json_of(main)
    if why:
        return None, why
    if name == "ledger" and isinstance(obj, dict):
        rounds = obj.get("rounds")
        if isinstance(rounds, list) and len(rounds) > int(lim["ledger_rounds"]):
            obj = dict(obj, rounds=rounds[-int(lim["ledger_rounds"]):])
    return obj, None


# --- signals ---------------------------------------------------------------
def _signals(bundle, ctx):
    """(signals, reason). Never guesses: no detector means `null`, not `[]`."""
    fn = (ctx or {}).get("signals_fn") if isinstance(ctx, dict) else None
    if callable(fn):
        try:
            got = fn(bundle)
            return [s for s in (got or []) if isinstance(s, dict)], None
        except Exception as exc:
            return None, "the supplied signals callable raised %s: %s" % (
                type(exc).__name__, str(exc)[:160])
    try:
        from . import signals as sigmod          # noqa: WPS433 (optional peer)
    except Exception:
        try:
            import signals as sigmod             # type: ignore
        except Exception:
            return None, ("no signals module is importable, so the signals are "
                          "unknown rather than empty")
    detect = None
    for fname in ("detect", "run", "evaluate", "all_signals"):
        cand = getattr(sigmod, fname, None)
        if callable(cand):
            detect = cand
            break
    if detect is None:
        return None, "the signals module exports no detect()"
    try:
        got = detect(bundle)
    except Exception as exc:
        return None, "signals.detect raised %s: %s" % (type(exc).__name__,
                                                       str(exc)[:160])
    return [s for s in (got or []) if isinstance(s, dict)], None


# --- trimming --------------------------------------------------------------
def _size(obj):
    """Characters of canonical JSON — the unit token_estimate divides by 4."""
    try:
        return len(corpus.canonical_json(obj))
    except Exception:
        return 0


def _t_out_tail_half(sections, lim, arg):
    sec = sections.get("out_tail")
    if not isinstance(sec, dict) or not sec.get("lines"):
        return None
    before = len(sec["lines"])
    keep = max(int(lim["out_tail_tail_lines"]), before // 2)
    if keep >= before:
        return None
    sec["lines"] = sec["lines"][-keep:]
    return {"action": "keep the last %d of %d line(s)" % (keep, before),
            "removed": before - keep}


def _t_out_tail_floor(sections, lim, arg):
    """Down to the floor: every kept-pattern line plus the last `arg`."""
    sec = sections.get("out_tail")
    if not isinstance(sec, dict) or not sec.get("lines"):
        return None
    before = len(sec["lines"])
    floor = _trim_out_tail(sec["lines"], None, int(arg))
    if len(floor) >= before:
        return None
    sec["lines"] = floor
    return {"action": ("keep every WARN/ERROR/FAIL/TIMEOUT/CANCELLED/SKIPPING "
                       "line plus the last %d: %d of %d line(s)"
                       % (int(arg), len(floor), before)),
            "removed": before - len(floor)}


def _t_trace(sections, lim, arg):
    sec = sections.get("trace")
    if not isinstance(sec, list) or len(sec) <= int(arg):
        return None
    before = len(sec)
    sections["trace"] = sec[-int(arg):]
    return {"action": "keep the last %d of %d record(s)" % (int(arg), before),
            "removed": before - int(arg)}


def _t_harvest_lines(sections, lim, arg):
    sec = sections.get("harvest")
    if not isinstance(sec, dict) or not sec.get("lines"):
        return None
    before = len(sec["lines"])
    if before <= int(arg):
        return None
    sec["lines"] = sec["lines"][-int(arg):]
    return {"action": "keep the last %d of %d log line(s)" % (int(arg), before),
            "removed": before - int(arg)}


def _t_registry(sections, lim, arg):
    sec = sections.get("registry_diff")
    if not isinstance(sec, dict) or sec.get("prev") is None:
        return None
    before = _size(sec)
    sec["prev"] = None
    sec["reason"] = ("the staged previous snapshot was dropped to fit the "
                     "budget; the current registry's identity is kept")
    return {"action": "drop the staged previous snapshot",
            "removed": before - _size(sec)}


def _t_ledger(sections, lim, arg):
    sec = sections.get("ledger")
    rounds = sec.get("rounds") if isinstance(sec, dict) else None
    if not isinstance(rounds, list) or len(rounds) <= int(arg):
        return None
    before = len(rounds)
    sec["rounds"] = rounds[-int(arg):]
    return {"action": "keep the last %d of %d round(s)" % (int(arg), before),
            "removed": before - int(arg)}


def _t_list(section):
    def _fn(sections, lim, arg):
        sec = sections.get(section)
        items = sec if isinstance(sec, list) else (
            sec.get("items") if isinstance(sec, dict) else None)
        if not isinstance(items, list) or len(items) <= int(arg):
            return None
        before = len(items)
        kept = items[-int(arg):]
        if isinstance(sec, list):
            sections[section] = kept
        else:
            sec["items"] = kept
        return {"action": "keep the last %d of %d entr(ies)" % (int(arg), before),
                "removed": before - int(arg)}
    return _fn


# Priority order, most droppable first. Each stage names the section it touches
# and the argument it trims to; the loop stops as soon as the budget is met.
TRIM_STAGES = (
    ("out_tail", _t_out_tail_half, None),
    ("trace", _t_trace, 10),
    ("harvest", _t_harvest_lines, 20),
    ("registry_diff", _t_registry, None),
    ("out_tail", _t_out_tail_floor, 40),
    ("ledger", _t_ledger, 2),
    ("corrections", _t_list("corrections"), 5),
    ("plan", _t_list("plan"), 3),
    ("ledger", _t_ledger, 1),
)


def trim(bundle, budget_tokens):
    """A copy of `bundle` trimmed to `budget_tokens`, saying what it removed.

    Priority order: `sacct`, `strategy` and `signals` are never touched; the
    `.out` tail goes first and never below its floor (every WARN, ERROR, FAIL,
    TIMEOUT, CANCELLED and SKIPPING line plus the last lines). Every stage that
    fires is recorded in `export.trim.steps` with what it removed, and a bundle
    that cannot reach the budget comes back with `ok` False rather than quietly
    short — the caller must be able to see that it is about to overflow.
    """
    try:
        out = json.loads(json.dumps(bundle, default=str))
    except Exception:
        return bundle
    if not isinstance(out, dict) or not isinstance(out.get("sections"), dict):
        return bundle
    lim = _limits(None)
    sections = out["sections"]
    before = corpus.token_estimate(sections)
    record = {"budget_tokens": None, "before_tokens": before,
              "after_tokens": before, "ok": True, "protected": list(NEVER_TRIM),
              "steps": [], "reason": None}
    budget = _num(budget_tokens)
    if budget is None or budget <= 0:
        record["reason"] = "no budget was given; nothing was trimmed"
        _put_trim(out, record)
        return out
    record["budget_tokens"] = int(budget)
    for name, fn, arg in TRIM_STAGES:
        if corpus.token_estimate(sections) <= budget:
            break
        if name in NEVER_TRIM or sections.get(name) is None:
            continue
        was = corpus.token_estimate(sections)
        try:
            got = fn(sections, lim, arg)
        except Exception as exc:
            record["steps"].append({"section": name, "action": "skipped",
                                    "removed": 0, "tokens_saved": 0,
                                    "reason": "%s: %s" % (type(exc).__name__,
                                                          str(exc)[:120])})
            continue
        if not got:
            continue
        now = corpus.token_estimate(sections)
        got.update({"section": name, "tokens_saved": was - now})
        record["steps"].append(got)
    after = corpus.token_estimate(sections)
    record["after_tokens"] = after
    if after > budget:
        record["ok"] = False
        record["reason"] = (
            "the bundle is %d tokens against a budget of %d after every trim "
            "stage; sacct, the strategy JSON, the signals and the .out floor "
            "are not trimmable, so the caller must raise num_ctx or split the "
            "review" % (after, int(budget)))
    out["token_estimate"] = after
    _put_trim(out, record)
    _seal(out)
    return out


def _put_trim(bundle, record):
    exp = bundle.get("export")
    if not isinstance(exp, dict):
        exp = {}
        bundle["export"] = exp
    exp["trim"] = record


# --- assembly --------------------------------------------------------------
def _empty_sections():
    return {name: None for name in SECTIONS}


def _seal(bundle):
    """Recompute token_estimate and the bundle hash, in that order."""
    bundle["sha256"] = ""
    bundle["token_estimate"] = corpus.token_estimate(bundle.get("sections") or {})
    bundle["sha256"] = corpus.sha256_str(corpus.canonical_json(bundle))
    return bundle


def _runner(ctx):
    """The caller's remote channel: `run`, `slurm_sh` or a bare callable."""
    if callable(ctx):
        return ctx
    if not isinstance(ctx, dict):
        return None
    for key in ("run", "slurm_sh", "sh"):
        fn = ctx.get(key)
        if callable(fn):
            return fn
    return None


def _run(fn, cmd, timeout):
    """(stdout, note). Never raises; a runner that misbehaves is a note."""
    try:
        try:
            res = fn(cmd, timeout)
        except TypeError:
            res = fn(cmd)
    except Exception as exc:
        return "", "remote command failed: %s: %s" % (type(exc).__name__,
                                                      str(exc)[:200])
    if isinstance(res, str):
        return res, None
    if isinstance(res, dict):
        out = res.get("stdout")
        note = None
        if res.get("ok") is False or (res.get("returncode") not in (None, 0)):
            note = ("remote command returned rc=%s: %s"
                    % (res.get("returncode"),
                       str(res.get("stderr") or "")[:200]))
        return (out if isinstance(out, str) else ""), note
    return "", "remote runner returned %s, not text" % type(res).__name__


def build(domain, round_num, step, job_id, ctx=None):
    """The evidence bundle for a live step. Never raises.

    `ctx` carries the machine, not the policy:
      run / slurm_sh   callable(command, timeout) -> {"stdout", ...} or str
      captured         pre-recorded gather output (skips the runner)
      stage            {"ledger": obj, "corrections": obj, "plan": obj, ...}
                       staged to /ocean in the same batched command when it fits
      num_ctx          the model window this bundle must fit inside
      budget_tokens    a tighter budget than the pre-registered default
      signals_fn       callable(bundle) -> [Signal]; else signals.py is imported
      limits, project, out_dir, log
    """
    started = time.time()
    ctx = ctx if isinstance(ctx, (dict,)) or callable(ctx) else {}
    cdict = ctx if isinstance(ctx, dict) else {}
    lim = _limits(cdict)
    dom, rnd, stp, jid = (_domain(domain), _round(round_num), _step(step),
                          _job(job_id))
    built_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))
    bundle = {
        "bundle_id": "%s_r%s_%s_%s_%d" % (dom, rnd, stp, jid or "nojob",
                                          int(started)),
        "sha256": "", "domain": dom, "round": rnd, "step": stp,
        "built_ts": built_ts, "sections": _empty_sections(),
        "token_estimate": 0,
        "caps": {"out_tail_lines": lim["out_tail_lines"],
                 "out_tail_tail_lines": lim["out_tail_tail_lines"],
                 "sacct_rows": lim["sacct_rows"],
                 "trace_records": lim["trace_records"],
                 "ledger_rounds": lim["ledger_rounds"],
                 "harvest_lines": lim["harvest_lines"],
                 "results_csv_lines": lim["results_csv_lines"],
                 "token_budget": lim["token_budget"]},
        "export": {"tool_version": TOOL_VERSION, "case_id": None,
                   "provenance": "live", "provenance_reason": None,
                   "job_id": jid or None, "artifacts": [], "missing": {},
                   "section_source": {}, "scrub": {}, "scrub_total": 0,
                   "unresolved_load_bearing": [], "warnings": [],
                   "build": {"built_epoch": round(started, 3),
                             "remote_calls": 0, "commands": [],
                             "nonce": nonce(dom, rnd, stp, jid),
                             "elapsed_s": None, "num_ctx": None,
                             "over_num_ctx": False, "notes": []}}}
    warn = bundle["export"]["warnings"]
    info = bundle["export"]["build"]
    try:
        ev = "EV" + info["nonce"]
        gather = remote_script(dom, rnd, stp, jid, limits=lim,
                               project=cdict.get("project"))
        text = cdict.get("captured")
        if isinstance(text, str):
            info["notes"].append("replayed a captured gather; nothing was run")
        else:
            stage = cdict.get("stage") if isinstance(cdict.get("stage"), dict) else {}
            commands = _pack(stage_commands(dom, stage, lim), gather, lim)
            info["commands"] = [len(c) for c in commands]
            fn = _runner(ctx)
            if fn is None:
                warn.append("no remote runner in ctx: nothing was gathered")
                text = ""
            else:
                text = ""
                for idx, cmd in enumerate(commands):
                    out, note = _run(fn, cmd, int(lim["remote_timeout_s"]))
                    info["remote_calls"] += 1
                    if note:
                        warn.append(note)
                    if idx == len(commands) - 1:
                        text = out
        parsed = parse_output(text or "", ev)
        for note in parsed["notes"]:
            warn.append(note)
        if parsed["unclaimed"]:
            warn.append("%d line(s) of gather output sat outside any section "
                        "(a login profile writing to stdout does this)"
                        % parsed["unclaimed"])
        sections, missing, source, artifacts, more = _section_values(parsed, lim)
        bundle["sections"].update(sections)
        bundle["export"]["missing"].update(missing)
        bundle["export"]["section_source"].update(source)
        bundle["export"]["artifacts"] = artifacts
        warn.extend(more)
        if isinstance(bundle["sections"].get("sacct"), list):
            # This job's own SU is measured from sacct; the round and campaign
            # totals can only come from the lab, so they stay null with a
            # reason when nothing was staged.
            staged_su = bundle["sections"].get("su")
            bundle["sections"]["su"] = _su_value(bundle["sections"]["sacct"],
                                                 staged_su, lim)
            bundle["export"]["missing"].pop("su", None)
            bundle["export"]["section_source"]["su"] = (
                "sacct" if not isinstance(staged_su, dict) else "sacct+staged")
        for name in SECTIONS:
            if bundle["sections"].get(name) is None:
                bundle["export"]["missing"].setdefault(
                    name, "no artifact behind this section in this build")
        sigs, why = _signals(bundle, cdict)
        bundle["sections"]["signals"] = sigs
        if sigs is None:
            bundle["export"]["missing"]["signals"] = why
        else:
            bundle["export"]["missing"].pop("signals", None)
            bundle["export"]["section_source"]["signals"] = "signals.detect"
    except Exception as exc:              # a bundle is never an exception
        warn.append("build failed after the gather: %s: %s"
                    % (type(exc).__name__, str(exc)[:300]))
    try:
        _seal(bundle)
        num_ctx = _num(cdict.get("num_ctx"))
        budget = _num(cdict.get("budget_tokens")) or lim["token_budget"]
        if num_ctx:
            info["num_ctx"] = int(num_ctx)
            budget = min(budget, num_ctx - lim["prompt_reserve_tokens"])
        if bundle["token_estimate"] > budget:
            bundle = trim(bundle, budget)
            info = bundle["export"]["build"]
            warn = bundle["export"]["warnings"]
            warn.append("the bundle was trimmed to %d tokens against a budget "
                        "of %d" % (bundle["token_estimate"], int(budget)))
        if num_ctx and bundle["token_estimate"] > num_ctx:
            info["over_num_ctx"] = True
            warn.append("REFUSED: %d tokens still exceed num_ctx %d; this "
                        "bundle must not be sent as it is"
                        % (bundle["token_estimate"], int(num_ctx)))
        info["elapsed_s"] = round(time.time() - started, 3)
        _seal(bundle)
        out_dir = cdict.get("out_dir")
        if out_dir:
            path = os.path.join(str(out_dir), "%s_%s_%d.json"
                                % (rnd, stp, int(started)))
            try:
                corpus.write_json(path, bundle)
                info["path"] = path
            except Exception as exc:
                warn.append("could not write the bundle to %s: %s"
                            % (path, type(exc).__name__))
    except Exception as exc:
        try:
            bundle["export"]["warnings"].append(
                "sealing failed: %s: %s" % (type(exc).__name__, str(exc)[:200]))
        except Exception:
            pass
    log = cdict.get("log")
    if log is not None:
        try:
            log.info("[evidence] %s r%s %s job %s: %d tokens, %d/%d sections, "
                     "%d warning(s)"
                     % (dom, rnd, stp, jid or "-", bundle["token_estimate"],
                        sum(1 for n in SECTIONS
                            if bundle["sections"].get(n) is not None),
                        len(SECTIONS), len(bundle["export"]["warnings"])))
        except Exception:
            pass
    return bundle


# --- validation ------------------------------------------------------------
def validate(bundle):
    """{"ok", "errors", "warnings"} against the frozen bundle shape.

    Checks what a consumer depends on: the key set corpus.py writes, all 14
    sections present, a token estimate computed the one way, a hash that
    verifies, and line-addressed sections whose lines really are
    [absolute_line, text].
    """
    rep = {"ok": False, "errors": [], "warnings": []}
    if not isinstance(bundle, dict):
        rep["errors"].append("bundle is not an object")
        return rep
    want = {"bundle_id", "sha256", "domain", "round", "step", "built_ts",
            "sections", "token_estimate", "caps", "export"}
    got = set(bundle)
    for key in sorted(want - got):
        rep["errors"].append("missing top-level key %s" % key)
    for key in sorted(got - want):
        rep["errors"].append("unexpected top-level key %s" % key)
    sections = bundle.get("sections")
    if not isinstance(sections, dict):
        rep["errors"].append("sections is not an object")
        sections = {}
    for name in SECTIONS:
        if name not in sections:
            rep["errors"].append("section %s is absent (null is the way to say "
                                 "there was nothing behind it)" % name)
        elif sections[name] is None:
            miss = (bundle.get("export") or {}).get("missing") or {}
            if not miss.get(name):
                rep["errors"].append("section %s is null with no reason" % name)
            else:
                rep["warnings"].append("section %s: %s" % (name, miss[name]))
    for name in ("out_tail", "harvest"):
        sec = sections.get(name)
        if isinstance(sec, dict) and isinstance(sec.get("lines"), list):
            for row in sec["lines"]:
                if (not isinstance(row, list) or len(row) != 2
                        or not isinstance(row[0], int)):
                    rep["errors"].append(
                        "section %s carries a line that is not "
                        "[absolute_line, text]" % name)
                    break
    est = corpus.token_estimate(sections)
    if bundle.get("token_estimate") != est:
        rep["errors"].append("token_estimate %r is not corpus.token_estimate "
                             "%d" % (bundle.get("token_estimate"), est))
    body = dict(bundle, sha256="")
    if bundle.get("sha256") != corpus.sha256_str(corpus.canonical_json(body)):
        rep["errors"].append("sha256 does not verify over the bundle")
    rep["ok"] = not rep["errors"]
    return rep


# --- CLI -------------------------------------------------------------------
def main(argv=None):
    """Print the gather script, or build a bundle from captured output.

    This module never opens a channel of its own, so `build` here reads the
    gather output from a file: run the printed script on the cluster, bring the
    text back, and the bundle is assembled from it.
    """
    ap = argparse.ArgumentParser(prog="evidence", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    for name in ("script", "build"):
        p = sub.add_parser(name)
        p.add_argument("--domain", required=True)
        p.add_argument("--round", default=None)
        p.add_argument("--step", required=True)
        p.add_argument("--jobid", required=True)
        if name == "build":
            p.add_argument("--from-file", required=True)
            p.add_argument("--out", default=None)
            p.add_argument("--num-ctx", type=int, default=None)
    try:
        args = ap.parse_args(argv)
    except SystemExit:
        return 2
    if not args.cmd:
        ap.print_help()
        return 2
    if args.cmd == "script":
        sys.stdout.write(remote_script(args.domain, args.round, args.step,
                                       args.jobid))
        return 0
    try:
        with open(args.from_file, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError as exc:
        sys.stderr.write("cannot read %s: %s\n" % (args.from_file, exc))
        return 1
    bundle = build(args.domain, args.round, args.step, args.jobid,
                   {"captured": text, "num_ctx": args.num_ctx})
    rep = validate(bundle)
    if args.out:
        corpus.write_json(args.out, bundle)
    else:
        sys.stdout.write(json.dumps(bundle, sort_keys=True, indent=1) + "\n")
    for err in rep["errors"]:
        sys.stderr.write("invalid: %s\n" % err)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
