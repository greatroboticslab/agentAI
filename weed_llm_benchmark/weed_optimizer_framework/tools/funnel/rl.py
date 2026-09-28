"""The reference labeller: the platform-native backend's client, answer
files, their validation, and the gold table (contract docs/FUNNEL_AUDIT.md
§4.3, DEC-1, DEC-2; runner docs/FUNNEL_AUDIT_RUNNER.md §4.11, §5.4.2, §7.1).

    python -m <package>.tools.funnel rl-b --prereg PATH --sheet-dirs D ...   (F7c, cluster GPU job)
    python -m <package>.tools.funnel ingest --prereg PATH                     (F7d, cluster)

RL-B. One request per item to an ollama server (/api/chat, stream false,
temperature 0, a seed per item and attempt), with the images base64-encoded:
the board and the item's panel cut from the sheet image (a pair item sends its
panel only). The prompt is the config's text with its {options} filled; the
engine holds no prompt. The model tag must be the config's, its /api/show
capabilities must hold "vision" (else nothing is asked), and its digest from
/api/tags is recorded. A reply is parsed strictly (`parse_item`): the first
non-empty line after any <think> block must be "<option>, YES|NO|NA", or the
reply is one JSON object {"option": n, "box": "YES|NO|NA"}. An HTTP error or a
reply that does not parse is retried up to max_attempts times; after that an
item the model answered is recorded as `unparsed` with every raw reply kept.
An item whose every request failed (the server never answered) stops the run
with RLError before its sheet's file is written: a server failure is not a
labeller's answer, and resume skips sheets whose file exists. Nothing is
guessed.

RL-A and people answer whole sheets. `answers_from_text` turns a sheet's
reply ("<panel>: <option>, YES|NO|NA" per line, or a JSON list) into the same
answer file format.

Validation (`validate_answers`) refuses an answer file whose sheet sha256
differs from the sheet's JSON, that misses or doubles an item, holds an option
out of range, names another backend than its directory, is an RL-A file for
a sheet that shows evaluation pixels (DEC-2), or whose RL-A labeller is of a
judged model family.

Ingest joins every valid answer file with the key (whose sha256 both sheet
indexes recorded before any answer existed) and with the locked sample (its
disjointness keys; refused when missing or not the locked file) and writes
gold_v1.csv (one row per item and labeller) with gold_v1.json (header,
inputs, counts). An RL-B file must record the model digest.

Nothing here names a domain.
"""
from __future__ import annotations

import base64
import collections
import io
import json
import re
import sys
from pathlib import Path

from ..inc import common as C
from . import (FUNNEL_DIR, FunnelError, RLError, StaleInput, file_record, header, read_json,
               read_jsonl, utc, write_csv_atomic, write_json_atomic)
from . import domain as D
from . import qualify as Q

ANSWER_RE = r"^\s*(\d{1,2})\s*[,;]\s*(YES|NO|NA)\b"
SHEET_ANSWER_RE = r"^\s*(\d{1,2})\s*:\s*(\d{1,2})\s*[,;]\s*(YES|NO|NA)\b"
PAIR_RE = r"^\s*(\d)\b"
SHEET_PAIR_RE = r"^\s*(\d{1,2})\s*:\s*(\d)\b"
_ANSWER = re.compile(ANSWER_RE, re.I)
_SHEET_ANSWER = re.compile(SHEET_ANSWER_RE, re.I)
_PAIR = re.compile(PAIR_RE)
_SHEET_PAIR = re.compile(SHEET_PAIR_RE)
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)
FORMAT = "funnel-rl-answers/1"
BACKENDS = ("RL-A", "RL-B", "human")
STATUSES = ("ok", "unparsed")
ANSWER_KEYS = ("item_id", "option", "box_ok", "raw", "attempts", "status")
FILE_KEYS = ("format", "labeller", "backend", "sheet_id", "sheet_sha256", "board_sha256", "answered_utc",
             "model_digest", "answers")
GOLD_FIELDS = ("item_id", "unit_id", "unit", "group", "stratum", "labeller", "backend", "option", "answer",
               "answer_level", "answer_taxon", "box_ok", "is_sentinel", "kt", "truth", "truth_kind",
               "correct_species", "correct_genus", "correct_plant", "sheet_id", "position", "answers_sha256",
               "pair_truth", "truth_taxon", "source", "lab", "near_dup3", "provenance")
SHEETS_DIRS = ("sheets_v1", "sheets_v1_cluster")
EVAL_DIR = "sheets_v1_cluster"
KEY_PATH = Path("sheets_v1_key") / "key.jsonl"
UNIT_KINDS = {"b": "box", "i": "image", "d": "image", "c": "class", "k": "cluster", "s": "source",
              "p": "pair", "t1": "box", "t2": "box", "t7": "photo", "G0": "box"}
BOX_OK = {"YES": "yes", "NO": "no", "NA": "na"}


def log(msg):
    print("[funnel.rl] %s" % msg, flush=True)


# ------------------------------------------------------------------ parsing
def _clean(text):
    body = _THINK.sub("", str(text or "")).strip()
    m = _FENCE.match(body)
    return m.group(1).strip() if m else body


def _first_line(body):
    for ln in body.splitlines():
        if ln.strip():
            return ln
    return ""


def _json_obj(body):
    if not body.startswith(("{", "[")):
        return None
    try:
        return json.loads(body)
    except ValueError:
        return None


def _box(v):
    if isinstance(v, str) and v.strip().upper() in BOX_OK:
        return BOX_OK[v.strip().upper()]
    return None


def _option(v, n_options):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        o = v
    elif isinstance(v, str) and v.strip().isdigit():
        o = int(v.strip())
    else:
        return None
    return o if 1 <= o <= int(n_options) else None


def parse_item(text, n_options):
    """(option, box_ok) of a one-item reply, or None. The reply (after any
    <think> block) is a first line "<option>, YES|NO|NA" or one JSON object
    {"option": n, "box": "YES|NO|NA"}; the option must be 1..n_options."""
    body = _clean(text)
    obj = _json_obj(body)
    if isinstance(obj, dict):
        o = _option(obj.get("option"), n_options)
        b = _box(obj.get("box", obj.get("box_ok")))
        return (o, b) if o is not None and b is not None else None
    m = _ANSWER.match(_first_line(body))
    if not m:
        return None
    o = _option(m.group(1), n_options)
    return (o, BOX_OK[m.group(2).upper()]) if o is not None else None


def parse_pair(text, n_options):
    """The option of a pair reply (first line "<option>" or {"option": n})."""
    body = _clean(text)
    obj = _json_obj(body)
    if isinstance(obj, dict):
        return _option(obj.get("option"), n_options)
    m = _PAIR.match(_first_line(body))
    return _option(m.group(1), n_options) if m else None


def parse_sheet(text, positions, n_options, pair=False):
    """{position: (option, box_ok)} of a whole-sheet reply ("<panel>:
    <option>, YES|NO|NA" per line, or a JSON list of {"panel", "option",
    "box"}); for a pair sheet {position: (option, None)}. A position given
    twice with different answers, or outside `positions`, is left out."""
    body = _clean(text)
    positions = set(int(p) for p in positions)
    got = collections.defaultdict(set)
    obj = _json_obj(body)
    if isinstance(obj, dict) and isinstance(obj.get("answers"), list):
        obj = obj["answers"]
    if isinstance(obj, list):
        for e in obj:
            if not isinstance(e, dict):
                continue
            p = _option(e.get("panel", e.get("position")), 99)
            o = _option(e.get("option"), n_options)
            b = None if pair else _box(e.get("box", e.get("box_ok")))
            if p is not None and o is not None and (pair or b is not None):
                got[p].add((o, b))
    else:
        rx = _SHEET_PAIR if pair else _SHEET_ANSWER
        for ln in body.splitlines():
            m = rx.match(ln)
            if not m:
                continue
            p = int(m.group(1))
            o = _option(m.group(2), n_options)
            if o is None:
                continue
            got[p].add((o, None if pair else BOX_OK[m.group(3).upper()]))
    return {p: next(iter(v)) for p, v in sorted(got.items()) if p in positions and len(v) == 1}


# ------------------------------------------------------------------ client
def _urllib_transport(method, url, body, timeout):
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, data=body, method=method,
                                 headers={"Content-Type": "application/json"} if body is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read() if hasattr(e, "read") else b""


class OllamaClient(object):
    """POST /api/chat with base64 images, stream false. transport(method,
    url, body bytes | None, timeout) -> (status, body bytes) is injected in
    tests; the default uses urllib."""

    def __init__(self, endpoint, model, transport=None, num_ctx=8192, timeout_s=900):
        if not endpoint:
            raise RLError("no ollama endpoint (set --endpoint or FUNNEL_OLLAMA_ENDPOINT)")
        self.endpoint = str(endpoint).rstrip("/")
        if not self.endpoint.startswith(("http://", "https://")):
            self.endpoint = "http://" + self.endpoint
        self.model = str(model)
        self.transport = transport or _urllib_transport
        self.num_ctx = int(num_ctx)
        self.timeout_s = float(timeout_s)

    def _call(self, method, path, payload=None):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        try:
            status, data = self.transport(method, self.endpoint + path, body, self.timeout_s)
        except Exception as e:
            raise RLError("%s %s failed: %s: %s" % (method, path, type(e).__name__, e))
        if int(status) != 200:
            raise RLError("%s %s returned HTTP %s: %s" % (method, path, status, (data or b"")[:200]))
        try:
            return json.loads((data or b"").decode("utf-8"))
        except ValueError as e:
            raise RLError("%s %s returned non-JSON (%s)" % (method, path, e))

    def check_vision(self):
        """RLError unless /api/show lists "vision" among the model's capabilities."""
        info = self._call("POST", "/api/show", {"model": self.model, "name": self.model})
        caps = info.get("capabilities")
        if not isinstance(caps, list):
            raise RLError("/api/show gives no capabilities for %s: vision cannot be confirmed" % self.model)
        if "vision" not in caps:
            raise RLError("model %s has no vision capability (%s)" % (self.model, caps))
        return caps

    def model_digest(self):
        tags = self._call("GET", "/api/tags")
        for m in tags.get("models") or []:
            if self.model in (m.get("name"), m.get("model")):
                if not m.get("digest"):
                    raise RLError("/api/tags lists %s without a digest" % self.model)
                return m["digest"]
        raise RLError("model %s is not in the ollama store (/api/tags)" % self.model)

    def ask(self, prompt, images, seed):
        payload = {"model": self.model, "stream": False,
                   "messages": [{"role": "user", "content": prompt,
                                 "images": [base64.b64encode(b).decode("ascii") for b in images]}],
                   "options": {"temperature": 0, "seed": int(seed), "num_ctx": self.num_ctx}}
        out = self._call("POST", "/api/chat", payload)
        msg = out.get("message") or {}
        content = msg.get("content")
        if not isinstance(content, str):
            raise RLError("/api/chat reply has no message content")
        return content


# ------------------------------------------------------------------ sheets
def _sheet_dirs(funnel_dir, sheet_dirs=None):
    dirs = [Path(d) for d in (sheet_dirs or [Path(funnel_dir) / d for d in SHEETS_DIRS])]
    return [d for d in dirs if (d / "index.json").is_file()]


def load_sheets(sheet_dirs):
    """{sheet_id: {"path", "doc", "dir", "index", "board_path"}} with every
    sheet JSON and image checked against its index."""
    out = {}
    for d in sheet_dirs:
        d = Path(d)
        idx = read_json(d / "index.json")
        for e in idx.get("sheets", []):
            p = d / ("%s.json" % e["sheet_id"])
            if file_record(p)["sha256"] != e["json_sha256"]:
                raise StaleInput("sheet %s does not hash to its index record" % p)
            doc = read_json(p)
            img = d / doc["image"]["file"]
            if file_record(img)["sha256"] != doc["image"]["sha256"]:
                raise StaleInput("sheet image %s does not hash to its sheet record" % img)
            board = None
            if doc.get("board"):
                board = d / doc["board"]["file"]
                if file_record(board)["sha256"] != doc["board"]["sha256"]:
                    raise StaleInput("board %s does not hash to its sheet record" % board)
            if e["sheet_id"] in out:
                raise RLError("sheet id %s is in two sheet directories" % e["sheet_id"])
            out[e["sheet_id"]] = {"path": p, "doc": doc, "dir": d, "index": idx, "board_path": board,
                                  "image_path": img}
    return out


def panel_png(image_path, rect):
    """The panel [x, y, w, h] cut from a sheet image, as PNG bytes."""
    from PIL import Image
    x, y, w, h = [int(v) for v in rect]
    with Image.open(image_path) as im:
        panel = im.convert("RGB").crop((x, y, x + w, y + h))
    buf = io.BytesIO()
    panel.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _options_of(domain, kind):
    return domain.pair_options() if kind == "pair" else domain.options()


def _prompt(domain, kind):
    from .sheets import fill_prompt
    rl = domain.raw.get("reference_labeller") or {}
    key = "pair_prompt" if kind == "pair" else "prompt"
    if not rl.get(key):
        raise RLError("reference_labeller.%s is not set" % key)
    return fill_prompt(rl[key], _options_of(domain, kind))


def run_rl_b(prereg, domain, funnel_dir=None, client=None, sheet_dirs=None, max_attempts=3, testing=False,
             endpoint=None):
    """rl_answers/RL-B/<sheet_id>.json for every sheet (resumable per sheet)."""
    prereg, domain = Q._load(prereg, domain)
    funnel_dir = Path(funnel_dir or FUNNEL_DIR)
    be = ((domain.raw.get("reference_labeller") or {}).get("backends") or {}).get("RL-B") or {}
    model = be.get("model")
    if not model:
        raise RLError("reference_labeller.backends.RL-B.model is not set")
    if client is None:
        client = OllamaClient(endpoint, model)
    if client.model != model:
        raise RLError("the client asks %s; the config's RL-B model is %s" % (client.model, model))
    client.check_vision()
    digest = client.model_digest()
    sheets = load_sheets(_sheet_dirs(funnel_dir, sheet_dirs))
    if not sheets:
        raise RLError("no sheets found (run sheets first)")
    out_dir = funnel_dir / "rl_answers" / "RL-B"
    labeller = "RL-B:%s" % model
    done = skipped = 0
    counts = collections.Counter()
    for sid in sorted(sheets):
        sh = sheets[sid]
        doc = sh["doc"]
        path = out_dir / ("%s.json" % sid)
        sheet_sha = file_record(sh["path"])["sha256"]
        if path.is_file():
            prev = read_json(path)
            if (prev.get("sheet_sha256") == sheet_sha and prev.get("model_digest") == digest
                    and not validate_answers(path, sh["path"], "RL-B", domain=domain)):
                skipped += 1
                for a in prev["answers"]:
                    counts[a["status"]] += 1
                continue
        kind = doc["kind"]
        options = _options_of(domain, kind)
        prompt = _prompt(domain, kind)
        board_bytes = sh["board_path"].read_bytes() if sh["board_path"] is not None else None
        answers = []
        for it in sorted(doc["items"], key=lambda x: x["position"]):
            images = ([board_bytes] if (kind != "pair" and board_bytes is not None) else [])
            images.append(panel_png(sh["image_path"], it["panel"]))
            raws, parsed, attempt, replies = [], None, 0, 0
            for attempt in range(1, int(max_attempts) + 1):
                seed = C.stable_int("funnel/v1/rl-b/" + it["item_id"]) + attempt - 1
                try:
                    text = client.ask(prompt, images, seed)
                except RLError as e:
                    raws.append("error: %s" % e)
                    continue
                replies += 1
                raws.append(text)
                parsed = (parse_pair(text, len(options)) if kind == "pair" else parse_item(text, len(options)))
                if parsed is not None:
                    break
            if parsed is None and replies == 0:
                # the server never answered: that is not the labeller's answer, and an answer file written
                # now would be skipped on resume. Stop; the sheets answered so far are kept.
                raise RLError("item %s of %s: every one of %d request(s) failed (%s); no answer file is written "
                              "for this sheet, rerun rl-b to resume" % (it["item_id"], sid, attempt,
                                                                        raws[-1] if raws else "no reply"))
            if parsed is None:
                rec = {"item_id": it["item_id"], "option": None, "box_ok": None, "raw": raws[-1] if raws else "",
                       "attempts": attempt, "status": "unparsed"}
            elif kind == "pair":
                rec = {"item_id": it["item_id"], "option": parsed, "box_ok": None, "raw": raws[-1],
                       "attempts": attempt, "status": "ok"}
            else:
                rec = {"item_id": it["item_id"], "option": parsed[0], "box_ok": parsed[1], "raw": raws[-1],
                       "attempts": attempt, "status": "ok"}
            if len(raws) > 1:
                rec["raw_history"] = raws[:-1]
            counts[rec["status"]] += 1
            answers.append(rec)
        write_json_atomic(path, {"format": FORMAT, "labeller": labeller, "backend": "RL-B", "sheet_id": sid,
                                 "sheet_sha256": sheet_sha,
                                 "board_sha256": doc["board"]["sha256"] if doc.get("board") else None,
                                 "answered_utc": utc(), "model_digest": digest, "answers": answers})
        done += 1
    run_doc = dict(header("funnel-rl-run/1", domain, prereg,
                          {("sheet:%s" % s): sheets[s]["path"] for s in sorted(sheets)},
                          modules=(sys.modules[__name__],), testing=testing),
                   backend="RL-B", model=model, model_digest=digest, max_attempts=int(max_attempts),
                   sheets_answered=done, sheets_skipped=skipped, answers=dict(counts))
    write_json_atomic(out_dir / "run.json", run_doc)
    log("rl-b: %d sheet(s) answered, %d already done; answers %s" % (done, skipped, dict(counts)))
    return run_doc


def answers_from_text(sheet_path, text, backend, labeller, out_dir=None, domain=None, answered_utc=None):
    """An answer file for a whole sheet answered in one reply (RL-A, a
    person): each position parsed with parse_sheet; a position the reply
    does not answer (or answers twice) is recorded as unparsed."""
    sheet_path = Path(sheet_path)
    doc = read_json(sheet_path)
    if backend not in BACKENDS:
        raise RLError("backend %r is not one of %s" % (backend, BACKENDS))
    if domain is None:
        raise RLError("answers_from_text needs the domain config (its option lists)")
    n_opt = len(_options_of(domain, doc["kind"]))
    pos = [it["position"] for it in doc["items"]]
    got = parse_sheet(text, pos, n_opt, pair=doc["kind"] == "pair")
    answers = []
    for it in sorted(doc["items"], key=lambda x: x["position"]):
        a = got.get(it["position"])
        answers.append({"item_id": it["item_id"], "option": a[0] if a else None, "box_ok": a[1] if a else None,
                        "raw": str(text), "attempts": 1, "status": "ok" if a else "unparsed"})
    rec = {"format": FORMAT, "labeller": labeller, "backend": backend, "sheet_id": doc["sheet_id"],
           "sheet_sha256": file_record(sheet_path)["sha256"],
           "board_sha256": doc["board"]["sha256"] if doc.get("board") else None,
           "answered_utc": answered_utc or utc(), "model_digest": None, "answers": answers}
    if out_dir is not None:
        write_json_atomic(Path(out_dir) / ("%s.json" % doc["sheet_id"]), rec)
    return rec


# --------------------------------------------------------------- validation
def _family_of(model, domain):
    for be in ((domain.raw.get("reference_labeller") or {}).get("backends") or {}).values():
        if isinstance(be, dict) and be.get("model") == model:
            return be.get("family")
    return None


def validate_answers(path, sheet_json, backend, domain=None):
    """Every reason the answer file at `path` is refused for the sheet whose
    JSON is at `sheet_json`, answered by `backend` (the directory's)."""
    path, sheet_json = Path(path), Path(sheet_json)
    probs = []
    try:
        rec = read_json(path)
    except FunnelError as e:
        return [str(e)]
    try:
        sheet = read_json(sheet_json)
    except FunnelError as e:
        return ["the sheet: %s" % e]
    if rec.get("format") != FORMAT:
        probs.append("format %r is not %s" % (rec.get("format"), FORMAT))
    missing = [k for k in FILE_KEYS if k not in rec]
    if missing:
        probs.append("missing keys %s" % missing)
    if path.parent.name != backend:
        probs.append("the file lies in %s/, not in the %s directory" % (path.parent.name, backend))
    if rec.get("backend") != backend:
        probs.append("backend %r is not the directory's %r" % (rec.get("backend"), backend))
    if rec.get("sheet_id") != sheet.get("sheet_id") or path.stem != sheet.get("sheet_id"):
        probs.append("sheet id %r / file %s do not name sheet %s" % (rec.get("sheet_id"), path.name,
                                                                     sheet.get("sheet_id")))
    if rec.get("sheet_sha256") != file_record(sheet_json)["sha256"]:
        probs.append("sheet_sha256 differs from the sheet's JSON")
    want_board = sheet["board"]["sha256"] if sheet.get("board") else None
    if rec.get("board_sha256") != want_board:
        probs.append("board_sha256 differs from the sheet's board")
    lab = str(rec.get("labeller") or "")
    prefix = "human:" if backend == "human" else "%s:" % backend
    if not lab.startswith(prefix) or not lab[len(prefix):]:
        probs.append("labeller %r is not %s<name>" % (lab, prefix))
    eval_sheet = sheet_json.parent.name == EVAL_DIR or sheet.get("kind") == "pair"
    idx_p = sheet_json.parent / "index.json"
    if idx_p.is_file():
        try:
            eval_sheet = eval_sheet or bool(read_json(idx_p).get("contains_eval_pixels"))
        except FunnelError:
            probs.append("the sheet directory's index.json is unreadable")
    if backend != "RL-B" and eval_sheet:
        probs.append("%s answers a sheet that shows evaluation pixels (DEC-2: RL-B only)" % backend)
    if backend in ("RL-A", "RL-B"):
        if domain is None:
            probs.append("no domain config to check the %s labeller against" % backend)
        else:
            be = ((domain.raw.get("reference_labeller") or {}).get("backends") or {}).get(backend) or {}
            model = lab[len(prefix):]
            if model != be.get("model"):
                probs.append("%s model %r is not the configured %r" % (backend, model, be.get("model")))
            if backend == "RL-B" and not (isinstance(rec.get("model_digest"), str) and rec.get("model_digest")):
                probs.append("an RL-B file records no model digest (/api/tags)")
            if backend == "RL-A":
                fam = _family_of(model, domain) or ""
                judged = [f.lower() for f in (domain.raw.get("reference_labeller") or {}).get("judged_families", [])]
                if fam.lower() in judged or any(f and f in model.lower() for f in judged):
                    probs.append("RL-A labeller %r is of a judged model family (%s)" % (model, fam or "by name"))
    n_opt = None
    if domain is not None:
        n_opt = len(_options_of(domain, sheet.get("kind")))
    want_ids = [it["item_id"] for it in sheet.get("items", [])]
    got_ids = [a.get("item_id") for a in rec.get("answers") or []]
    dup = sorted(i for i, c in collections.Counter(got_ids).items() if c > 1)
    if dup:
        probs.append("items answered more than once: %s" % dup[:5])
    miss = sorted(set(want_ids) - set(got_ids))
    if miss:
        probs.append("items not answered: %s" % miss[:5])
    extra = sorted(set(got_ids) - set(want_ids))
    if extra:
        probs.append("answers for items not on the sheet: %s" % extra[:5])
    for a in rec.get("answers") or []:
        st = a.get("status")
        if st not in STATUSES:
            probs.append("item %s: status %r" % (a.get("item_id"), st))
            continue
        opt = a.get("option")
        if st == "ok":
            if not isinstance(opt, int) or isinstance(opt, bool) or opt < 1 or (n_opt is not None and opt > n_opt):
                probs.append("item %s: option %r out of range 1..%s" % (a.get("item_id"), opt, n_opt))
            if sheet.get("kind") != "pair" and a.get("box_ok") not in ("yes", "no", "na"):
                probs.append("item %s: box_ok %r" % (a.get("item_id"), a.get("box_ok")))
        elif opt is not None:
            probs.append("item %s: an unparsed answer carries option %r" % (a.get("item_id"), opt))
    return probs


# ------------------------------------------------------------------ ingest
def _unit_kind(unit_id):
    pre = str(unit_id or "").split(":", 1)[0]
    return UNIT_KINDS.get(pre, "")


def _answer_fields(option, kind, options, pair_options):
    """(answer, answer_level, answer_taxon) of an option number."""
    if option is None:
        return "unparsed", "none", ""
    if kind == "pair":
        o = pair_options[int(option) - 1]
        return o["answer"], "none", ""
    o = options[int(option) - 1]
    if o["kind"] == "tail":
        return o["answer"], "none", ""
    return o["answer"], o["rank"] or "species", o["taxon"] or ""


def _correct_cols(answer, answer_taxon, answer_level, key_row, domain):
    out = {}
    for lv in D.LEVELS:
        t = key_row.get("truth")
        tl = Q.truth_label(None if t in (None, "") else int(t), key_row.get("truth_kind"),
                           key_row.get("truth_taxon"), lv, domain)
        al = Q.answer_label(answer, answer_taxon, answer_level, lv, domain)
        out["correct_%s" % lv] = int(tl is not None and al is not None and al == tl)
    return out


def ingest(prereg, domain, funnel_dir=None, answer_dirs=None, testing=False):
    """gold_v1.csv and gold_v1.json from every answer file (all must validate)."""
    prereg, domain = Q._load(prereg, domain)
    funnel_dir = Path(funnel_dir or FUNNEL_DIR)
    key_path = funnel_dir / KEY_PATH
    key_rec = file_record(key_path)
    sheet_dirs = _sheet_dirs(funnel_dir)
    if not sheet_dirs:
        raise RLError("no sheet index under %s" % funnel_dir)
    lock = prereg.sample_lock or {}
    for d in sheet_dirs:
        idx = read_json(d / "index.json")
        if idx.get("key_sha256") != key_rec["sha256"]:
            raise StaleInput("%s records key %s; the key hashes to %s" % (d / "index.json",
                                                                          str(idx.get("key_sha256"))[:12],
                                                                          key_rec["sha256"][:12]))
        if lock.get("sample_sha256") and idx.get("sample_sha256") != lock["sample_sha256"]:
            raise StaleInput("%s was made from another sample than the lock's" % (d / "index.json"))
    sheets = load_sheets(sheet_dirs)
    key = {r["item_id"]: r for r in read_jsonl(key_path)}
    # the sample carries every item's disjointness keys, which the labeller's scopes are computed from:
    # without it (or with another sample) a scope would silently ignore near-duplicate and provenance sharing
    sp = funnel_dir / "sample_v1.csv"
    if not sp.is_file():
        raise RLError("%s is missing: gold rows need the sample's disjointness keys" % sp)
    sample_rec = file_record(sp)
    for d in sheet_dirs:
        want = read_json(d / "index.json").get("sample_sha256")
        if sample_rec["sha256"] != want or (lock.get("sample_sha256") and sample_rec["sha256"] != lock["sample_sha256"]):
            raise StaleInput("%s does not hash to the sample the sheets (and the sample lock) were made from" % sp)
    from . import read_csv
    _h, srows = read_csv(sp)
    sample = {r["item_id"]: r for r in srows}
    inputs = {"key": key_rec, "sample": sample_rec}
    for d in sheet_dirs:
        inputs["index:%s" % d.name] = file_record(d / "index.json")
    if answer_dirs is None:
        base = funnel_dir / "rl_answers"
        answer_dirs = sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
    options, pair_options = domain.options(), domain.pair_options()
    problems, rows, not_in_key = [], [], []
    files = 0
    per = collections.Counter()
    for d in [Path(x) for x in answer_dirs]:
        backend = d.name
        if backend not in BACKENDS:
            problems.append("%s: directory name is not a backend (%s)" % (d, BACKENDS))
            continue
        for p in sorted(d.glob("*.json")):
            if p.name == "run.json":
                continue
            rec = read_json(p) if p.is_file() else {}
            sid = rec.get("sheet_id") or p.stem
            sh = sheets.get(sid)
            if sh is None:
                problems.append("%s: sheet %s is in no sheet index" % (p, sid))
                continue
            probs = validate_answers(p, sh["path"], backend, domain=domain)
            if probs:
                problems.extend("%s: %s" % (p, x) for x in probs)
                continue
            files += 1
            fsha = file_record(p)["sha256"]
            inputs["answers:%s/%s" % (backend, p.name)] = file_record(p)
            pos_of = {it["item_id"]: it["position"] for it in sh["doc"]["items"]}
            for a in rec["answers"]:
                k = key.get(a["item_id"])
                if k is None:
                    not_in_key.append(a["item_id"])
                    continue
                s = sample.get(a["item_id"], {})
                opt = a.get("option") if a.get("status") == "ok" else None
                ans, lvl, tax = _answer_fields(opt, sh["doc"]["kind"], options, pair_options)
                qual = bool(k.get("is_sentinel")) or k.get("group") == Q.IDENTITY_GROUP
                corr = (_correct_cols(ans, tax, lvl, k, domain) if qual and sh["doc"]["kind"] != "pair"
                        else {"correct_%s" % lv: "" for lv in D.LEVELS})
                rows.append({"item_id": a["item_id"], "unit_id": k.get("unit_id"),
                             "unit": s.get("unit") or _unit_kind(k.get("unit_id")), "group": k.get("group"),
                             "stratum": k.get("stratum"), "labeller": rec["labeller"], "backend": backend,
                             "option": "" if opt is None else opt, "answer": ans, "answer_level": lvl,
                             "answer_taxon": tax, "box_ok": a.get("box_ok") or "",
                             "is_sentinel": int(bool(k.get("is_sentinel"))), "kt": k.get("kt") or "",
                             "truth": "" if k.get("truth") is None else k.get("truth"),
                             "truth_kind": k.get("truth_kind") or "",
                             "sheet_id": sid, "position": pos_of[a["item_id"]], "answers_sha256": fsha,
                             "pair_truth": k.get("pair_truth") or "", "truth_taxon": k.get("truth_taxon") or "",
                             "source": s.get("source", ""), "lab": s.get("lab", ""),
                             "near_dup3": s.get("near_dup3", ""), "provenance": s.get("provenance", ""),
                             **corr})
                per[(backend, k.get("group"))] += 1
    if problems:
        raise RLError("%d answer problem(s); nothing ingested: %s" % (len(problems), "; ".join(problems[:8])))
    rows.sort(key=lambda r: (r["item_id"], r["labeller"]))
    dup = [k for k, c in collections.Counter((r["item_id"], r["labeller"]) for r in rows).items() if c > 1]
    if dup:
        raise RLError("an item is answered twice by one labeller: %s" % dup[:3])
    gold_path = funnel_dir / "gold_v1.csv"
    gold_sha = write_csv_atomic(gold_path, GOLD_FIELDS, rows)
    doc = dict(header("funnel-gold/1", domain, prereg, inputs, modules=(sys.modules[__name__],),
                      testing=testing),
               gold_sha256=gold_sha, rows=len(rows), answer_files=files,
               by_backend_group={"%s|%s" % k: v for k, v in sorted(per.items())},
               not_in_key=sorted(set(not_in_key)), key_sha256=key_rec["sha256"],
               unanswered=sorted(set(key) - {r["item_id"] for r in rows})[:200],
               n_unanswered=len(set(key) - {r["item_id"] for r in rows}))
    write_json_atomic(funnel_dir / "gold_v1.json", doc)
    log("ingest: %d gold row(s) from %d answer file(s); %d item(s) unanswered"
        % (len(rows), files, doc["n_unanswered"]))
    return doc
