"""The collector's command line (docs/CONTINUOUS_LOOP.md §6.3 L15, L16, L26; §7.2).

    python -m weed_optimizer_framework.tools.collect VERB [--config C] [--inc-dir D] [--result PATH] [--testing]

  plan     --out PATH [--classes A,B] [--providers p,q] [--resolve-names]   lever L15 (lab)
  fetch    --source ID [--max-bytes B] [--candidates PATH] [--out STAGING]  lever L16 (lab hook or job)
  intake   --source ID [--lock PATH] [--registry PATH] [--keep-work]        after L16 (job)
  names    --source ID [--out DIR] [--candidates PATH]                      lever L26 (lab)
  summary  [--extra-ledger PATH ...]                                       INC_DIR/intake/state.json
  probe    [--providers p,q]                                               the network probe (job)

--config defaults to collect/domains/<domain>.json for the one domain config
present (a path or a domain name may be given). --inc-dir overrides INC_DIR;
fetch --out <INC_DIR>/intake/staging/ (the autopilot's lab hook) names the
same directory from its staging root.

Output: one closing line "[collect] <verb>: <JSON>" on stdout, and the same
JSON (format collect-result/1) in --result when given (the autopilot's
detached lab processes read it). Exit codes: 0 done; 2 refused or held (a
CollectError: the JSON carries status "refused", "held" or "closed", the
code, the reasons and whether a person is asked, "risk": "R3"); 1 crash
(any other exception, with its traceback).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

from . import FORMATS, CollectError, DOMAINS_DIR, Refusal, utc, write_json_atomic

VERBS = ("plan", "fetch", "intake", "names", "summary", "probe")
JOB_VERBS = ("plan", "fetch", "intake", "summary", "probe")        # run_inc_collect.sh accepts these


class CLIError(CollectError):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise CLIError(message)


def build_parser():
    common = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    common.add_argument("--config", default=None)
    common.add_argument("--inc-dir", default=None)
    common.add_argument("--result", default=None)
    common.add_argument("--testing", action="store_true")
    p = _Parser(prog="python -m weed_optimizer_framework.tools.collect", allow_abbrev=False)
    sub = p.add_subparsers(dest="verb")
    sp = sub.add_parser("plan", parents=[common], allow_abbrev=False)
    sp.add_argument("--out", required=True)
    sp.add_argument("--classes", default=None)
    sp.add_argument("--providers", default=None)
    sp.add_argument("--resolve-names", action="store_true")
    sf = sub.add_parser("fetch", parents=[common], allow_abbrev=False)
    sf.add_argument("--source", required=True)
    sf.add_argument("--max-bytes", type=int, default=None)
    sf.add_argument("--candidates", default=None)
    sf.add_argument("--out", default=None, help="the staging root, <INC_DIR>/intake/staging/ (the lab hook's form)")
    si = sub.add_parser("intake", parents=[common], allow_abbrev=False)
    si.add_argument("--source", required=True)
    si.add_argument("--lock", default=None)
    si.add_argument("--registry", default=None)
    si.add_argument("--keep-work", action="store_true")
    sn = sub.add_parser("names", parents=[common], allow_abbrev=False)
    sn.add_argument("--source", required=True)
    sn.add_argument("--out", default=None)
    sn.add_argument("--candidates", default=None)
    ss = sub.add_parser("summary", parents=[common], allow_abbrev=False)
    ss.add_argument("--extra-ledger", action="append", default=[])
    sr = sub.add_parser("probe", parents=[common], allow_abbrev=False)
    sr.add_argument("--providers", default=None)
    return p


def default_config():
    found = sorted(DOMAINS_DIR.glob("*.json"))
    configs = []
    for f in found:
        try:
            with open(f) as fh:
                if json.load(fh).get("format") == FORMATS["domain"]:
                    configs.append(f)
        except (OSError, ValueError):
            continue
    if len(configs) != 1:
        raise CLIError("--config is required (%d collector configs in %s)" % (len(configs), DOMAINS_DIR))
    return str(configs[0])


def run(argv):
    a = build_parser().parse_args(argv)
    if not a.verb:
        raise CLIError("a verb is required: %s" % ", ".join(VERBS))
    from . import config as CF
    cfg = CF.load(a.config or default_config())
    inc = a.inc_dir
    if a.verb == "plan":
        from .plan import plan
        cls = [c.strip() for c in a.classes.split(",")] if a.classes else None
        prov = [p.strip() for p in a.providers.split(",")] if a.providers else None
        return plan(cfg, a.out, classes=cls, providers=prov, inc=inc, resolve_names=a.resolve_names,
                    testing=a.testing)
    if a.verb == "fetch":
        from .fetch import fetch
        if a.out:
            out = Path(a.out).resolve()
            if out.name != "staging" or out.parent.name != "intake":
                raise CLIError("fetch --out must be <INC_DIR>/intake/staging/ (got %s)" % a.out)
            if inc and Path(inc).resolve() != out.parent.parent:
                raise CLIError("fetch --out %s is not the staging root of --inc-dir %s" % (a.out, inc))
            inc = str(out.parent.parent)
        return fetch(cfg, a.source, max_bytes=a.max_bytes, candidates_path=a.candidates, inc=inc,
                     testing=a.testing)
    if a.verb == "intake":
        from .intake import intake
        return intake(cfg, a.source, inc=inc, testing=a.testing, registry_path=a.registry, lock_path=a.lock,
                      keep_work=a.keep_work)
    if a.verb == "names":
        from .fetch import load_candidates
        from .names import run_names
        return run_names(cfg, a.source, out_dir=a.out, inc=inc,
                         candidates=load_candidates(inc, a.candidates), testing=a.testing)
    if a.verb == "summary":
        from .probe import summary
        return summary(cfg, inc=inc, extra=a.extra_ledger, testing=a.testing)
    if a.verb == "probe":
        from .probe import probe
        prov = [p.strip() for p in a.providers.split(",")] if a.providers else None
        return probe(cfg, inc=inc, testing=a.testing, providers=prov)
    raise CLIError("unknown verb %r" % a.verb)


def _emit(verb, doc, result_path):
    doc = dict(doc, format=FORMATS["result"], verb=verb, utc=utc())
    print("[collect] %s: %s" % (verb, json.dumps(doc, sort_keys=True, default=str)), flush=True)
    if result_path:
        write_json_atomic(result_path, json.loads(json.dumps(doc, default=str)))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    verb = argv[0] if argv and not argv[0].startswith("-") else None
    result_path = None
    for i, x in enumerate(argv):
        if x == "--result" and i + 1 < len(argv):
            result_path = argv[i + 1]
        elif x.startswith("--result="):
            result_path = x.split("=", 1)[1]
    try:
        out = run(argv)
        _emit(verb, dict(out or {}, status=(out or {}).get("status", "done")), result_path)
        return 0
    except Refusal as e:
        st = {"hold": "held", "close": "closed", "refuse": "refused"}[e.action]
        _emit(verb, dict(e.record(), status=st, error=str(e)), result_path)
        print("[collect] %s refused: %s" % (verb, e), file=sys.stderr)
        return 2
    except CollectError as e:
        _emit(verb, {"status": "refused", "code": type(e).__name__, "error": str(e), "risk": None}, result_path)
        print("[collect] %s refused: %s" % (verb, e), file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 - a crash: traceback, exit 1
        traceback.print_exc()
        try:
            _emit(verb, {"status": "crashed", "error": "%s: %s" % (type(e).__name__, e)}, result_path)
        except Exception:  # noqa: BLE001
            pass
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(main())
