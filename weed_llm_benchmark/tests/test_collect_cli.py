#!/usr/bin/env python3
"""The collector's command line, the network probe and the state summary
(docs/CONTINUOUS_LOOP.md §6.2 "detached lab processes with a result file",
§6.6 stop-losses, §7.2 placement).

Pinned:
  * exit codes: 0 done; 2 refused or held (the closing JSON says status
    refused/held/closed, the code, every failure and whether a person is asked);
    1 a crash (traceback, status crashed); an unknown verb or option is 2;
  * one closing line "[collect] <verb>: <JSON>" on stdout, and the same JSON
    (format collect-result/1) in --result;
  * the probe: outside Slurm it writes placement_lab.json and never the
    cluster's placement.json; inside a Slurm job it writes placement.json,
    placing a provider on the cluster only when it answered and the config
    does not keep it on the lab (GitHub stays on the lab), an FTP server is
    tried with its login, and an unreachable provider keeps the lab hook;
  * the summary: intake/state.json with every source's folded state, bytes
    against the caps, the ledgers' chain check, and the recent outcomes in
    time order (yield, zero with its decision reasons, failed) that the DATA
    lane's three-in-a-row stop-loss reads.

Run:  python3 tests/test_collect_cli.py
"""
import io
import json
import os
import pathlib
import sys
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_collect_world as W  # noqa: E402

TMP = W.setup("collect_cli_")
check, raises = W.check, W.raises


def run(argv):
    from weed_optimizer_framework.tools.collect import __main__ as CLI
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = CLI.main(argv)
    lines = [ln for ln in out.getvalue().splitlines() if ln.startswith("[collect] ")]
    doc = json.loads(lines[-1].split(": ", 1)[1]) if lines else None
    return rc, doc, err.getvalue()


def test_cli():
    print("the command line")
    res = TMP / "r.json"
    rc, doc, err = run(["fetch", "--source", "nosuch", "--result", str(res)])
    check("an unknown source: exit 2, status refused, the code", rc == 2 and doc["status"] == "refused"
          and doc["code"] == "unknown_source", (rc, doc))
    check("--result holds the same JSON (collect-result/1)", json.loads(res.read_text())["code"] == "unknown_source"
          and json.loads(res.read_text())["format"] == "collect-result/1")
    rc, doc, err = run(["fetch", "--source", "mh_weed16"])
    check("another campaign's source: exit 2, status held, every failure listed",
          rc == 2 and doc["status"] == "held" and [f["code"] for f in doc["failures"]] == ["owned_by_funnel"], doc)
    rc, doc, err = run(["fetch", "--source", "nosuch", "--out", str(TMP / "elsewhere")])
    check("fetch --out must be <INC_DIR>/intake/staging/", rc == 2 and "intake/staging" in doc["error"], doc)
    rc, doc, err = run(["fetch", "--source", "nosuch", "--out", str(TMP / "labinc" / "intake" / "staging") + "/"])
    check("... and names the INC_DIR (the autopilot's lab hook form)", rc == 2 and doc["code"] == "unknown_source"
          and (TMP / "labinc" / "intake" / ".lock").exists(), doc)
    rc, doc, err = run(["summary"])
    check("summary: exit 0, status summarised", rc == 0 and doc["status"] == "summarised" and doc["verb"] == "summary")
    rc, doc, err = run(["nosuchverb"])
    check("an unknown verb: exit 2", rc == 2, (rc, doc))
    rc, doc, err = run(["fetch"])
    check("a missing required option: exit 2", rc == 2 and "source" in (doc or {}).get("error", ""), doc)
    from weed_optimizer_framework.tools.collect import plan as PL
    real = PL.plan

    def boom(*a, **k):
        raise ValueError("planted crash")
    PL.plan = boom
    try:
        rc, doc, err = run(["plan", "--out", str(TMP / "c.json")])
    finally:
        PL.plan = real
    check("a crash: exit 1, a traceback, status crashed", rc == 1 and doc["status"] == "crashed"
          and "Traceback" in err and "planted crash" in doc["error"], (rc, doc))


def test_probe(cfg):
    from weed_optimizer_framework.tools.collect import intake_dir
    from weed_optimizer_framework.tools.collect import probe as PR
    print("the network probe")
    ok_urls = {"https://zenodo.org/api/records?size=1", cfg.provider("weedai")["base_url"] + "/api/set_csrf/",
               "https://api.github.com/rate_limit"}
    routes = {("GET", u): (200, {}) for u in ok_urls}

    def down(p, d, h):
        raise OSError("network is unreachable")
    routes[("GET", "https://www.kaggle.com/api/v1/datasets/list?search=probe&page=1")] = down
    host = cfg.provider("mediatum")["ftp"]["host"]
    net = W.make_net(routes, ftp={host: W.FakeFtp({"x": b"1"})})
    r = PR.probe(cfg, net=net, testing=True)
    check("outside Slurm: placement_lab.json, never the cluster's placement.json",
          (intake_dir() / "placement_lab.json").is_file() and not (intake_dir() / "placement.json").exists(), r)
    os.environ["SLURM_JOB_ID"] = "777"
    try:
        r = PR.probe(cfg, net=net, testing=True)
    finally:
        os.environ.pop("SLURM_JOB_ID", None)
    doc = json.loads((intake_dir() / "placement.json").read_text())
    pv = doc["providers"]
    check("inside Slurm: placement.json with the job id", doc["in_slurm"] and doc["slurm_job_id"] == "777")
    check("a provider that answered is placed on the cluster", pv["zenodo"]["placement"] == "cluster"
          and pv["weedai"]["placement"] == "cluster", pv["zenodo"])
    check("a lab-only provider stays on the lab although it answered", pv["github"]["reachable"]
          and pv["github"]["placement"] == "lab")
    check("an unreachable provider keeps the lab hook", pv["kaggle"]["placement"] == "lab"
          and not pv["kaggle"]["reachable"])
    check("an FTP server is tried with its login", pv["mediatum"]["ftp"]["reachable"] is True, pv["mediatum"])


def test_summary(cfg):
    from weed_optimizer_framework.tools.collect import intake_dir, state as S
    from weed_optimizer_framework.tools.collect import probe as PR
    print("the state summary")
    S.append(None, "src_a", "fetch_started", ts="2026-09-28T10:00:00Z")
    S.append(None, "src_a", "fetched", bytes=100, ts="2026-09-28T10:01:00Z")
    S.append(None, "src_a", "intaken", batch="i0001_src_a", ts="2026-09-28T10:02:00Z",
             **{"yield": {"target_boxes": 0, "rejected": {"no_box": 3}}})
    S.append(None, "src_b", "fetch_started", ts="2026-09-28T11:00:00Z")
    S.append(None, "src_b", "fetch_failed", reason="download_failed", ts="2026-09-28T11:01:00Z")
    S.append(None, "src_c", "fetch_started", ts="2026-09-28T12:00:00Z")
    S.append(None, "src_c", "fetched", bytes=50, ts="2026-09-28T12:01:00Z")
    S.append(None, "src_c", "intaken", batch="i0002_src_c", ts="2026-09-28T12:02:00Z",
             **{"yield": {"target_boxes": 7, "rejected": {}}})
    import datetime
    r = PR.summary(cfg, testing=True, now=datetime.datetime(2026, 9, 28, 20, tzinfo=datetime.timezone.utc))
    doc = json.loads((intake_dir() / "state.json").read_text())
    out = [(o["source"], o["outcome"]) for o in doc["recent_outcomes"]]
    check("recent outcomes in time order: zero, failed, yield", out[-3:] == [("src_a", "zero"), ("src_b", "failed"),
                                                                            ("src_c", "yield")], out)
    check("a zero outcome lists its decision reasons", [o for o in doc["recent_outcomes"] if o["source"] ==
                                                        "src_a"][0]["reasons"] == {"no_box": 3})
    check("bytes in the last 24 h and in total, against the caps", doc["bytes"]["last_24h"] == 150
          and doc["bytes"]["total"] == 150 and doc["bytes"]["daily_cap"] == cfg.raw["budgets"]["bytes_daily"])
    check("the folded state of every source", doc["sources"]["src_b"]["failed_attempts"] == 1
          and doc["sources"]["src_c"]["status"] == "intaken" and doc["sources"]["src_a"]["batches"] == ["i0001_src_a"])
    check("the ledgers' chains verify; the yield floors are unset until a person sets them",
          r["chain_problems"] == 0 and doc["floors"] == {"floor_gb": None, "floor_su": None})
    led = intake_dir() / "sources.jsonl"
    lines = led.read_text().splitlines()
    i = [n for n, ln in enumerate(lines) if '"bytes":100' in ln][0]
    lines[i] = lines[i].replace('"bytes":100', '"bytes":999')
    led.write_text("\n".join(lines) + "\n")
    r = PR.summary(cfg, testing=True)
    check("an edited ledger line breaks the chain and the summary says so", r["chain_problems"] >= 1, r)


def main():
    try:
        cfg = W.config()
        W.build_cache()
        test_cli()
        test_probe(cfg)
        test_summary(cfg)
    finally:
        W.cleanup()
    print("\n%d failure(s)" % len(W.FAILURES))
    return 1 if W.FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
