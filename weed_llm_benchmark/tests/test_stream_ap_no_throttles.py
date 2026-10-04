#!/usr/bin/env python3
"""No time-based compute throttles by default (docs/CONTINUOUS_LOOP.md 6.6,
amendment 2026-10-04, decided by the owner).

Live, 2026-10-04: the cluster sat idle about 9 h while the stream's next
segment (L18) was filed "awaiting approval" for two reasons only: "estimated
116.2 SU exceeds the 86.52 SU left under today's cap of 120" and "L18 already
ran 1 time(s) in the last 24 h (limit 1)". The campaign had daily_cap_su 180,
silently capped at db.DEFAULT_DOMAIN_CONFIG's code default budget.daily_cap
120. The amendment removes every default daily cap, the default monthly
window, the per-day lever counts and the per-day byte caps; an explicitly
declared cap or window still applies, and the fuses that never delay healthy
work stay (in flight, attempts and bytes per source, the lifetime envelopes).

Pinned:
  * the defaults: no daily_cap in db.DEFAULT_DOMAIN_CONFIG, none in inc2's
    BUDGET or STREAM_DEFAULTS, no window, no collect_gb_daily, no per-day
    key in stream_levers.json's limits, no bytes_daily in the collector's
    weed config;
  * budget: with no cap declared, a 116.2 SU request passes with 0 SU "left
    today" under the old cap (120 SU charged today), and with the incident's
    86.52; state reports the cap and its remainder as none; need_daily no
    longer refuses an undeclared cap or window; a campaign's 180 is no
    longer cut to 120;
  * a cap or window a campaign (or a domain) declares still refuses;
  * the executor: L18 runs twice in 24 h within the envelope; in_flight 1
    still refuses a second concurrent L18; the campaign envelope (the fuse)
    still refuses; L16 and L17 have no per-day job count and no per-day byte
    cap, and the per-source byte cap stays;
  * the stream's configure command clears daily_cap_su, window_cap_su and
    collect_gb_daily (`none`), without enabling the campaign; `enable` takes
    `none` as well; the lifetime envelope cannot be cleared that way;
  * the collector: no daily byte hold and no daily clip without a declared
    bytes_daily; a declared one still holds;
  * the deploy: deploy_funnel.sh ships tools/db.py to the lab only (the
    cluster's copy belongs to the paused harvest) and its pre-flight runs
    this file, so a tree that declares a default cap again is refused before
    anything is copied.
"""
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_stream_ap_world as W  # noqa: E402
from test_stream_ap_world import B, C, LS, NAME, OWNER, S, X, World, check  # noqa: E402

DAY = 86400.0
NOW = W.T0 + 5 * DAY + 20 * 3600.0          # 2026-10-06T20:00Z: late in a UTC day, as the incident was


def charged(su, epoch, n, campaign=NAME, action="inc_build_segment", lever="L18"):
    """One charged execution of `su` SU at `epoch` (an estimate still committed)."""
    return {"campaign": campaign, "action": action, "lever": lever, "params": {"verb": "build", "stream": "s"},
            "child_exp": "s_s%03d" % n, "charged": True, "est_su": su, "epoch": epoch, "ts": W.utc(epoch),
            "status": "executed", "run_id": "r%d" % n, "job_ids": [str(700 + n)]}


def base_dir():
    return str(pathlib.Path(tempfile.mkdtemp(prefix="nothr_", dir=str(W.TMP))))


def t_defaults():
    print("the defaults hold no time-based cap")
    from weed_optimizer_framework.tools import db
    from weed_optimizer_framework.tools.inc2 import stream as ST
    bud = db.DEFAULT_DOMAIN_CONFIG.get("budget") or {}
    check("db.DEFAULT_DOMAIN_CONFIG's budget has no daily_cap; the lifetime su_envelope stays",
          "daily_cap" not in bud and bud.get("su_envelope") == 1500, bud)
    check("inc2's BUDGET has no daily cap and no monthly window; the envelope and its end stay",
          "daily_cap_su" not in ST.BUDGET and "window_cap_su" not in ST.BUDGET
          and ST.BUDGET.get("envelope_su") == 1000 and ST.BUDGET.get("until") == "2026-12-31", ST.BUDGET)
    d = S.STREAM_DEFAULTS
    check("STREAM_DEFAULTS: daily_cap_su, window_cap_su and collect_gb_daily are none; the envelope, its end and "
          "collect_gb_envelope stay", d["daily_cap_su"] is None and d["window_cap_su"] is None
          and d["collect_gb_daily"] is None and d["envelope_su"] == 1000.0
          and d["envelope_end_utc"] == "2026-12-31T23:59:59Z" and d["collect_gb_envelope"] == 200.0, d)
    menu = LS.load_menu()
    lim = {k: v for k, v in (menu.get("limits") or {}).items() if not k.startswith("_")}
    per_day = sorted("%s.%s" % (lv, k) for lv, row in lim.items() for k in row
                     if "day" in k or "daily" in k or "window" in k)
    check("stream_levers.json's limits name no per-day or windowed count", per_day == [], per_day)
    check("  and keep the fuses: L16 3 attempts, 2 in flight, 50 GB per source; L17 2 in flight; L18 1 in flight; "
          "L22 2 in total; L25 1; L21 1 per milestone; L27 1 per rollback",
          lim["L16"] == {"attempts_per_source": 3, "in_flight": 2, "gb_per_source": 50}
          and lim["L17"] == {"in_flight": 2} and lim["L18"] == {"in_flight": 1} and lim["L22"] == {"total": 2}
          and lim["L25"] == {"total": 1} and lim["L21"] == {"per_milestone": 1} and lim["L27"] == {"per_rollback": 1},
          lim)
    raw = json.loads((W.PKG_ROOT / "weed_optimizer_framework" / "tools" / "collect" / "domains" /
                      "weed.json").read_text())
    bu = raw["budgets"]
    check("the collector's weed config declares no daily byte cap; the per-source cap, the byte envelope and the "
          "attempts stay", bu.get("bytes_daily") is None and bu["bytes_per_source"] == 50e9
          and bu["bytes_envelope"] == 200e9 and bu["attempts_per_source"] == 3, bu)


def t_budget():
    print("budget: an absent daily cap or window is no cap")
    stream = {"name": NAME, "mode": "stream", "envelope_su": 1000.0}
    today = [charged(120.0, NOW - 3600.0, 1)]
    st = B.state(stream, today, None, NOW, base_dir=base_dir())
    check("no daily cap by default: reported as none (cap, remainder, source)",
          st["daily_cap_su"] is None and st["daily_remaining_su"] is None and st["sources"]["daily_cap_su"].startswith("none")
          and st["today_su"] == 120.0, st)
    ok, why = B.fits(st, 116.2)
    check("a 116.2 SU request with 120 SU already charged today (0 left under the old 120 cap) fits", ok, why)
    ok, why = B.fits(st, 116.2, need_daily=True)
    check("  under the envelope rule too (need_daily no longer refuses an undeclared cap)", ok, why)
    st_i = B.state(stream, [charged(33.48, NOW - 3600.0, 1)], None, NOW, base_dir=base_dir())
    check("the incident's numbers (33.48 SU charged today, so 86.52 left under the old cap): 116.2 SU fits",
          B.fits(st_i, 116.2)[0], B.fits(st_i, 116.2)[1])
    env = B.envelope(dict(stream, daily_cap_su=180.0))
    check("a campaign's 180 SU daily cap is no longer cut to the code default's 120",
          env["daily_cap_su"] == 180.0 and env["domain_daily_cap_su"] is None and not env["reasons"], env)
    month = [charged(100.0, NOW - (2 + i) * DAY, i + 1) for i in range(4)]
    st_m = B.state(stream, month, None, NOW, base_dir=base_dir())
    check("no monthly window by default: 400 SU this month, window reported as none",
          st_m["window_cap_su"] is None and st_m["window_remaining_su"] is None and st_m["window_su"] == 400.0, st_m)
    ok, why = B.fits(st_m, 116.2, need_daily=True)
    check("  and a 116.2 SU request fits (no 'no monthly window is declared' refusal)", ok, why)
    check("  su_remaining is the envelope's (1000 - 400)", st_m["su_remaining"] == 600.0, st_m["su_remaining"])
    print("a declared cap or window still refuses")
    st_d = B.state(dict(stream, daily_cap_su=120.0), today, None, NOW, base_dir=base_dir())
    ok, why = B.fits(st_d, 116.2)
    check("a campaign's explicit daily_cap_su 120 refuses 116.2 SU with 120 charged today (today's cap)",
          not ok and any("today's cap" in x for x in why) and st_d["daily_remaining_su"] == 0.0, why)
    st_w = B.state(dict(stream, window_cap_su=350.0), month, None, NOW, base_dir=base_dir())
    ok, why = B.fits(st_w, 116.2)
    check("a campaign's explicit window_cap_su 350 refuses 116.2 SU with 400 this month",
          not ok and any("this month's window" in x for x in why) and st_w["window_remaining_su"] == -50.0, why)
    st_dom = B.state(dict(stream, daily_cap_su=180.0), today, {"su_envelope": 1500, "daily_cap": 100}, NOW,
                     base_dir=base_dir())
    ok, why = B.fits(st_dom, 10.0)
    check("a domain that declares a daily_cap still caps a campaign's (180 -> 100) and refuses past it",
          st_dom["daily_cap_su"] == 100.0 and not ok and any("today's cap of 100" in x for x in why), (st_dom, why))
    print("the envelope fuses stay")
    st_e = B.state(dict(stream, envelope_su=300.0), [charged(250.0, NOW - 3 * DAY, 1)], None, NOW,
                   base_dir=base_dir())
    ok, why = B.fits(st_e, 116.2)
    check("the campaign envelope (300, 250 committed) refuses 116.2 SU", not ok
          and any("campaign envelope" in x for x in why), why)
    st_x = B.state(dict(stream, envelope_su=5000.0), [charged(1450.0, NOW - 3 * DAY, 1)], None, NOW,
                   base_dir=base_dir())
    ok, why = B.fits(st_x, 116.2)
    check("the domain's su_envelope (1,500 across every campaign) refuses past it", not ok
          and any("of the domain's" in x for x in why), why)
    exp = B.state({"name": "weedinc", "envelope_su": 300}, [charged(120.0, NOW - 60, 1, campaign="weedinc")], None,
                  NOW, base_dir=base_dir())
    check("an experiment campaign has no daily cap by default either", exp["daily_cap_su"] is None
          and B.fits(exp, 150.0, need_daily=True)[0], (exp, B.fits(exp, 150.0, need_daily=True)))


def t_executor():
    print("the executor: L18 twice in 24 h within the envelope; in flight, the envelope and declared caps refuse")
    from test_stream_ap_replay import train_world
    w = train_world("nothr_l18")
    w.tick(1)
    cite = [{"artifact": "campaign/context.json", "pointer": "/M", "value": w.M}]
    ctx = X.Context(slurm_sh=w, resources=W.RES, clock=w.clock, lab_repo=str(w.lab), domain="weed",
                    diagnoses=[{"id": "D22", "fired": True, "levers": ["L18"], "cites": cite}])
    base = {"name": NAME, "mode": "stream", "autonomy": "envelope", "autonomy_granted_by": OWNER,
            "envelope_su": 1000.0, "data_autonomy": "on", "last_milestone_pool": "P_0"}

    def l18(n, est=116.2):
        w.advance(5)
        params = {"pkg": "inc2", "stream": w.sid, "k": 2, "exp": "%s_s%03d" % (w.sid, n)}
        p = LS.proposal(NAME, "L18", params, trigger=["D22"], cites=cite, est=est, attempt=n,
                        child_exp=params["exp"])
        p["lever"] = "L18"
        return p
    w._activate()
    r1 = X.submit(l18(1), campaign=base, ctx=ctx)
    check("an L18 of 116.2 SU runs within the envelope", r1["status"] == "executed" and r1.get("basis") == "envelope",
          r1["reasons"])
    r2 = X.submit(l18(2), campaign=base, ctx=ctx)
    check("a second L18 of 116.2 SU the same UTC day runs too: no per-day L18 count, no daily cap",
          r2["status"] == "executed" and r2.get("basis") == "envelope", r2["reasons"])
    bud = X.budget_now(base, ctx)
    check("  the budget reports 232.4 SU today against no daily cap and no window",
          abs(bud["today_su"] - 232.4) < 1e-6 and bud["daily_cap_su"] is None and bud["window_cap_su"] is None, bud)
    r3 = X.submit(l18(3), campaign=dict(base, in_flight={"L18": 1}), ctx=ctx)
    check("in_flight 1 still refuses a second concurrent L18 (filed, 'in flight')", r3["status"] == "filed"
          and any("in flight (limit 1)" in x for x in r3["reasons"]), r3["reasons"])
    r4 = X.submit(l18(4), campaign=dict(base, envelope_su=300.0), ctx=ctx)
    check("the campaign envelope (300 SU, 232.4 committed) still refuses a third", r4["status"] == "filed"
          and any("campaign envelope" in x for x in r4["reasons"]), r4["reasons"])
    r5 = X.submit(l18(5), campaign=dict(base, daily_cap_su=180.0), ctx=ctx)
    check("an explicitly declared daily_cap_su (180) still refuses past it (today's cap)", r5["status"] == "filed"
          and any("today's cap" in x for x in r5["reasons"]), r5["reasons"])
    r6 = X.submit(l18(6), campaign=dict(base, window_cap_su=300.0), ctx=ctx)
    check("an explicitly declared window_cap_su (300) still refuses past it", r6["status"] == "filed"
          and any("this month's window" in x for x in r6["reasons"]), r6["reasons"])
    now = w.clock()
    jobs = [{"campaign": NAME, "lever": lv, "action": act, "status": "executed", "epoch": now - 60 * (i + 1),
             "charged": True, "params": {"source": "s%d" % i, "max_bytes": int(5e9)}}
            for lv, act, n in (("L16", "inc_stream_collect", 13), ("L17", "inc_stream_admit", 7)) for i in range(n)]
    orig = X.executions
    X.executions = lambda ctx=None: jobs
    try:
        w16 = X.stream_limits(w.xctx, {"name": NAME}, "L16", {"action": "inc_stream_collect",
                                                            "params": {"source": "new", "max_bytes": int(10e9)}})
        w17 = X.stream_limits(w.xctx, {"name": NAME}, "L17", {"action": "inc_stream_admit", "params": {}})
        big = X.stream_limits(w.xctx, {"name": NAME}, "L16", {"action": "inc_stream_collect",
                                                            "params": {"source": "s0", "max_bytes": int(46e9)}})
    finally:
        X.executions = orig
    check("13 L16 jobs and 65 GB fetched in the last 24 h leave a 14th fetch free (no per-day count, no per-day GB)",
          w16 == [], w16)
    check("7 L17 jobs in the last 24 h leave an 8th free", w17 == [], w17)
    check("  the per-source cap stays (50 GB unless a person approves)", any("would reach 51.0 GB" in x for x in big),
          big)


def t_configure():
    print("the stream's configure command clears the caps")
    w = World("nothr_cfg", autonomy="off", data_autonomy="off")
    S.configure_stream(NAME, OWNER, daily_cap_su=180.0, window_cap_su=350.0, collect_gb_daily=20.0,
                       cfg_hooks=w.hooks, lab_repo=str(w.lab))
    c0 = w.config()
    check("set-up: the live campaign's explicit caps (180, 350, 20 GB)", (c0.get("daily_cap_su"),
          c0.get("window_cap_su"), c0.get("collect_gb_daily")) == (180.0, 350.0, 20.0), c0)
    w.set_config(enabled=False, paused_reason="a person's pause")
    rc = S.main(["--config", str(w.cfg), "--lab-repo", str(w.lab), "configure", "--name", NAME, "--by", OWNER,
                 "--daily-cap-su", "none", "--window-cap-su", "none", "--collect-gb-daily", "none"])
    c1 = w.config()
    full = S.stream_config(c1, NAME)
    check("`configure --daily-cap-su none --window-cap-su none --collect-gb-daily none` clears all three",
          rc == 0 and c1.get("daily_cap_su") is None and c1.get("window_cap_su") is None
          and c1.get("collect_gb_daily") is None and full["daily_cap_su"] is None
          and full["window_cap_su"] is None and full["collect_gb_daily"] is None, (rc, c1))
    check("  without enabling the campaign or lifting its pause, and the envelope is unchanged",
          c1.get("enabled") is False and c1.get("paused_reason") == "a person's pause"
          and full["envelope_su"] == 1000.0, c1)
    led = [json.loads(x) for x in S.StreamPaths(str(w.lab), w.domain).ledger.read_text().splitlines() if x.strip()]
    cfg_ev = [e for e in led if e.get("event") == "configured"]
    check("  the stream ledger records the change, by whom", cfg_ev and cfg_ev[-1]["decided_by"] == OWNER
          and cfg_ev[-1]["config"]["daily_cap_su"] is None and cfg_ev[-1]["config"]["window_cap_su"] is None,
          cfg_ev[-1:])
    st = B.state(S.stream_config(c1, NAME), [charged(120.0, NOW - 3600.0, 1)], None, NOW, base_dir=base_dir())
    check("  and the budget then has no daily cap and no window", st["daily_cap_su"] is None
          and st["window_cap_su"] is None and B.fits(st, 116.2)[0], st)
    rc = S.main(["--config", str(w.cfg), "--lab-repo", str(w.lab), "configure", "--name", NAME, "--by", OWNER,
                 "--daily-cap-su", "150"])
    check("a person can declare a cap again (150)", rc == 0 and w.config().get("daily_cap_su") == 150.0,
          w.config().get("daily_cap_su"))
    rc = S.main(["--config", str(w.cfg), "--lab-repo", str(w.lab), "enable", "--name", NAME, "--by", OWNER,
                 "--daily-cap-su", "none"])
    check("`enable --daily-cap-su none` clears it as well (and enables)", rc == 0
          and w.config().get("daily_cap_su") is None and w.config().get("enabled") is True, w.config())
    try:
        S.main(["--config", str(w.cfg), "--lab-repo", str(w.lab), "configure", "--name", NAME, "--by", OWNER,
                "--envelope-su", "none"])
        bad = False
    except SystemExit as e:
        bad = e.code == 2
    check("the lifetime envelope cannot be cleared with `none` (a usage error)", bad
          and S.stream_config(w.config(), NAME)["envelope_su"] == 1000.0, w.config().get("envelope_su"))
    raised = False
    try:
        S.configure_stream(NAME, OWNER, envelope_su=S.CLEAR, cfg_hooks=w.hooks, lab_repo=str(w.lab))
    except ValueError:
        raised = True
    check("  nor through configure_stream", raised)
    raised = False
    try:
        S.configure_stream(NAME, OWNER, daily_cap_su=-1.0, cfg_hooks=w.hooks, lab_repo=str(w.lab))
    except ValueError:
        raised = True
    check("a negative cap is still refused", raised)


def t_collector():
    print("the collector: no daily byte cap unless one is declared")
    from weed_optimizer_framework.tools.collect import fetch as F
    from weed_optimizer_framework.tools.collect import prefilter as PF

    class Cfg(object):
        def __init__(self, bu):
            self.bu = bu

        def budgets(self):
            return dict(self.bu)

    bu = {"bytes_per_source": 50e9, "bytes_envelope": 200e9, "attempts_per_source": 3, "approved_bytes": {}}
    ctx = {"state": {}, "bytes_today": 120e9, "bytes_total": 120e9}
    real_free = F.disk_free
    F.disk_free = lambda path: None             # this machine's free disk is not what is tested
    try:
        cap = F._byte_cap(Cfg(dict(bu, bytes_daily=None)), ctx, "src", None, str(W.TMP))
        cap1 = F._byte_cap(Cfg(bu), ctx, "src", None, str(W.TMP))
        cap2 = F._byte_cap(Cfg(dict(bu, bytes_daily=150e9)), ctx, "src", None, str(W.TMP))
    finally:
        F.disk_free = real_free
    check("with bytes_daily null or absent, 120 GB fetched in 24 h does not clip the next fetch (50 GB per source)",
          cap == int(50e9) and cap1 == int(50e9), (cap, cap1))
    check("  a declared 150 GB daily cap clips it to 30 GB", cap2 == int(30e9), cap2)
    check("PF.daily_byte_cap: absent or null is none; a number is the cap",
          PF.daily_byte_cap({}) is None and PF.daily_byte_cap({"bytes_daily": None}) is None
          and PF.daily_byte_cap({"bytes_daily": 5e9}) == 5e9)


def t_deploy():
    """The deploy pre-flight guards the defaults. main moved while this change
    was in review (11c01b0 set db.py's daily_cap to an interim 1500), and a
    merge resolved to that side kept a code-default daily cap; the pre-flight
    set did not run this file, so nothing refused it before the copy."""
    print("the deploy ships db.py to the lab only and its pre-flight runs this file")
    import subprocess
    script = W.PKG_ROOT / "deploy" / "deploy_funnel.sh"
    dry = subprocess.run(["bash", str(script), "--dry-run"], capture_output=True, text=True, cwd=str(W.PKG_ROOT))
    check("deploy_funnel.sh --dry-run (copies nothing, needs no ssh) exits 0", dry.returncode == 0, dry.stderr[-300:])
    pkg = [ln.split(": ", 1)[1] for ln in dry.stdout.splitlines() if ln.startswith("package: ")]
    lab = [ln.split(": ", 1)[1] for ln in dry.stdout.splitlines() if ln.startswith("lab-only: ")]
    pre = [ln.split(": ", 1)[1] for ln in dry.stdout.splitlines() if ln.startswith("pre-flight: ")]
    check("it ships tools/db.py, whose DEFAULT_DOMAIN_CONFIG the lab's ticker reads, to the lab only",
          "weed_optimizer_framework/tools/db.py" in lab and "weed_optimizer_framework/tools/db.py" not in pkg,
          (lab, pkg[-5:]))
    check("its pre-flight runs tests/test_stream_ap_no_throttles.py and tests/test_inc_ap_governance.py",
          "tests/test_stream_ap_no_throttles.py" in pre and "tests/test_inc_ap_governance.py" in pre, pre)


def main():
    for fn in (t_defaults, t_budget, t_executor, t_configure, t_collector, t_deploy):
        W.run_case(fn.__name__, fn)
    return W.closing()


if __name__ == "__main__":
    sys.exit(main())
