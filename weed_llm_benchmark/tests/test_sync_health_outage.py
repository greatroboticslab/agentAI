#!/usr/bin/env python3
"""sync_health: an announced cluster outage turns the red stale-data alarm into a notice that names the outage
and the data's date, and only while the record still explains the staleness.

Bridges-2 went offline on 2026-11-09 for a datacenter move announced for four to six weeks; the red
"DATA MAY BE STALE" bar stood on every page for the whole outage. These checks write a heartbeat and an outage
record into a temporary tree and read the verdict and the banner. No dashboard, no Mongo.

Run:  python -m pytest tests/test_sync_health_outage.py     (or) python tests/test_sync_health_outage.py
"""
import calendar
import json
import os
import pathlib
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools import sync_health as SH  # noqa: E402

_fails = []


def ck(name, cond):
    print(("  ok   " if cond else "  FAIL ") + name)
    if not cond:
        _fails.append(name)


def ts(s):
    return float(calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ")))


LAST_OK = ts("2026-11-09T11:51:52Z")
NOW = ts("2026-11-22T19:30:00Z")
RECORD = {"note": "Bridges-2 is being moved to a new datacenter (2026-11-09 to about 2026-12-14)",
          "since_utc": "2026-11-09T12:00:00Z", "until_utc": "2026-12-14T23:59:59Z",
          "source": "https://www.psc.edu/resources/bridges-2/"}


def setup(tmp, record=None, raw=None, last_ok=LAST_OK):
    fw = pathlib.Path(tmp) / "results" / "framework"
    fw.mkdir(parents=True, exist_ok=True)
    (fw / "sync_status.json").write_text(json.dumps({"last_success_ts": last_ok, "last_outcome": "failed",
                                                     "detail": "registry pull failed (rc=255)"}))
    out = fw / "sync_outage.json"
    if out.exists():
        out.unlink()
    if raw is not None:
        out.write_text(raw)
    elif record is not None:
        out.write_text(json.dumps(record))
    SH._CTX["repo"] = str(tmp)
    SH._CACHE.update(at=0.0, val=None)


def banner(now):
    SH._CACHE.update(at=now, val=SH.verdict(now))     # the banner reads the memoised verdict
    return SH.banner_html()


def main():
    with tempfile.TemporaryDirectory() as tmp:
        setup(tmp)
        v = SH.verdict(NOW)
        ck("no record: crit, no known outage", v["level"] == "crit" and v["known_outage"] is None)
        ck("no record: the red alarm", "DATA MAY BE STALE" in banner(NOW) and "#7f1d1d" in banner(NOW))

        setup(tmp, RECORD)
        v = SH.verdict(NOW)
        ck("record: the level stays crit (the health API's 503 is the heartbeat's)", v["level"] == "crit")
        ck("record: the outage is named", (v["known_outage"] or {}).get("note") == RECORD["note"])
        b = banner(NOW)
        ck("record: a notice, not the red alarm", "DATA MAY BE STALE" not in b and "#fef3c7" in b)
        ck("record: the notice names the outage", "Cluster offline: Bridges-2 is being moved" in b)
        ck("record: the notice gives the data's date", "as of 2026-11-09" in b)

        ck("record: still a notice a week past the announced end",
           SH.verdict(ts("2026-12-21T23:00:00Z"))["known_outage"] is not None)
        late = ts("2026-12-22T00:00:01Z")
        ck("record: past the end plus a week, the red alarm again",
           SH.verdict(late)["known_outage"] is None and "DATA MAY BE STALE" in banner(late))

        setup(tmp, RECORD, last_ok=ts("2026-11-10T08:00:00Z"))
        ck("a sync that succeeded after the outage began spends the record",
           SH.verdict(NOW)["known_outage"] is None)

        for name, rec in (("no note", dict(RECORD, note="")),
                          ("no until", {k: v for k, v in RECORD.items() if k != "until_utc"}),
                          ("bad time", dict(RECORD, since_utc="2026-11-09"))):
            setup(tmp, rec)
            ck("record with %s: explains nothing" % name, SH.verdict(NOW)["known_outage"] is None)
        setup(tmp, raw="{not json")
        ck("unreadable record: explains nothing", SH.verdict(NOW)["known_outage"] is None)

        setup(tmp, dict(RECORD, note="<b>x</b> & y"))
        ck("the note is escaped in the page", "&lt;b&gt;x&lt;/b&gt; &amp; y" in banner(NOW))

        setup(tmp, RECORD, last_ok=NOW - 600)
        v = SH.verdict(NOW)
        ck("fresh data: ok, no banner, no outage read", v["ok"] and v["known_outage"] is None
           and banner(NOW) == "")
    SH._CTX.pop("repo", None)
    SH._CACHE.update(at=0.0, val=None)
    return 0 if not _fails else 1


def test_sync_health_outage():
    assert main() == 0, _fails


if __name__ == "__main__":
    rc = main()
    print("%d check(s) failed" % len(_fails) if _fails else "all checks passed")
    sys.exit(rc)
