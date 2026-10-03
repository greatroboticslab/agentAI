#!/usr/bin/env python3
"""A lab fetch's wall clock is sized from the bytes it may move.

Live, 2026-10-02: the lab fetch of zenodo_15808623 (one 49,704,794,641-byte
zip) ran under the lab runner's flat 6 h limit. At the 2.4 MB/s the lab
measured that file needs about 5.8 h, so the limit would kill it near the end,
and a killed fetch restarts from zero (the collector resumes only inside its
own process). lab_job_timeout gives a fetch 2 h plus max_bytes at 1 MB/s,
between 6 h and 48 h; every other lab job keeps 6 h.

Pinned:
  * the sizing rule, its floor, its cap, and bad inputs;
  * in the stream world of test_stream_ap_world, a lab fetch (L16L) of a
    49.7 GB source is launched with that timeout, and a lab fetch of a small
    source with the 6 h floor.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_stream_ap_world as W  # noqa: E402
from test_stream_ap_world import S, World, check  # noqa: E402

H = 3600
SIU = 49704794641


def test_rule():
    print("the sizing rule")
    t = S.lab_job_timeout
    check("a 49.7 GB fetch gets 2 h + 49.7e9 B at 1 MB/s (about 15.8 h)", t("fetch", SIU) == int(2 * H + SIU / 1e6),
          t("fetch", SIU))
    check("a 1 GB fetch keeps the 6 h floor", t("fetch", 1e9) == 6 * H, t("fetch", 1e9))
    check("a 500 GB fetch is capped at 48 h", t("fetch", 5e11) == 48 * H, t("fetch", 5e11))
    check("sync, discover and names keep 6 h whatever their bytes",
          all(t(k, SIU) == 6 * H for k in ("sync", "discover", "names")))
    check("a fetch with no, bad or non-positive max_bytes keeps 6 h",
          all(t("fetch", b) == 6 * H for b in (None, "", "x", 0, -5)))


CAND = {"id": "gh_big", "provider": "github", "licence": "MIT", "target_classes": ["Purslane"],
        "expected_target_boxes": 900}


def launched_fetch(tag, nbytes):
    w = World(tag)
    w.ready_r0()
    w.queue(0)
    w.placement({"hf": "pass", "ftp": "fail"})
    w.candidates([dict(CAND, bytes=nbytes)])
    w.tick(3)
    got = [x for x in w.runner.launched if "fetch" in x["argv"] and "gh_big" in x["argv"]]
    return got


def test_launch():
    print("the lab runner is given the sized timeout")
    big = launched_fetch("labto_big", 4.9e10)
    want = S.lab_job_timeout("fetch", int(4.9e10))
    check("a lab fetch (L16L) of a 49 GB source is launched with %d s (%.1f h)" % (want, want / 3600.0),
          big and big[0].get("timeout") == want and want > 6 * H, [(x["argv"][-6:], x.get("timeout")) for x in big])
    small = launched_fetch("labto_small", 1e9)
    check("a lab fetch of a 1 GB source is launched with the 6 h floor",
          small and small[0].get("timeout") == 6 * H, [x.get("timeout") for x in small])


def main():
    test_rule()
    test_launch()
    return W.closing()


if __name__ == "__main__":
    sys.exit(main())
