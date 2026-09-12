#!/usr/bin/env python3
"""A shadow verdict has to say whether it may be reported, not only whether it was applied.

Between 2026-09-04 and 09-11 nine campaign reviews were produced by a 4.7 GB model
on the lab's RTX 3060 while 458 GB of verified weights sat on the cluster. Every
one was written to disk with a model name, an endpoint, `mode: shadow` and
`applied: false` — and not one field said the verdict was unfit to report.
`applied: false` answers "did the loop act on it", which is a different question
from "may this appear in a result".

These pin the answer to the second question: a loopback endpoint is a lab verdict
whatever model name sits beside it, and a model outside the verified cluster
deployments is not a cluster verdict either.

Run:  python tests/test_review_authority.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools.round_scheduler import _review_authority  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def main():
    a = _review_authority("qwen2.5-coder:7b", "http://127.0.0.1:11434/v1")
    check("the 2026-09-04 wiring is not authoritative", a["authoritative"] is False)
    check("the 2026-09-04 wiring is placed on the lab", a["place"] == "lab")
    check("it says why", bool(a["why"]))

    # A big model name does not launder a loopback endpoint: a compute node has
    # no persistent endpoint, so loopback can only ever be the lab box.
    a = _review_authority("glm-4.7-flash", "http://localhost:11434/v1")
    check("a cluster model name on loopback is still the lab",
          a["place"] == "lab" and a["authoritative"] is False)

    for tag in ("glm-4.7-flash", "qwen3.8:27b", "qwen3:14b", "deepseek-v3:671b",
                "gemma4"):
        a = _review_authority(tag, "http://gpu012.bridges2.psc.edu:8100/v1")
        check("%s off a compute node is authoritative" % tag,
              a["place"] == "cluster" and a["authoritative"] is True)

    a = _review_authority("some-model-nobody-deployed", "http://gpu012:8100/v1")
    check("an unverified model is not authoritative", a["authoritative"] is False)
    check("an unverified model is placed unknown", a["place"] == "unknown")

    a = _review_authority("", "")
    check("no model and no endpoint is not authoritative", a["authoritative"] is False)

    print("%d failure(s)" % len(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
