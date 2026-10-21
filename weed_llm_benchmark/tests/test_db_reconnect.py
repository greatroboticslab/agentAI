#!/usr/bin/env python3
"""db.py retries a failed Mongo connection instead of caching the failure.

After the 2026-10-20 reboot the dashboard started 60 s before mongod; the first
probe failed and every later call returned the JSON fallback (no domains) until
a manual restart. These checks need no Mongo: pymongo is replaced by a fake.

Run:  python -m pytest tests/test_db_reconnect.py     (or) python tests/test_db_reconnect.py
"""
import os
import sys
import types
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools import db  # noqa: E402

_fails = []


def ck(name, cond):
    print(("  ok   " if cond else "  FAIL ") + name)
    if not cond:
        _fails.append(name)


class _FakeAdmin:
    def __init__(self, up):
        self.up = up

    def command(self, _):
        if not self.up:
            raise ConnectionError("Connection refused")
        return {"ok": 1}


class _FakeClient:
    up = False                 # flipped to True to model mongod coming up
    made, closed = [], []

    def __init__(self, url, serverSelectionTimeoutMS=None):
        self.admin = _FakeAdmin(_FakeClient.up)
        _FakeClient.made.append(self)

    def __getitem__(self, name):
        return types.SimpleNamespace(name=name)

    def close(self):
        _FakeClient.closed.append(self)


def run():
    saved_mod = sys.modules.get("pymongo")
    saved_url = os.environ.get("AGENTAI_MONGO_URL")
    sys.modules["pymongo"] = types.SimpleNamespace(MongoClient=_FakeClient)
    os.environ["AGENTAI_MONGO_URL"] = "mongodb://127.0.0.1:27017/agentai"
    try:
        db.reset()
        ck("first probe fails while mongod is down", db.available() is False)
        ck("the failure is time-stamped", db._state["failed_at"] is not None)
        ck("the failed client is closed", len(_FakeClient.closed) == 1)

        n = len(_FakeClient.made)
        db.available()
        ck("no new attempt inside the retry window", len(_FakeClient.made) == n)

        _FakeClient.up = True
        db.available()
        ck("still cached inside the window even if mongod is up", len(_FakeClient.made) == n)

        db._state["failed_at"] -= db._RETRY_AFTER_S + 1
        ck("after the window the next call reconnects", db.available() is True)
        ck("a success clears the failure stamp", db._state["failed_at"] is None)
        ck("exactly one new attempt", len(_FakeClient.made) == n + 1)
        ck("the live client is not closed", len(_FakeClient.closed) == 1)

        db.available()
        ck("a live handle is reused, no further attempts", len(_FakeClient.made) == n + 1)
    finally:
        db.reset()
        if saved_mod is None:
            sys.modules.pop("pymongo", None)
        else:
            sys.modules["pymongo"] = saved_mod
        if saved_url is None:
            os.environ.pop("AGENTAI_MONGO_URL", None)
        else:
            os.environ["AGENTAI_MONGO_URL"] = saved_url


def test_db_reconnect():
    run()
    assert not _fails, _fails


if __name__ == "__main__":
    run()
    print("FAILED: %s" % _fails if _fails else "all ok")
    sys.exit(1 if _fails else 0)
