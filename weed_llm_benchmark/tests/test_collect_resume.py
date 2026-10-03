#!/usr/bin/env python3
"""Resumable downloads (collect/transport.py: Net.download and
FtpSession.download) against a fake HTTP server behind Net._open and a fake
ftplib.FTP; no socket is opened.

The incident (2026-10-02): a 49.7 GB archive ended after 9.1 GB because the
server closed the stream; the size check refused it and deleted all 9.1 GB,
and two such failures held the stream's data lane. Pinned:
  (a) a stream dropped at 30% resumes with "Range: bytes=<n>-" and a 206:
      one resume, the exact bytes, sha256 and the published md5; the
      server's Content-Length alone also detects the cut; with no size known
      at all, a 206 of total "*" that ends cleanly before its own end
      (Content-Range last + 1) resumes again;
  (b) several drops (an early end, a reset, an IncompleteRead) each resume
      from the bytes held, with a 1 s pause while streams make progress; a
      503 to a resume is retried at the same bytes;
  (c) a server that ignores the Range (200) sends the whole file again: the
      file and its hashes restart from zero and are still exact; a smaller
      file then leaves no stale tail of the first; bytes streamed after the
      restart count as progress; that restart is the file's one restart
      from zero, so a server that never honours a Range fails at its second
      200 (3 requests, not one whole file per resume);
  (d) a Content-Range that does not start at n, a wrong total, a 416 or a
      changed strong ETag restart from zero once (never a splice); a second
      refusal fails cleanly; a total of "*" and a changed weak ETag continue;
  (e) a server that breaks every stream: after download.resume_attempts
      resumes the download fails with no file and no .tmp, the pause
      doubling while no progress is made; the default is 20 resumes;
  (f) max_bytes holds across resumes (bytes streamed over several resumes,
      a 206 whose total is over the cap, and a 200 to a Range request whose
      Content-Length is over the cap);
  (g) a checksum mismatch after a resumed download (a wrong published md5,
      or a file that changed under a resume without an ETag) leaves nothing;
  (h) the size-scaled deadline holds for the whole file: no resume past it;
  (i) unchanged: a 200 whose Content-Length is under the listed size fails
      the size check without a resume; the first request still retries a
      503 (download.attempts);
  (j) FTP: a broken transfer logs in again and resumes with REST; a 426 and
      a short 226 resume; a refused REST restarts from zero once; a transfer
      that never completes fails cleanly; a refused RETR is a ProviderError;
      a first login refused or broken off is a ProviderError, and a login
      that fails (first or of a resume) closes its connection;
  (k) through fetch (L16): fetch.json records each file's resumes, outside
      its checksums; a full disk leaves no blob;
  (l) a local write error is not a network error: ENOSPC on the 2nd write
      (HTTP and FTP) fails at once, with no resume and nothing left (no
      file, no .tmp); a chunk that never reached the disk fails the on-disk
      size check although its streamed sha256 is the published one; a write
      error reported only at the final flush leaves nothing.

Run:  python3 tests/test_collect_resume.py
"""
import contextlib
import errno
import ftplib
import hashlib
import http.client
import io
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_collect_world as W  # noqa: E402

TMP = W.setup("collect_resume_")
check, raises = W.check, W.raises
URL = "https://files.example/data.zip"
L = 2 * (1 << 20) + (1 << 19) + 123             # 2.5 MiB and change: crosses the 1 MiB chunks
DATA = random.Random(7).randbytes(L)
OTHER = random.Random(8).randbytes(L)           # the same size, other bytes (a file that changed)


def at(frac):
    return int(L * frac)


def sha(b, algo="sha256"):
    return hashlib.new(algo, b).hexdigest()


def left_nothing(dest):
    dest = pathlib.Path(dest)
    return not dest.exists() and not dest.with_name(dest.name + ".tmp").exists()


class Body(object):
    """A response body that ends at `data`'s end: cleanly ("eof"), or with a
    connection reset ("reset") or an IncompleteRead ("incomplete")."""

    def __init__(self, data, broken=False, how="eof"):
        self.buf = io.BytesIO(data)
        self.broken, self.how = broken, how
        self.closed = False

    def read(self, n=-1):
        b = self.buf.read(n)
        if not b and self.broken and self.how == "reset":
            raise ConnectionResetError(104, "Connection reset by peer")
        if not b and self.broken and self.how == "incomplete":
            raise http.client.IncompleteRead(b"", 4096)
        return b

    def close(self):
        self.closed = True


def server(steps, data=DATA, **cfg):
    """A Net whose HTTP primitive is a scripted file server. Each request takes
    the next step (a dict; {} once they run out):
      cut          the absolute offset where the stream breaks (how: eof,
                   reset, incomplete); by default the body runs to the end
      ignore_range answer 200 with the whole file whatever the Range
      status       answer this status with an empty body (416, 503, 404)
      cr_start, cr_total   what the 206's Content-Range claims
      etag         the answer's ETag; data: other bytes for this answer
      no_length    no Content-Length header
    log holds (the Range header sent or None, the status); pauses the waits."""
    from weed_optimizer_framework.tools.collect.transport import Net

    class Server(Net):
        def __init__(self):
            d = {"base_timeout_s": 30, "min_rate_bytes_per_s": 1e9, "read_timeout_s": 5, "attempts": 1}
            d.update(cfg)
            super().__init__(d)
            self.content = data
            self.steps = list(steps)
            self.log, self.pauses = [], []

        def _pause(self, seconds):
            self.pauses.append(seconds)

        def _open(self, url, params=None, headers=None, data=None, method=None, timeout=None):
            rng = dict(headers or {}).get("Range")
            s = dict(self.steps.pop(0)) if self.steps else {}
            if s.get("status"):
                self.log.append((rng, s["status"]))
                return s["status"], {}, io.BytesIO(b"")
            content = s.get("data", self.content)
            start = int(rng.split("=")[1].rstrip("-")) if rng and not s.get("ignore_range") else 0
            hd = {"ETag": s["etag"]} if s.get("etag") else {}
            if start:
                st = 206
                hd["Content-Range"] = "bytes %d-%d/%s" % (s.get("cr_start", start), len(content) - 1,
                                                          s.get("cr_total", len(content)))
                if not s.get("no_length"):
                    hd["Content-Length"] = str(len(content) - start)
            else:
                st = 200
                if not s.get("no_length"):
                    hd["Content-Length"] = str(len(content))
            cut = s.get("cut")
            self.log.append((rng, st))
            return st, hd, Body(content[start:] if cut is None else content[start:max(cut, start)],
                                broken=cut is not None, how=s.get("how", "eof"))

    return Server()


def ranges(net):
    return [r for r, _st in net.log]


class FlakyFile(object):
    """The .tmp's file handle on a failing disk: its k-th write raises ENOSPC
    ("enospc") or returns without writing ("lost": a chunk that never reached
    the disk), or its flush raises EDQUOT ("flush")."""

    def __init__(self, fh, k, mode):
        self.fh, self.k, self.mode, self.writes = fh, k, mode, 0

    def write(self, b):
        self.writes += 1
        if self.writes == self.k and self.mode == "enospc":
            raise OSError(errno.ENOSPC, "No space left on device")
        if self.writes == self.k and self.mode == "lost":
            return len(b)
        return self.fh.write(b)

    def flush(self):
        if self.mode == "flush":
            raise OSError(errno.EDQUOT, "Disk quota exceeded")
        return self.fh.flush()

    def __getattr__(self, name):
        return getattr(self.fh, name)


@contextlib.contextmanager
def flaky_disk(k, mode="enospc"):
    """Every download in the block writes its .tmp through a FlakyFile."""
    from weed_optimizer_framework.tools.collect import transport as T
    real = T._Sink

    class Sink(real):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.fh = FlakyFile(self.fh, k, mode)
    T._Sink = Sink
    try:
        yield
    finally:
        T._Sink = real


def test_one_drop():
    print("(a) a stream dropped at 30% resumes once")
    dest = TMP / "a" / "f.zip"
    net = server([{"cut": at(0.3)}])
    r = net.download(URL, dest, expected_size=L, checksums={"md5": sha(DATA, "md5")}, what="a")
    check("one resume, with Range: bytes=<n>- answered by a 206", r["resumes"] == 1
          and net.log == [(None, 200), ("bytes=%d-" % at(0.3), 206)], (r.get("resumes"), net.log))
    check("the exact bytes, sha256 and the published md5", dest.read_bytes() == DATA and r["bytes"] == L
          and r["sha256"] == sha(DATA) and r["md5"] == sha(DATA, "md5"))
    check("the result keeps its keys and adds resumes", set(r) == {"bytes", "sha256", "md5", "resumes"}, sorted(r))
    check("no .tmp is left and the pause was 1 s", not dest.with_name("f.zip.tmp").exists() and net.pauses == [1.0],
          net.pauses)
    check("the resume is a call of the session's record too (net.calls)", net.calls == [(URL, 200), (URL, 206)],
          net.calls)
    dest2 = TMP / "a" / "g.zip"
    net = server([{"cut": at(0.3)}])
    r = net.download(URL, dest2)
    check("without a listed size, the server's Content-Length detects the cut", r["resumes"] == 1
          and dest2.read_bytes() == DATA and r["bytes"] == L, r)
    dest3 = TMP / "a" / "h.zip"
    net = server([{"cut": at(0.3), "how": "reset", "no_length": True}, {"no_length": True, "cr_total": "*"}])
    r = net.download(URL, dest3)
    check("with no size known at all, a reset stream is resumed, never taken as the whole file",
          r["resumes"] == 1 and dest3.read_bytes() == DATA, (r, net.log))
    for i, length in enumerate((False, True)):
        dest4 = TMP / "a" / ("star%d.zip" % i)
        net = server([{"cut": at(0.3), "how": "reset", "no_length": True},
                      {"cr_total": "*", "no_length": not length, "cut": at(0.6)},
                      {"cr_total": "*", "no_length": not length}])
        r = net.download(URL, dest4)
        check("with no size known, a 206 \"bytes n-%d/*\"%s that ends cleanly at 60%% is short of its own end: "
              "resumed again, exact" % (L - 1, " with a Content-Length" if length else ""), r["resumes"] == 2
              and ranges(net) == [None, "bytes=%d-" % at(0.3), "bytes=%d-" % at(0.6)] and r["bytes"] == L
              and dest4.read_bytes() == DATA and r["sha256"] == sha(DATA), (r, net.log))


def test_several_drops():
    print("(b) several drops")
    dest = TMP / "b" / "f.zip"
    cuts = [(at(0.2), "eof"), (at(0.45), "reset"), (at(0.7), "incomplete"), (at(0.9), "eof")]
    net = server([{"cut": c, "how": h} for c, h in cuts])
    r = net.download(URL, dest, expected_size=L, checksums={"sha256": sha(DATA)})
    check("each drop resumes from the bytes held", r["resumes"] == 4
          and ranges(net) == [None] + ["bytes=%d-" % c for c, _h in cuts], net.log)
    check("an early end, a reset and an IncompleteRead all resume; the file is exact",
          dest.read_bytes() == DATA and r["sha256"] == sha(DATA))
    check("the pause stays 1 s while the streams make progress", net.pauses == [1.0] * 4, net.pauses)
    dest2 = TMP / "b" / "g.zip"
    n = at(0.3)
    net = server([{"cut": n}, {"status": 503}, {}])
    r = net.download(URL, dest2, expected_size=L, checksums={"sha256": sha(DATA)})
    check("a 503 to a resume is retried at the same bytes (the pause doubling: no progress); exact",
          r["resumes"] == 2 and net.log == [(None, 200), ("bytes=%d-" % n, 503), ("bytes=%d-" % n, 206)]
          and net.pauses == [1.0, 2.0] and dest2.read_bytes() == DATA and r["sha256"] == sha(DATA),
          (net.log, net.pauses))


def test_range_ignored():
    from weed_optimizer_framework.tools.collect import ProviderError
    print("(c) a server that ignores the Range")
    dest = TMP / "c" / "f.zip"
    net = server([{"cut": at(0.3)}, {"ignore_range": True}])
    r = net.download(URL, dest, expected_size=L, checksums={"md5": sha(DATA, "md5")})
    check("a 200 to a Range request restarts the file from zero; still exact", r["resumes"] == 1
          and net.log == [(None, 200), ("bytes=%d-" % at(0.3), 200)] and dest.read_bytes() == DATA
          and r["sha256"] == sha(DATA), net.log)
    dest2 = TMP / "c" / "g.zip"
    net = server([{"cut": at(0.3)}, {"ignore_range": True, "cut": at(0.5)}])
    r = net.download(URL, dest2, expected_size=L, checksums={"sha256": sha(DATA)})
    check("... and after that restart a later drop resumes from the restarted bytes (the hashes restarted too)",
          r["resumes"] == 2 and ranges(net) == [None, "bytes=%d-" % at(0.3), "bytes=%d-" % at(0.5)]
          and dest2.read_bytes() == DATA, net.log)
    small = OTHER[:at(0.4)]
    dest3 = TMP / "c" / "h.zip"
    net = server([{"cut": at(0.6)}, {"ignore_range": True, "data": small}])
    r = net.download(URL, dest3)
    check("a 200 to a Range request with a smaller file: the .tmp is emptied first, no stale tail of the first "
          "file is left", r["resumes"] == 1 and dest3.read_bytes() == small and r["bytes"] == len(small)
          and r["sha256"] == sha(small), (r, net.log))
    dest4 = TMP / "c" / "i.zip"
    net = server([{"cut": at(0.5)}, {"ignore_range": True, "cut": at(0.3)}, {}])
    r = net.download(URL, dest4, expected_size=L, checksums={"sha256": sha(DATA)})
    check("bytes streamed after a restart from zero count as progress (bytes moved, not bytes held): the pause "
          "stays 1 s", r["resumes"] == 2 and net.pauses == [1.0, 1.0] and dest4.read_bytes() == DATA,
          (net.pauses, net.log))
    dest5 = TMP / "c" / "j.zip"
    net = server([{"cut": at(0.3)}] + [{"ignore_range": True, "cut": at(0.3)}] * 30)
    e = raises(lambda: net.download(URL, dest5, expected_size=L, max_bytes=L), ProviderError)
    check("a server that never honours a Range: the second 200 to a Range request fails (3 requests, not one "
          "from zero per resume), leaving nothing", e is not None and "ignored the Range" in str(e)
          and "refused again" in str(e) and len(net.log) == 3 and left_nothing(dest5), (e, net.log))
    dest6 = TMP / "c" / "k.zip"
    net = server([{"cut": at(0.3)}, {"status": 416}, {"cut": at(0.5)}, {"ignore_range": True}])
    e = raises(lambda: net.download(URL, dest6, expected_size=L), ProviderError)
    check("... and a 200 to a Range request after a 416's restart from zero fails the same way", e is not None
          and "ignored the Range" in str(e) and len(net.log) == 4 and left_nothing(dest6), (e, net.log))


def test_refused_resumes():
    from weed_optimizer_framework.tools.collect import ProviderError
    print("(d) a refused resume restarts from zero once")
    n = at(0.3)
    cases = [("a Content-Range that does not start at n", {"cr_start": n - 100}),
             ("a Content-Range whose total is not the listed size", {"cr_total": L + 1}),
             ("a 416", {"status": 416})]
    for i, (name, step) in enumerate(cases):
        dest = TMP / "d" / ("f%d.zip" % i)
        net = server([{"cut": n}, step])
        r = net.download(URL, dest, expected_size=L, checksums={"sha256": sha(DATA)})
        check("%s: a restart from zero (no Range), exact" % name, r["resumes"] == 2
              and ranges(net) == [None, "bytes=%d-" % n, None] and dest.read_bytes() == DATA, net.log)
    dest = TMP / "d" / "etag.zip"
    net = server([{"cut": n, "etag": '"v1"'}, {"etag": '"v2"', "data": OTHER}, {"etag": '"v2"', "data": OTHER}])
    r = net.download(URL, dest, expected_size=L)
    check("a changed strong ETag is never spliced: the restart takes the new file whole",
          r["resumes"] == 2 and dest.read_bytes() == OTHER and r["sha256"] == sha(OTHER), net.log)
    dest = TMP / "d" / "weak.zip"
    net = server([{"cut": n, "etag": 'W/"a"'}, {"etag": 'W/"b"'}])
    r = net.download(URL, dest, expected_size=L, checksums={"sha256": sha(DATA)})
    check("a weak ETag (which may change for the same bytes) does not refuse a resume", r["resumes"] == 1
          and dest.read_bytes() == DATA, net.log)
    dest = TMP / "d" / "star.zip"
    net = server([{"cut": n}, {"cr_total": "*"}])
    r = net.download(URL, dest, expected_size=L, checksums={"sha256": sha(DATA)})
    check("a Content-Range total of * continues the file (the final checks decide)", r["resumes"] == 1
          and dest.read_bytes() == DATA, net.log)
    dest = TMP / "d" / "twice.zip"
    net = server([{"cut": n}, {"status": 416}, {"cut": at(0.5)}, {"status": 416}])
    e = raises(lambda: net.download(URL, dest, expected_size=L), ProviderError)
    check("a second refusal after the restart fails cleanly", e is not None and "refused again" in str(e)
          and left_nothing(dest) and len(net.log) == 4, (e, net.log))


def test_budget():
    from weed_optimizer_framework.tools.collect import ProviderError
    from weed_optimizer_framework.tools.collect import transport as T
    print("(e) the resume budget")
    dest = TMP / "e" / "f.zip"
    net = server([{"cut": at(0.3)}] + [{"cut": 0, "how": "reset"}] * 10, resume_attempts=3)
    e = raises(lambda: net.download(URL, dest, expected_size=L, checksums={"sha256": sha(DATA)}), ProviderError)
    check("after download.resume_attempts (3) resumes the download fails", e is not None
          and "resume_attempts" in str(e) and len(net.log) == 4, (e, net.log))
    check("... with no file and no .tmp left", left_nothing(dest))
    check("the pause doubles while no stream makes progress", net.pauses == [1.0, 2.0, 4.0], net.pauses)
    check("the default budget is 20 resumes", T.Net({}).resume_attempts == 20)
    net = server([{"cut": at(0.3)}], resume_attempts=0)
    e = raises(lambda: net.download(URL, TMP / "e" / "g.zip", expected_size=L), ProviderError)
    check("resume_attempts 0 turns resuming off (one request, nothing left)", e is not None and len(net.log) == 1
          and left_nothing(TMP / "e" / "g.zip"), net.log)


def test_cap():
    from weed_optimizer_framework.tools.collect import ByteCapExceeded
    print("(f) max_bytes across resumes")
    dest = TMP / "f" / "f.zip"
    steps = [{"cut": at(0.25), "how": "reset", "no_length": True},
             {"cut": at(0.5), "how": "reset", "no_length": True, "cr_total": "*"},
             {"cut": at(0.75), "how": "reset", "no_length": True, "cr_total": "*"},
             {"no_length": True, "cr_total": "*"}]
    net = server(steps)
    e = raises(lambda: net.download(URL, dest, max_bytes=at(0.6)), ByteCapExceeded)
    check("the cap counts the bytes of every resumed stream (each 25%, the cap 60%: trips in the third)",
          e is not None and len(net.log) == 3 and left_nothing(dest), (e, net.log))
    dest = TMP / "f" / "g.zip"
    net = server([{"cut": at(0.3), "how": "reset", "no_length": True}, {"no_length": True}])
    e = raises(lambda: net.download(URL, dest, max_bytes=at(0.5)), ByteCapExceeded)
    check("a 206 whose total is over the cap refuses before its bytes", e is not None and "Content-Range total"
          in str(e) and len(net.log) == 2 and left_nothing(dest), (e, net.log))
    dest = TMP / "f" / "h.zip"
    net = server([{"cut": at(0.3), "how": "reset", "no_length": True}, {"ignore_range": True}])
    e = raises(lambda: net.download(URL, dest, max_bytes=at(0.5)), ByteCapExceeded)
    check("a 200 to a Range request whose Content-Length is over the cap refuses before its bytes",
          e is not None and "Content-Length %d over" % L in str(e) and len(net.log) == 2 and left_nothing(dest),
          (e, net.log))


def test_mismatch():
    from weed_optimizer_framework.tools.collect import ChecksumMismatch
    print("(g) a checksum mismatch after a resume")
    dest = TMP / "g" / "f.zip"
    net = server([{"cut": at(0.3)}])
    e = raises(lambda: net.download(URL, dest, expected_size=L, checksums={"md5": "0" * 32}), ChecksumMismatch)
    check("a wrong published md5 after a resumed download leaves nothing", e is not None and len(net.log) == 2
          and left_nothing(dest), (e, net.log))
    dest = TMP / "g" / "g.zip"
    net = server([{"cut": at(0.3)}, {"data": OTHER}])
    e = raises(lambda: net.download(URL, dest, expected_size=L, checksums={"sha256": sha(DATA)}), ChecksumMismatch)
    check("a file that changed under a resume (no ETag: spliced) fails its checksum and leaves nothing",
          e is not None and len(net.log) == 2 and left_nothing(dest), (e, net.log))


def test_deadline():
    from weed_optimizer_framework.tools.collect import ProviderError
    print("(h) the deadline holds for the whole file")
    dest = TMP / "h" / "f.zip"
    net = server([{"cut": at(0.3)}], base_timeout_s=0.5, min_rate_bytes_per_s=1e15)
    e = raises(lambda: net.download(URL, dest, expected_size=L), ProviderError)
    check("no resume whose pause would cross the size-scaled deadline", e is not None and "deadline" in str(e)
          and len(net.log) == 1 and left_nothing(dest), (e, net.log))


def test_unchanged():
    from weed_optimizer_framework.tools.collect import ChecksumMismatch
    print("(i) unchanged behaviour")
    dest = TMP / "i" / "f.zip"
    net = server([{"data": DATA[:L - 10]}])
    e = raises(lambda: net.download(URL, dest, expected_size=L), ChecksumMismatch)
    check("a 200 that delivers all of a Content-Length under the listed size fails the size check, no resume",
          e is not None and "the provider lists" in str(e) and len(net.log) == 1 and left_nothing(dest), (e, net.log))
    dest = TMP / "i" / "g.zip"
    net = server([{"status": 503}], attempts=2)
    r = net.download(URL, dest, expected_size=L)
    check("the first request still retries a 503 (download.attempts), and that is not a resume",
          r["resumes"] == 0 and net.log == [(None, 503), (None, 200)] and dest.read_bytes() == DATA, net.log)


class FakeFTP(object):
    """ftplib.FTP's surface for FtpSession. plan: one step per RETR (cut, how:
    reset, 426, eof; rest_refused; perm); login_plan: one entry per login,
    None or the exception it raises; instances: every connection, each with
    closed set once close() or quit() ran."""
    files, plan, logins, retrs, login_plan, instances = {}, [], [], [], [], []

    def __init__(self, host, timeout=None):
        self.host = host
        self.closed = False
        FakeFTP.instances.append(self)

    def login(self, user, password):
        FakeFTP.logins.append((user, password))
        e = FakeFTP.login_plan.pop(0) if FakeFTP.login_plan else None
        if e is not None:
            raise e

    def voidcmd(self, cmd):
        return "200 ok"

    def size(self, path):
        return len(self.files[path])

    def retrbinary(self, cmd, callback, blocksize=8192, rest=None):
        path = cmd.split(" ", 1)[1]
        step = FakeFTP.plan.pop(0) if FakeFTP.plan else {}
        FakeFTP.retrs.append(rest)
        if step.get("perm"):
            raise ftplib.error_perm("550 %s: no such file" % path)
        if rest is not None and step.get("rest_refused"):
            raise ftplib.error_perm("502 REST not implemented")
        data = self.files[path]
        end = len(data) if step.get("cut") is None else step["cut"]
        for i in range(rest or 0, end, blocksize):
            callback(data[i:min(i + blocksize, end)])
        how = step.get("how", "reset") if step.get("cut") is not None else None
        if how == "reset":
            raise ConnectionResetError(104, "Connection reset by peer")
        if how == "426":
            raise ftplib.error_temp("426 Connection closed; transfer aborted.")
        return "226 Transfer complete."

    def quit(self):
        self.closed = True

    def close(self):
        self.closed = True


def ftp_session(plan, login_plan=(), **cfg):
    FakeFTP.files = {"jpegs/X/1.zip": DATA}
    FakeFTP.plan, FakeFTP.logins, FakeFTP.retrs = list(plan), [], []
    FakeFTP.login_plan, FakeFTP.instances = list(login_plan), []
    net = server([], **cfg)
    return net, net.ftp("ftp.example", "user", "pw")


def test_ftp():
    from weed_optimizer_framework.tools.collect import ProviderError
    print("(j) FTP")
    real = ftplib.FTP
    ftplib.FTP = FakeFTP
    try:
        sums = {"sha512": sha(DATA, "sha512")}
        net, ses = ftp_session([{"cut": at(0.3), "how": "reset"}])
        dest = TMP / "j" / "a.zip"
        r = ses.download("jpegs/X/1.zip", dest, expected_size=L, checksums=sums)
        check("a reset transfer logs in again and resumes with REST at the bytes held", r["resumes"] == 1
              and FakeFTP.retrs == [None, at(0.3)] and len(FakeFTP.logins) == 2, (FakeFTP.retrs, FakeFTP.logins))
        check("... exact, against the published sha512", dest.read_bytes() == DATA and r["sha512"] == sums["sha512"]
              and r["bytes"] == L)
        net, ses = ftp_session([{"cut": at(0.3), "how": "426"}, {"cut": at(0.6), "how": "eof"}])
        dest = TMP / "j" / "b.zip"
        r = ses.download("jpegs/X/1.zip", dest, expected_size=L, checksums=sums)
        check("a 426 (new login) and a 226 short of the size (same login) both resume", r["resumes"] == 2
              and FakeFTP.retrs == [None, at(0.3), at(0.6)] and len(FakeFTP.logins) == 2
              and dest.read_bytes() == DATA, (FakeFTP.retrs, FakeFTP.logins))
        net, ses = ftp_session([{"cut": at(0.3), "how": "reset"}, {"rest_refused": True}])
        dest = TMP / "j" / "c.zip"
        r = ses.download("jpegs/X/1.zip", dest, expected_size=L, checksums=sums)
        check("a refused REST restarts from zero once; exact", r["resumes"] == 2
              and FakeFTP.retrs == [None, at(0.3), None] and dest.read_bytes() == DATA, FakeFTP.retrs)
        net, ses = ftp_session([{"cut": at(0.3), "how": "reset"}] + [{"cut": 0, "how": "reset"}] * 6,
                               resume_attempts=2)
        dest = TMP / "j" / "d.zip"
        e = raises(lambda: ses.download("jpegs/X/1.zip", dest, expected_size=L, checksums=sums), ProviderError)
        check("a transfer that never completes fails after resume_attempts resumes, leaving nothing",
              e is not None and len(FakeFTP.retrs) == 3 and left_nothing(dest), (e, FakeFTP.retrs))
        net, ses = ftp_session([{"perm": True}])
        dest = TMP / "j" / "e.zip"
        e = raises(lambda: ses.download("jpegs/X/1.zip", dest, expected_size=L), ProviderError)
        check("a refused RETR is a ProviderError (a fetch_failed event, not a crash), leaving nothing",
              e is not None and "550" in str(e) and left_nothing(dest), e)
        for name, err in (("refused (530)", ftplib.error_perm("530 Login incorrect.")),
                          ("broken off (EOFError)", EOFError("connection closed"))):
            e = raises(lambda: ftp_session([], login_plan=[err]), Exception)
            check("a first login %s is a ProviderError (not a raw ftplib error fetch does not catch), and its "
                  "connection is closed" % name, isinstance(e, ProviderError) and "login" in str(e)
                  and len(FakeFTP.instances) == 1 and FakeFTP.instances[0].closed, (repr(e), FakeFTP.instances))
        net, ses = ftp_session([{"cut": at(0.3), "how": "reset"}],
                               login_plan=[None, ConnectionResetError(104, "Connection reset by peer"), None])
        dest = TMP / "j" / "f.zip"
        r = ses.download("jpegs/X/1.zip", dest, expected_size=L, checksums=sums)
        check("a login of a resume that fails on a network error closes its connection (no leak); the next resume "
              "logs in again; exact", r["resumes"] == 2 and len(FakeFTP.logins) == 3
              and [c.closed for c in FakeFTP.instances] == [True, True, False]
              and FakeFTP.retrs == [None, at(0.3)] and dest.read_bytes() == DATA,
              (r, FakeFTP.retrs, [c.closed for c in FakeFTP.instances]))
    finally:
        ftplib.FTP = real


def test_disk_errors():
    from weed_optimizer_framework.tools.collect import ChecksumMismatch, ProviderError
    print("(l) a local write error is not a network error")
    dest = TMP / "l" / "f.zip"
    net = server([{}])
    with flaky_disk(2):
        e = raises(lambda: net.download(URL, dest, expected_size=L, checksums={"sha256": sha(DATA)}), Exception)
    check("HTTP: ENOSPC on the 2nd write is a ProviderError at once (no resume), leaving no file and no .tmp",
          type(e) is ProviderError and "No space left" in str(e) and net.log == [(None, 200)] and left_nothing(dest),
          (repr(e), net.log))
    real = ftplib.FTP
    ftplib.FTP = FakeFTP
    try:
        net, ses = ftp_session([{}])
        dest = TMP / "l" / "g.zip"
        with flaky_disk(2):
            e = raises(lambda: ses.download("jpegs/X/1.zip", dest, expected_size=L,
                                            checksums={"sha512": sha(DATA, "sha512")}), Exception)
        check("FTP: ENOSPC on the 2nd write is a ProviderError, not a broken transfer (no new login, no REST), "
              "leaving nothing", type(e) is ProviderError and "No space left" in str(e) and FakeFTP.retrs == [None]
              and len(FakeFTP.logins) == 1 and left_nothing(dest), (repr(e), FakeFTP.retrs))
    finally:
        ftplib.FTP = real
    dest = TMP / "l" / "h.zip"
    net = server([{}])
    with flaky_disk(2, "lost"):
        e = raises(lambda: net.download(URL, dest, expected_size=L, checksums={"sha256": sha(DATA)}), Exception)
    check("a chunk that never reached the disk fails the on-disk size check although the streamed sha256 is the "
          "published one, leaving nothing", isinstance(e, ChecksumMismatch) and "on disk" in str(e)
          and left_nothing(dest), repr(e))
    dest = TMP / "l" / "i.zip"
    net = server([{}])
    with flaky_disk(0, "flush"):
        e = raises(lambda: net.download(URL, dest, expected_size=L), Exception)
    check("a write error reported only at the final flush is a ProviderError, leaving nothing",
          type(e) is ProviderError and "Disk quota" in str(e) and left_nothing(dest), repr(e))


def zenodo_record(rid, first):
    """(net, candidates path) of a Zenodo record whose one file, big.zip, is
    DATA: the first answer sends its first `first` bytes (Content-Length L), a
    Range is answered with a 206 of the rest."""
    furl = "https://zenodo.org/api/records/%s/files/big.zip/content" % rid
    rec = {"id": rid, "metadata": {"title": "Plant boxes %s" % rid, "description": "Palmer amaranth bounding boxes",
                                   "license": {"id": "cc-by-4.0"}},
           "files": [{"key": "big.zip", "size": L, "checksum": "md5:%s" % sha(DATA, "md5"),
                      "links": {"self": furl}}]}

    def serve(params, data, headers):
        rng = headers.get("Range")
        if not rng:
            return 200, DATA[:first], {"Content-Length": str(L)}
        n = int(rng.split("=")[1].rstrip("-"))
        return 206, DATA[n:], {"Content-Range": "bytes %d-%d/%d" % (n, L - 1, L), "Content-Length": str(L - n)}
    net = W.make_net({("GET", "https://zenodo.org/api/records/%s" % rid): (200, rec), ("GET", furl): serve})
    pauses = []
    net._pause = pauses.append
    cpath = TMP / "lab" / ("cands_%s.json" % rid)
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps({"format": "collect-candidates/1", "candidates": [
        {"source_id": "zenodo_%s" % rid, "provider": "zenodo", "ref": rid, "title": "Plant boxes %s" % rid}]}))
    return net, cpath


def test_through_fetch(cfg):
    from weed_optimizer_framework.tools.collect import Refusal
    from weed_optimizer_framework.tools.collect import fetch as F
    from weed_optimizer_framework.tools.collect import staging_dir
    print("(k) through fetch (L16)")
    rid = "4242"
    net, cpath = zenodo_record(rid, at(0.3))
    r = F.fetch(cfg, "zenodo_%s" % rid, candidates_path=cpath, net=net)
    sd = staging_dir("zenodo_%s" % rid)
    doc = json.loads((sd / "fetch.json").read_text())
    f = doc["files"][0]
    check("a fetch whose stream broke at 30% completes", r["status"] == "fetched" and r["bytes"] == L
          and (sd / "blobs" / sha(DATA)).read_bytes() == DATA, r)
    check("fetch.json records the file's resumes, outside its checksums", f["resumes"] == 1
          and set(f["checksums"]) == {"sha256", "md5"} and f["checksums"]["md5"] == sha(DATA, "md5"), f)
    rid = "4243"
    net, cpath = zenodo_record(rid, L)
    with flaky_disk(2):
        e = raises(lambda: F.fetch(cfg, "zenodo_%s" % rid, candidates_path=cpath, net=net), Refusal)
    sd = staging_dir("zenodo_%s" % rid)
    left = sorted(p.name for p in (sd / "blobs").iterdir()) if (sd / "blobs").is_dir() else []
    check("a full disk (ENOSPC on the 2nd write) is a download_failed refusal that leaves no blob (no "
          "blobs/<published sha256>), no .incoming file and no fetch.json", e is not None
          and e.code == "download_failed" and "No space left" in e.detail and left == []
          and not (sd / "fetch.json").exists(), (e and (e.code, e.detail), left))


def main():
    try:
        cfg = W.config()
        W.build_cache()
        from weed_optimizer_framework.tools.collect import fetch as F
        F.disk_free = lambda p: 10 ** 13               # the test machine's free space is not the subject here
        test_one_drop()
        test_several_drops()
        test_range_ignored()
        test_refused_resumes()
        test_budget()
        test_cap()
        test_mismatch()
        test_deadline()
        test_unchanged()
        test_ftp()
        test_through_fetch(cfg)
        test_disk_errors()
    finally:
        W.cleanup()
    print("\n%d failure(s)" % len(W.FAILURES))
    return 1 if W.FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
