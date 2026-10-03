"""The collector's network: HTTP through urllib and FTP through ftplib
(docs/CONTINUOUS_LOOP.md §7.2: a byte cap, a timeout scaled to size and
checksum verification where the provider publishes one).

Every provider calls only Net's methods, so tests replace the two primitives
(Net._open for HTTP, Net.ftp for FTP) and exercise the same caps and checks:
  request / get_json / post   small bodies, read whole; HTTP errors are
                              returned as a status (429 and 5xx retried with
                              backoff, config download.attempts times)
  download                    streamed to <dest>.tmp in 1 MiB chunks, hashed
                              on the fly (sha256 always, plus the algorithm a
                              provider publishes: md5, sha1 as a git blob,
                              sha512, sha256), aborted past max_bytes or past
                              the deadline base_timeout_s + size /
                              min_rate_bytes_per_s, then checked against the
                              expected size and checksum and moved into place
                              atomically; a mismatch leaves nothing behind.
A stream that breaks off is resumed, not started over (Net.download and
FtpSession.download). Servers do close long streams early: on 2026-10-02 a
49.7 GB archive ended after 9.1 GB, the size check refused it and all 9.1 GB
were deleted. So when a body ends before the size the listing or the answer
gives (Content-Length, Content-Range; with no size known and a total of
"*", the 206's own end), or a read raises a network error, the partial .tmp
and the running hashes are kept and the rest is asked for with
"Range: bytes=<n>-" (FTP: REST <n>, after a new login if the connection
died):
  * only a 206 whose Content-Range starts at n continues the file, and only
    when its total is the known size (a total of "*" is taken: the final
    checks still decide) and, when both answers carry a strong ETag, the
    ETag is the first answer's (a changed file is never spliced);
  * a 200 to a Range request (the server ignored it: the whole file again),
    a 416, a Content-Range that does not continue the file or a changed
    ETag (FTP: a refused REST) restarts the .tmp and the hashes from zero
    once; a second one fails, so a server that never honours a Range moves
    the file about twice, not once per resume;
  * at most download.resume_attempts (default 20) resumes per file, each
    after a pause of 1 s while the streams make progress, doubling up to
    60 s while they do not; the size-scaled deadline and max_bytes hold for
    the whole file across its resumes (max_bytes bounds the bytes kept; the
    one restart from zero bounds the bytes moved);
  * only a read or a transfer is a network error. Writing the .tmp is not:
    a full disk, a quota or an I/O error fails the download at once (a
    ProviderError, nothing left), and the .tmp's size on disk must be the
    bytes counted, so a chunk that never reached the disk can never pass
    under the published checksum;
  * the final size and checksum checks are unchanged, so a resumed file is
    accepted only if it is byte for byte the published one; the result
    records "resumes", the follow-up requests the file took.
Secrets (API keys, tokens) are sent in headers or parameters and never
written: every URL that reaches a record is redacted (redact()).
"""
from __future__ import annotations

import ftplib
import hashlib
import http.client
import http.cookiejar
import io
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import ByteCapExceeded, ChecksumMismatch, CredentialsMissing, ProviderError

USER_AGENT = "collect-intake/1 (research data collection)"
SECRET_PARAMS = ("api_key", "key", "token", "access_token", "password", "signature", "sig", "x-goog-signature",
                 "x-goog-credential", "x-amz-signature", "x-amz-credential", "x-amz-security-token", "awsaccesskeyid",
                 "googleaccessid", "credential", "auth", "authorization")   # signed storage links (export redirects)
CHUNK = 1 << 20
HASH_ALGOS = ("sha256", "md5", "sha1", "sha512", "git-blob-sha1")
RETRY_STATUS = (429, 500, 502, 503, 504)
# What a broken HTTP connection raises: urllib's and the socket's errors (OSError,
# timeouts included) and http.client's (IncompleteRead of a chunked body cut short).
NET_ERRORS = (urllib.error.URLError, OSError, http.client.HTTPException)
# The same for ftplib: a dead control connection (EOFError), a 4xx reply such as
# "426 transfer aborted", a reply out of turn. A 5xx (error_perm) is a refusal.
FTP_NET_ERRORS = (OSError, EOFError, ftplib.error_temp, ftplib.error_reply, ftplib.error_proto)
_CONTENT_RANGE = re.compile(r"^\s*bytes\s+(\d+)-(\d+)/(\d+|\*)\s*$", re.I)


def redact(url, params=None):
    p = {k: ("<redacted>" if k.lower() in SECRET_PARAMS else v) for k, v in sorted((params or {}).items())}
    parsed = urllib.parse.urlsplit(url)
    q = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    q = [(k, "<redacted>" if k.lower() in SECRET_PARAMS else v) for k, v in q]
    netloc = parsed.netloc.split("@")[-1]
    base = urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, urllib.parse.urlencode(q, safe="<>"), ""))
    extra = urllib.parse.urlencode(sorted(p.items()), safe="<>")
    return base + (("&" if "?" in base else "?") + extra if extra else "")


def read_secret(env_names=(), files=()):
    """(value, where) of the first secret found in the environment variables
    or files named, else (None, None). The value is never logged or saved."""
    for e in env_names or ():
        v = (os.environ.get(e) or "").strip()
        if v:
            return v, "env:%s" % e
    for f in files or ():
        p = Path(os.path.expanduser(os.path.expandvars(str(f))))
        try:
            if p.is_file():
                v = p.read_text(encoding="utf-8", errors="ignore").strip()
                if v:
                    return v, "file:%s" % f
        except OSError:
            continue
    return None, None


def _header(hd, name):
    """A header of an answer's header dict, whatever the case of its name."""
    name = name.lower()
    for k, v in (hd or {}).items():
        if str(k).lower() == name:
            return v
    return None


def _int_header(hd, name):
    v = str(_header(hd, name) or "").strip()
    return int(v) if v.isdigit() else None


def content_range(hd):
    """(first, last, total or None for "*") of an answer's Content-Range, else None."""
    m = _CONTENT_RANGE.match(str(_header(hd, "content-range") or ""))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), None if m.group(3) == "*" else int(m.group(3))


def _strong_etag(hd):
    """The answer's ETag when it is a strong one (a weak W/ tag may name other bytes)."""
    e = str(_header(hd, "etag") or "").strip()
    return e if e and not e.startswith("W/") else None


def _close(body):
    try:
        body.close()
    except Exception:  # noqa: BLE001
        pass


class _Sink(object):
    """A file being downloaded: counts, hashes and caps every chunk."""

    def __init__(self, dest, max_bytes=None, deadline=None, algos=("sha256",), what=""):
        self.dest = Path(dest)
        self.tmp = self.dest.with_name(self.dest.name + ".tmp")
        self.dest.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.tmp, "wb")
        self.n = 0
        self.seen = 0           # every byte written, across restarts: the progress a resume measures
        self.max_bytes = max_bytes
        self.deadline = deadline
        self.what = what
        self.h = {}
        for a in set(algos) | {"sha256"}:
            if a == "git-blob-sha1":
                continue
            self.h[a] = hashlib.new(a)

    def restart(self):
        """Back to zero bytes: the .tmp and the running hashes start over (the
        server sent the whole file again, or refused a resume)."""
        self.fh.seek(0)
        self.fh.truncate()
        self.n = 0
        self.h = {a: hashlib.new(a) for a in self.h}

    def _write_failed(self, e):
        """A local write error (a full disk, a quota, EIO) as a ProviderError:
        not a network error, so it is never resumed (module docstring)."""
        return ProviderError("%s: writing %s failed (%s: %s)" % (self.what, self.tmp.name, type(e).__name__, e))

    def write(self, chunk):
        if self.max_bytes is not None and self.n + len(chunk) > self.max_bytes:
            raise ByteCapExceeded("%s: more than the %d-byte cap" % (self.what, self.max_bytes))
        if self.deadline is not None and time.time() > self.deadline:
            raise ProviderError("%s: the download passed its size-scaled deadline" % self.what)
        try:
            self.fh.write(chunk)
        except OSError as e:
            raise self._write_failed(e)
        # counted and hashed only once the file holds it: a chunk that never reached the .tmp is not in n
        self.n += len(chunk)
        self.seen += len(chunk)
        for h in self.h.values():
            h.update(chunk)

    def abort(self):
        _close(self.fh)         # a buffered write that fails on close does not matter: the .tmp goes
        try:
            os.unlink(self.tmp)
        except OSError:
            pass

    def finish(self, expected_size=None, checksums=None):
        try:
            try:
                self.fh.flush()
                os.fsync(self.fh.fileno())
                on_disk = os.fstat(self.fh.fileno()).st_size
            except OSError as e:
                raise self._write_failed(e)
            finally:
                _close(self.fh)
            if on_disk != self.n:
                raise ChecksumMismatch("%s: %d bytes on disk, %d bytes streamed and hashed"
                                       % (self.what, on_disk, self.n))
            if expected_size is not None and int(expected_size) != self.n:
                raise ChecksumMismatch("%s: %d bytes, the provider lists %d" % (self.what, self.n, int(expected_size)))
            digests = {a: h.hexdigest() for a, h in self.h.items()}
            for algo, want in (checksums or {}).items():
                if algo == "git-blob-sha1":
                    got = _git_blob_sha1(self.tmp, self.n)
                else:
                    got = digests.get(algo)
                if got is None or str(want).lower() != got:
                    raise ChecksumMismatch("%s: %s %s, the provider publishes %s"
                                           % (self.what, algo, (got or "none")[:12], str(want)[:12]))
                digests[algo] = got
            os.replace(self.tmp, self.dest)
        except BaseException:
            try:
                os.unlink(self.tmp)
            except OSError:
                pass
            raise
        return dict(digests, bytes=self.n)


def _git_blob_sha1(path, n):
    h = hashlib.sha1(("blob %d\0" % n).encode("ascii"))
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


class _Resumes(object):
    """The resume budget of one file (module docstring): at most
    net.resume_attempts resumes, a pause before each (1 s after a stream that
    made progress, doubling up to 60 s while none does) that may not cross
    the file's deadline, and one restart from zero after a refused or
    ignored resume."""

    def __init__(self, net, sink, what):
        self.net, self.sink, self.what = net, sink, what
        self.count = 0
        self.stalls = 0
        self.restarted = False
        self._seen = sink.seen

    def wait(self, why):
        """Pause before the next resume, or raise ProviderError when the
        budget or the deadline is spent (the caller aborts the sink)."""
        s = self.sink
        self.stalls = 0 if s.seen > self._seen else self.stalls + 1
        self._seen = s.seen
        if self.count >= self.net.resume_attempts:
            raise ProviderError("%s: gave up at %d bytes after %d resume(s), the download.resume_attempts limit (%s)"
                                % (self.what, s.n, self.count, why))
        pause = float(min(2 ** self.stalls, 60)) if self.stalls else 1.0
        if s.deadline is not None and time.time() + pause > s.deadline:
            raise ProviderError("%s: no time left before the size-scaled deadline to resume at %d bytes (%s)"
                                % (self.what, s.n, why))
        self.net._pause(pause)
        self.count += 1

    def restart(self, why):
        """The one restart from zero a refused (or ignored) resume gets; a second fails."""
        if self.restarted:
            raise ProviderError("%s: a resume was refused again after the restart from zero (%s)" % (self.what, why))
        self.restarted = True
        self.sink.restart()


class Net(object):
    """HTTP and FTP with the collector's caps (module docstring)."""

    def __init__(self, download_cfg=None, headers=None):
        d = dict(download_cfg or {})
        self.base_timeout = float(d.get("base_timeout_s", 120))
        self.min_rate = float(d.get("min_rate_bytes_per_s", 1e6))
        self.read_timeout = float(d.get("read_timeout_s", 60))
        self.attempts = int(d.get("attempts", 3))
        self.resume_attempts = int(d.get("resume_attempts", 20))
        self.headers = dict({"User-Agent": USER_AGENT}, **(headers or {}))
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar))
        self.calls = []

    def cookie(self, name):
        """The value of a cookie a server set during this session, else None."""
        for c in self.jar:
            if c.name == name:
                return c.value
        return None

    # ---- deadlines
    def deadline_for(self, size):
        s = float(size) if size else 0.0
        return time.time() + self.base_timeout + s / self.min_rate

    def _pause(self, seconds):
        """The wait between the tries of a download (tests replace it)."""
        time.sleep(seconds)

    # ---- the HTTP primitive (tests replace it)
    def _open(self, url, params=None, headers=None, data=None, method=None, timeout=None):
        """(status, headers dict, readable body). HTTP errors are returned."""
        q = urllib.parse.urlencode(sorted((params or {}).items()))
        full = url + (("&" if "?" in url else "?") + q if q else "")
        req = urllib.request.Request(full, data=data, headers=dict(self.headers, **(headers or {})),
                                     method=method)
        try:
            r = self.opener.open(req, timeout=timeout or self.read_timeout)
            return r.status, dict(r.headers.items()), r
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers.items()) if e.headers else {}, io.BytesIO(e.read() or b"")

    def request(self, url, params=None, headers=None, data=None, method=None):
        """(status, body bytes, headers) with retries on 429 and 5xx and on
        network failures; the last failure raises ProviderError."""
        last = None
        for attempt in range(self.attempts):
            try:
                st, hd, body = self._open(url, params, headers, data, method)
                raw = body.read() if hasattr(body, "read") else (body or b"")
                try:
                    body.close()
                except Exception:  # noqa: BLE001
                    pass
            except (urllib.error.URLError, OSError) as e:
                last = e
                time.sleep(min(2 ** attempt, 8))
                continue
            self.calls.append((redact(url, params), st))
            if st in (429, 500, 502, 503, 504) and attempt + 1 < self.attempts:
                time.sleep(min(2 ** attempt * (2 if st == 429 else 1), 16))
                continue
            return st, raw or b"", hd or {}
        raise ProviderError("GET %s failed after %d attempts (%s)" % (redact(url, params), self.attempts, last))

    def get_json(self, url, params=None, headers=None, what=None, data=None, method=None):
        """(status, parsed JSON or None, sha256 of the body, headers)."""
        st, raw, hd = self.request(url, params, headers, data, method)
        sha = hashlib.sha256(raw).hexdigest()
        if st not in (200, 206):
            return st, None, sha, hd
        try:
            return st, json.loads(raw.decode("utf-8")), sha, hd
        except (UnicodeDecodeError, ValueError) as e:
            raise ProviderError("%s is not JSON (%s)" % (what or redact(url, params), e))

    def download(self, url, dest, params=None, headers=None, max_bytes=None, expected_size=None, checksums=None,
                 what=None):
        """Stream url to dest, resuming a stream that breaks off (module
        docstring). Returns {"bytes", "sha256", other digests, "resumes"}."""
        what = what or redact(url, params)
        algos = tuple(a for a in (checksums or {}) if a in HASH_ALGOS)
        if max_bytes is not None and expected_size is not None and int(expected_size) > int(max_bytes):
            raise ByteCapExceeded("%s: %d bytes listed, over the %d-byte cap" % (what, int(expected_size), max_bytes))
        deadline = self.deadline_for(expected_size or max_bytes or 0)       # one deadline for the whole file
        st, hd, body = self._first_answer(url, params, headers, max_bytes, what)
        sink = _Sink(dest, max_bytes=max_bytes, deadline=deadline, algos=algos, what=what)
        res = _Resumes(self, sink, what)
        want = int(expected_size) if expected_size is not None else None
        etag = None
        try:
            while True:
                # (st, hd, body) continues the file at sink.n: a 200 from zero, or a 206 _resume checked.
                # end: with no size known, the least the file has, a 206 of total "*" names its own end
                end = None
                if st == 200:
                    said = _int_header(hd, "content-length")
                    etag = _strong_etag(hd)
                    if expected_size is None:
                        want = said
                else:
                    cr = content_range(hd)
                    said = cr[2]
                    if want is None:
                        want = said
                    if said is None:
                        cl = _int_header(hd, "content-length")
                        end = max(cr[1] + 1, sink.n + cl if cl is not None else 0)
                err = self._drain(body, sink)
                # the end: the size is reached; or the body ended cleanly with no size known (at a 206's own
                # end when it names one), or after all the answer said the file has (a size that differs from
                # the listing then fails the size check below, as before). A broken stream of unknown size,
                # or a "*" 206 that ends short of its own end, is never taken as whole.
                done = want is not None and sink.n >= want
                if err is None:
                    done = done or (want is None and (end is None or sink.n >= end)) or (said is not None
                                                                                         and sink.n >= said)
                if done:
                    break
                why = err or "the stream ended at %d of %s bytes" % (sink.n, want if want is not None else
                                                                    "at least %d" % end)
                st, hd, body = self._resume(url, params, headers, max_bytes, what, sink, res, why, want, etag)
        except BaseException:
            sink.abort()
            raise
        out = sink.finish(expected_size=want, checksums=checksums)
        out["resumes"] = res.count
        return out

    def _first_answer(self, url, params, headers, max_bytes, what):
        """The first 200 of a download, with retries on 429, 5xx and network
        failures (config download.attempts)."""
        last = None
        for attempt in range(self.attempts):
            try:
                st, hd, body = self._open(url, params, headers, timeout=self.read_timeout)
            except NET_ERRORS as e:
                last = e
                self._pause(min(2 ** attempt, 8))
                continue
            self.calls.append((redact(url, params), st))
            if st != 200:
                _close(body)
                if st in RETRY_STATUS and attempt + 1 < self.attempts:
                    self._pause(min(2 ** attempt, 8))
                    continue
                raise ProviderError("%s answered HTTP %s" % (what, st))
            cl = _int_header(hd, "content-length")
            if max_bytes is not None and cl is not None and cl > max_bytes:
                _close(body)
                raise ByteCapExceeded("%s: Content-Length %s over the %d-byte cap" % (what, cl, max_bytes))
            return st, hd, body
        raise ProviderError("%s failed after %d attempts (%s)" % (what, self.attempts, last))

    @staticmethod
    def _drain(body, sink):
        """Stream a body into the sink until it ends. Returns the network
        error that broke a read off, as text, else None; the sink's own
        refusals (the byte cap, the deadline, a failed write to the .tmp)
        propagate: only the read is a network error."""
        try:
            while True:
                try:
                    chunk = body.read(CHUNK)
                except NET_ERRORS as e:
                    return "%s: %s" % (type(e).__name__, e)
                if not chunk:
                    return None
                sink.write(chunk)
        finally:
            _close(body)

    def _resume(self, url, params, headers, max_bytes, what, sink, res, why, want, etag):
        """The next answer that continues the file at sink.n (module
        docstring), each try after res.wait(); a 200 to a Range request
        restarts the sink, as the file's one restart from zero."""
        while True:
            res.wait(why)
            h = dict(headers or {})
            if sink.n:
                h["Range"] = "bytes=%d-" % sink.n
            try:
                st, hd, body = self._open(url, params, h, timeout=self.read_timeout)
            except NET_ERRORS as e:
                why = "%s: %s" % (type(e).__name__, e)
                continue
            self.calls.append((redact(url, params), st))
            if st == 200:
                cl = _int_header(hd, "content-length")
                if max_bytes is not None and cl is not None and cl > max_bytes:
                    _close(body)
                    raise ByteCapExceeded("%s: Content-Length %s over the %d-byte cap" % (what, cl, max_bytes))
                if sink.n:                  # the server ignored the Range: this is the whole file again
                    try:
                        res.restart("HTTP 200 to bytes=%d-: the server ignored the Range" % sink.n)
                    except BaseException:
                        _close(body)
                        raise
                return st, hd, body
            if st == 206:
                cr = content_range(hd)
                tag = _strong_etag(hd)
                if cr is None:
                    why = "a 206 without a Content-Range"
                elif cr[0] != sink.n:
                    why = "Content-Range %s for bytes=%d-" % (_header(hd, "content-range"), sink.n)
                elif want is not None and cr[2] is not None and cr[2] != want:
                    why = "Content-Range total %d, not %d" % (cr[2], want)
                elif etag and tag and tag != etag:
                    why = "the ETag changed from %s to %s" % (etag, tag)
                elif max_bytes is not None and cr[2] is not None and cr[2] > max_bytes:
                    _close(body)
                    raise ByteCapExceeded("%s: Content-Range total %d over the %d-byte cap" % (what, cr[2], max_bytes))
                else:
                    return st, hd, body
                _close(body)
                res.restart(why)
                continue
            _close(body)
            if st == 416:
                why = "HTTP 416 for bytes=%d-" % sink.n
                res.restart(why)
            elif st in RETRY_STATUS:
                why = "HTTP %s" % st
            else:
                raise ProviderError("%s answered HTTP %s to a resume at %d bytes" % (what, st, sink.n))

    # ---- FTP (tests replace ftp())
    def ftp(self, host, user="anonymous", password="", timeout=None):
        return FtpSession(host, user, password, timeout or self.read_timeout, self)


class FtpSession(object):
    """One FTP login (ftplib), listing and streamed, capped downloads."""

    def __init__(self, host, user, password, timeout, net):
        self.net = net
        self.host = host
        self.user, self.password, self.timeout = user, password, timeout
        self.ftp = None
        try:
            self._connect()
        except ftplib.error_perm as e:          # a ProviderError, as for the logins of a resume (download)
            raise ProviderError("ftp://%s: the login was refused (%s)" % (host, e))
        except FTP_NET_ERRORS as e:
            raise ProviderError("ftp://%s: the login failed (%s: %s)" % (host, type(e).__name__, e))

    def _connect(self):
        """Connect and log in; a login that fails closes its connection."""
        ftp = ftplib.FTP(self.host, timeout=self.timeout)
        try:
            ftp.login(self.user, self.password)
        except BaseException:
            try:
                ftp.close()
            except Exception:  # noqa: BLE001
                pass
            raise
        self.ftp = ftp

    def _drop(self):
        """Forget a connection a transfer broke (a resume logs in again)."""
        try:
            self.ftp.close()
        except Exception:  # noqa: BLE001
            pass
        self.ftp = None

    def nlst(self, path):
        from ftplib import error_perm
        try:
            return sorted(self.ftp.nlst(path))
        except error_perm as e:
            raise ProviderError("ftp://%s/%s: %s" % (self.host, path, e))

    def size(self, path):
        self.ftp.voidcmd("TYPE I")
        try:
            return int(self.ftp.size(path))
        except Exception as e:  # noqa: BLE001
            raise ProviderError("ftp://%s/%s: no size (%s)" % (self.host, path, e))

    def read(self, path, max_bytes=None):
        buf = bytearray()

        def cb(b):
            buf.extend(b)
            if max_bytes is not None and len(buf) > max_bytes:
                raise ByteCapExceeded("ftp://%s/%s: more than %d bytes" % (self.host, path, max_bytes))
        self.ftp.retrbinary("RETR %s" % path, cb)
        return bytes(buf)

    def download(self, path, dest, max_bytes=None, expected_size=None, checksums=None):
        """Stream path to dest through the sink. A transfer that breaks off
        (a network error, a 4xx reply, an end short of expected_size) resumes
        with REST at the bytes kept, after a new login when the connection
        died, on Net.download's budget (module docstring); a REST the server
        refuses (5xx) restarts from zero once. A refused RETR or login is a
        ProviderError, and so is a failed write to the .tmp: the callback
        (sink.write) raises it, never an OSError, which FTP_NET_ERRORS would
        take for a broken transfer. Returns {"bytes", "sha256", digests,
        "resumes"}."""
        what = "ftp://%s/%s" % (self.host, path)
        if max_bytes is not None and expected_size is not None and int(expected_size) > int(max_bytes):
            raise ByteCapExceeded("%s: %d bytes listed, over the %d-byte cap" % (what, int(expected_size), max_bytes))
        sink = _Sink(dest, max_bytes=max_bytes, deadline=self.net.deadline_for(expected_size or max_bytes or 0),
                     algos=tuple(checksums or {}), what=what)
        res = _Resumes(self.net, sink, what)
        why = None
        try:
            while True:
                if why is not None:
                    res.wait(why)
                if self.ftp is None:
                    try:
                        self._connect()
                    except ftplib.error_perm as e:
                        raise ProviderError("%s: the login was refused (%s)" % (what, e))
                    except FTP_NET_ERRORS as e:
                        self.ftp = None
                        why = "logging in again: %s: %s" % (type(e).__name__, e)
                        continue
                rest = sink.n
                try:
                    # sink.write turns a local write error into a ProviderError, outside FTP_NET_ERRORS
                    self.ftp.retrbinary("RETR %s" % path, sink.write, blocksize=CHUNK, rest=rest or None)
                except ftplib.error_perm as e:
                    if not rest:
                        raise ProviderError("%s: %s" % (what, e))
                    why = "REST %d refused (%s)" % (rest, e)
                    res.restart(why)
                    continue
                except FTP_NET_ERRORS as e:
                    why = "%s: %s" % (type(e).__name__, e)
                    self._drop()
                    continue
                if expected_size is None or sink.n >= int(expected_size):
                    break
                why = "the transfer ended at %d of %d bytes" % (sink.n, int(expected_size))
        except BaseException:
            sink.abort()
            raise
        out = sink.finish(expected_size=expected_size, checksums=checksums)
        out["resumes"] = res.count
        return out

    def close(self):
        try:
            self.ftp.quit()
        except Exception:  # noqa: BLE001
            pass


def need_secret(env_names, files, provider):
    v, where = read_secret(env_names, files)
    if not v:
        raise CredentialsMissing("provider %s needs a credential in %s" % (
            provider, ", ".join(["$%s" % e for e in env_names or ()] + [str(f) for f in files or ()]) or "?"))
    return v, where
