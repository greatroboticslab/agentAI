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
Secrets (API keys, tokens) are sent in headers or parameters and never
written: every URL that reaches a record is redacted (redact()).
"""
from __future__ import annotations

import hashlib
import http.cookiejar
import io
import json
import os
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


class _Sink(object):
    """A file being downloaded: counts, hashes and caps every chunk."""

    def __init__(self, dest, max_bytes=None, deadline=None, algos=("sha256",), what=""):
        self.dest = Path(dest)
        self.tmp = self.dest.with_name(self.dest.name + ".tmp")
        self.dest.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.tmp, "wb")
        self.n = 0
        self.max_bytes = max_bytes
        self.deadline = deadline
        self.what = what
        self.h = {}
        for a in set(algos) | {"sha256"}:
            if a == "git-blob-sha1":
                continue
            self.h[a] = hashlib.new(a)

    def write(self, chunk):
        self.n += len(chunk)
        if self.max_bytes is not None and self.n > self.max_bytes:
            raise ByteCapExceeded("%s: more than the %d-byte cap" % (self.what, self.max_bytes))
        if self.deadline is not None and time.time() > self.deadline:
            raise ProviderError("%s: the download passed its size-scaled deadline" % self.what)
        for h in self.h.values():
            h.update(chunk)
        self.fh.write(chunk)

    def abort(self):
        try:
            self.fh.close()
        finally:
            try:
                os.unlink(self.tmp)
            except OSError:
                pass

    def finish(self, expected_size=None, checksums=None):
        self.fh.flush()
        os.fsync(self.fh.fileno())
        self.fh.close()
        try:
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


class Net(object):
    """HTTP and FTP with the collector's caps (module docstring)."""

    def __init__(self, download_cfg=None, headers=None):
        d = dict(download_cfg or {})
        self.base_timeout = float(d.get("base_timeout_s", 120))
        self.min_rate = float(d.get("min_rate_bytes_per_s", 1e6))
        self.read_timeout = float(d.get("read_timeout_s", 60))
        self.attempts = int(d.get("attempts", 3))
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
        """Stream url to dest (module docstring). Returns {"bytes", "sha256",
        other digests}."""
        what = what or redact(url, params)
        algos = tuple(a for a in (checksums or {}) if a in HASH_ALGOS)
        if max_bytes is not None and expected_size is not None and int(expected_size) > int(max_bytes):
            raise ByteCapExceeded("%s: %d bytes listed, over the %d-byte cap" % (what, int(expected_size), max_bytes))
        last = None
        for attempt in range(self.attempts):
            deadline = self.deadline_for(expected_size or max_bytes or 0)
            try:
                st, hd, body = self._open(url, params, headers, timeout=self.read_timeout)
            except (urllib.error.URLError, OSError) as e:
                last = e
                time.sleep(min(2 ** attempt, 8))
                continue
            self.calls.append((redact(url, params), st))
            if st != 200:
                try:
                    body.close()
                except Exception:  # noqa: BLE001
                    pass
                if st in (429, 500, 502, 503, 504) and attempt + 1 < self.attempts:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                raise ProviderError("%s answered HTTP %s" % (what, st))
            cl = {str(k).lower(): v for k, v in (hd or {}).items()}.get("content-length")
            if max_bytes is not None and cl and str(cl).isdigit() and int(cl) > max_bytes:
                body.close()
                raise ByteCapExceeded("%s: Content-Length %s over the %d-byte cap" % (what, cl, max_bytes))
            sink = _Sink(dest, max_bytes=max_bytes, deadline=deadline, algos=algos, what=what)
            try:
                for chunk in iter(lambda: body.read(CHUNK), b""):
                    sink.write(chunk)
                body.close()
            except (ByteCapExceeded, ProviderError):
                sink.abort()
                raise
            except (urllib.error.URLError, OSError) as e:
                sink.abort()
                last = e
                time.sleep(min(2 ** attempt, 8))
                continue
            size = expected_size if expected_size is not None else (int(cl) if cl and str(cl).isdigit() else None)
            return sink.finish(expected_size=size, checksums=checksums)
        raise ProviderError("%s failed after %d attempts (%s)" % (what, self.attempts, last))

    # ---- FTP (tests replace ftp())
    def ftp(self, host, user="anonymous", password="", timeout=None):
        return FtpSession(host, user, password, timeout or self.read_timeout, self)


class FtpSession(object):
    """One FTP login (ftplib), listing and streamed, capped downloads."""

    def __init__(self, host, user, password, timeout, net):
        from ftplib import FTP
        self.net = net
        self.host = host
        self.ftp = FTP(host, timeout=timeout)
        self.ftp.login(user, password)

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
        what = "ftp://%s/%s" % (self.host, path)
        if max_bytes is not None and expected_size is not None and int(expected_size) > int(max_bytes):
            raise ByteCapExceeded("%s: %d bytes listed, over the %d-byte cap" % (what, int(expected_size), max_bytes))
        sink = _Sink(dest, max_bytes=max_bytes, deadline=self.net.deadline_for(expected_size or max_bytes or 0),
                     algos=tuple(checksums or {}), what=what)
        try:
            self.ftp.retrbinary("RETR %s" % path, sink.write, blocksize=CHUNK)
        except BaseException:
            sink.abort()
            raise
        return sink.finish(expected_size=expected_size, checksums=checksums)

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
