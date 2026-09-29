#!/usr/bin/env python3
"""A small world for the collector tests (group D of docs/CONTINUOUS_LOOP.md §9),
and, run as a script, a check that the world is what the tests assume.

    import test_collect_world as W
    TMP = W.setup("collect_x_")          # BEFORE any weed_optimizer_framework import
    ...
    cfg = W.config()                     # collect/domains/weed.json, loaded
    W.build_cache(W.CACHE_NAMES)         # the funnel taxonomy cache, from the recorded GBIF answers

What it provides:
  * setup: a temporary INC_DIR and REPO (the funnel prereg and its contract
    copied in, so funnel.taxonomy.build_cache runs offline through
    tests/fixtures/funnel/gbif_recording.json), a registry path of its own
    (COLLECT_REGISTRY), and the network blocked (every call must go through
    a fake);
  * grid_img: pictures whose dHashes are far apart for different seeds and a
    bit or two apart for a painted near copy (tests/test_inc_step0_fixes.py's
    construction);
  * FakeNet: transport.Net with its HTTP primitive (_open) and FTP replaced by
    routes; downloads still run through transport's capped, checked sink;
  * FakeFtp: an FTP server of named files;
  * FakeGuard: inc2.guard.GuardV2's interface (check_path, add_intake,
    index_record, counts) over planted evaluation and base images, with the
    same order of checks (unhashable, near_eval_v2, near_eval_variant,
    base_copy, near_dup_intake);
  * zip_bytes, coco and yolo helpers.

Run:  python3 tests/test_collect_world.py
"""
import collections
import io
import json
import os
import pathlib
import random
import shutil
import socket
import sys
import tempfile
import zipfile
import funnel_prereg as FPR  # noqa: E402

TESTS = pathlib.Path(__file__).resolve().parent
ROOT = TESTS.parent
GIT = ROOT.parent
TMP = None
FAILURES = []

# names the funnel taxonomy cache is built with (all present in the GBIF recording)
CACHE_NAMES = ["Portulaca oleracea", "Chenopodium album", "Carpetweed", "Crabgrass", "Cassia obtusifolia",
               "Ipomoea obscura", "Achillea millefolium", "Amaranthus palmeri", "Taraxacum officinale",
               "Morningglory", "Kochia", "Ragweed", "Sicklepod", "Waterhemp", "Redroot Pigweed", "Nutsedge",
               "weed", "crop", "0", "5", "12", "Lambsquarters", "Soybean", "Canola"]


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, str(detail)[:1500]))
        FAILURES.append(name)


def raises(fn, exc):
    try:
        fn()
    except exc as e:
        return e
    return None


class _NoNet(socket.socket):
    def __init__(self, *a, **k):
        raise AssertionError("the test touched the network")


def setup(prefix):
    """The temporary world; call before importing weed_optimizer_framework."""
    global TMP
    TMP = pathlib.Path(tempfile.mkdtemp(prefix=prefix))
    os.environ["INC_DIR"] = str(TMP / "inc")
    os.environ["REPO"] = str(TMP / "repo")
    os.environ["COLLECT_REGISTRY"] = str(TMP / "repo" / "results" / "framework" / "dataset_registry.json")
    os.environ.pop("FUNNEL_GBIF_RECORD", None)
    os.environ.pop("SLURM_JOB_ID", None)
    for k in ("KAGGLE_API_TOKEN", "ROBOFLOW_API_KEY", "HF_TOKEN", "GITHUB_TOKEN"):
        os.environ.pop(k, None)
    os.environ["HOME"] = str(TMP / "home")               # no key file of the real home is read
    (TMP / "home").mkdir(parents=True)
    (TMP / "repo" / "docs").mkdir(parents=True)
    (TMP / "inc" / "funnel").mkdir(parents=True)
    shutil.copyfile(GIT / "docs" / "FUNNEL_AUDIT.md", TMP / "repo" / "docs" / "FUNNEL_AUDIT.md")
    FPR.write_pre_draw(TMP / "inc" / "funnel" / "prereg_v1.json",
                       ROOT / "results" / "framework" / "inc" / "funnel" / "prereg_v1.json")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(TESTS))
    socket.socket = _NoNet
    return TMP


def cleanup():
    if TMP is not None:
        shutil.rmtree(str(TMP), ignore_errors=True)


def config():
    """The weed collector config. The licence fallback (license_audit, a live
    lookup) answers "unreachable" in every test unless a test sets its own."""
    from weed_optimizer_framework.tools.collect import config as CF
    from weed_optimizer_framework.tools.collect import licence as LIC
    if LIC.DETECT is None:
        LIC.DETECT = lambda slug, info: {"license": "unreachable", "license_source": "test:no-network"}
    return CF.load("weed")


def build_cache(names=None):
    import funnel_world as FWD
    return FWD.build_taxonomy_cache(TMP / "inc" / "funnel", list(names or CACHE_NAMES))


# ------------------------------------------------------------------ images
def grid_img(path, seed, paint=(), size=(144, 128), fmt=None):
    """A 9x8 grid of random grey levels blown up: its dHash is the grid's own
    left-right comparisons (copies 0 bits apart, other seeds ~32 bits apart;
    `paint` rows move it by a bit or two)."""
    from PIL import Image
    rng = random.Random(seed)
    grid = [[rng.randrange(256) for _ in range(9)] for _ in range(8)]
    for r in paint:
        grid[r][8] = 255 if grid[r][8] <= grid[r][7] else 0
    im = Image.new("L", (9, 8))
    im.putdata([v for row in grid for v in row])
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.resize(size, Image.NEAREST).convert("RGB").save(path, format=fmt, quality=95)
    return path


def img_bytes(seed, paint=(), size=(144, 128)):
    p = TMP / "scratch_img" / ("%s_%s.png" % (seed, "_".join(map(str, paint))))
    grid_img(p, seed, paint=paint, size=size, fmt="PNG")
    return p.read_bytes()


def zip_bytes(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(files):
            zf.writestr(name, files[name])
    return buf.getvalue()


# ------------------------------------------------------------------ network
class FakeFtp(object):
    """An FTP server: {path: bytes}; directories are implied by paths."""

    def __init__(self, files):
        self.files = dict(files)
        self.logins = []


class _FakeFtpSession(object):
    def __init__(self, server, net, host):
        self.server, self.net, self.host = server, net, host

    def nlst(self, path):
        from weed_optimizer_framework.tools.collect import ProviderError
        p = path.rstrip("/") + "/"
        kids = set()
        for f in self.server.files:
            if f.startswith(p):
                rest = f[len(p):].split("/")
                kids.add(p + rest[0])
        if not kids:
            raise ProviderError("ftp://%s/%s: no such directory" % (self.host, path))
        return sorted(kids)

    def size(self, path):
        from weed_optimizer_framework.tools.collect import ProviderError
        if path not in self.server.files:
            raise ProviderError("no size for %s" % path)
        return len(self.server.files[path])

    def read(self, path, max_bytes=None):
        return self.server.files[path]

    def download(self, path, dest, max_bytes=None, expected_size=None, checksums=None):
        from weed_optimizer_framework.tools.collect.transport import _Sink
        data = self.server.files[path]
        sink = _Sink(dest, max_bytes=max_bytes, deadline=None, algos=tuple(checksums or {}), what=path)
        try:
            for i in range(0, len(data), 7):
                sink.write(data[i:i + 7])
        except BaseException:
            sink.abort()
            raise
        return sink.finish(expected_size=expected_size, checksums=checksums)

    def close(self):
        pass


def make_net(routes=None, ftp=None):
    from weed_optimizer_framework.tools.collect.transport import Net

    class FakeNet(Net):
        """Net with routes: {(METHOD, url): handler or (status, body[, headers])}.
        A handler gets (params, data, headers) and returns (status, body[, headers]).
        Unknown routes answer 404."""

        def __init__(self):
            super().__init__({"base_timeout_s": 30, "min_rate_bytes_per_s": 1e9, "read_timeout_s": 5,
                              "attempts": 1})
            self.routes = dict(routes or {})
            self.ftp_servers = dict(ftp or {})
            self.cookies = {}
            self.log = []

        def route(self, url, answer, method="GET"):
            self.routes[(method, url)] = answer

        def _open(self, url, params=None, headers=None, data=None, method=None, timeout=None):
            m = method or ("POST" if data is not None else "GET")
            self.log.append((m, url, dict(params or {}), dict(headers or {})))
            ans = self.routes.get((m, url))
            if ans is None and m == "HEAD":
                ans = self.routes.get(("GET", url))
                if ans is not None and not callable(ans):
                    ans = (ans[0], b"", dict(ans[2] if len(ans) > 2 else {}, **{"Content-Length": str(len(ans[1]))}))
            if ans is None:
                return 404, {}, io.BytesIO(b"{}")
            if callable(ans):
                ans = ans(dict(params or {}), data, dict(headers or {}))
            st, body = ans[0], ans[1]
            hd = dict(ans[2]) if len(ans) > 2 else {}
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode("utf-8")
            return st, hd, io.BytesIO(body)

        def cookie(self, name):
            return self.cookies.get(name)

        def ftp(self, host, user="anonymous", password="", timeout=None):
            from weed_optimizer_framework.tools.collect import ProviderError
            srv = self.ftp_servers.get(host)
            if srv is None:
                raise ProviderError("ftp://%s unreachable" % host)
            srv.logins.append((user, password))
            return _FakeFtpSession(srv, self, host)

    return FakeNet()


# ------------------------------------------------------------------ guard
class FakeGuard(object):
    """inc2.guard.GuardV2's interface over planted images (module docstring)."""

    def __init__(self, eval_paths=(), base_paths=()):
        from weed_optimizer_framework.tools.near_dup import NearHashIndex
        self._eval, self._base, self._intake = NearHashIndex(), NearHashIndex(), NearHashIndex()
        for i, p in enumerate(eval_paths):
            self._eval.add(self._dh(p), ("dev", "e%d" % i), max_bits=6)
        for i, p in enumerate(base_paths):
            self._base.add(self._dh(p), ("base_v2", "b%d" % i), max_bits=6)
        self.counts = collections.Counter()
        self.n_intake = 0

    @staticmethod
    def _dh(p):
        from weed_optimizer_framework.tools.inc import common as C
        return int(C.dhash(p))

    @staticmethod
    def variants(p):
        from PIL import Image
        from weed_optimizer_framework.tools.funnel import leak as L
        try:
            with Image.open(p) as im:
                im.load()
                return {k: int(v) for k, v in L.dhash_variants(im).items()}
        except Exception:  # noqa: BLE001
            return None

    def add_intake(self, h, owner):
        self._intake.add(int(h), owner, max_bits=3)
        self.n_intake += 1

    def check_path(self, path):
        from weed_optimizer_framework.tools.inc import common as C
        h = C.dhash(path)
        v = self.variants(path)
        reason, match = self._decide(h, v)
        self.counts[reason or "pass"] += 1
        return reason, match, (h, v)

    def _decide(self, h, v):
        if h is None or v is None:
            return "unhashable", {"why": "unreadable"}
        m = self._eval.find(h)
        if m is not None:
            return "near_eval_v2", {"key": m[0][1], "bits": m[1]}
        for name, hv in sorted(v.items()):
            if name == "id":
                continue
            m = self._eval.find(hv)
            if m is not None:
                return "near_eval_variant", {"key": m[0][1], "bits": m[1], "variant": name}
        for name, hv in sorted(v.items()):
            m = self._base.find(hv)
            if m is not None:
                return "base_copy", {"key": m[0][1], "bits": m[1], "variant": name}
        m = self._intake.find(h)
        if m is not None:
            return "near_dup_intake", {"owner": m[0], "bits": m[1]}
        return None, None

    def index_record(self):
        return {"fake": True, "intake": self.n_intake}


# ------------------------------------------------------------------ annotation helpers
def coco_doc(images, categories, annotations):
    return {"images": images, "categories": categories, "annotations": annotations}


def main():
    setup("collect_world_")
    try:
        from weed_optimizer_framework.tools.collect import names as NM
        cfg = config()
        check("the weed collector config loads with its pins", cfg.name == "weed" and cfg.unmapped_id == 13
              and cfg.other_id == 12 and len(cfg.eppo) > 20)
        build_cache()
        nm = NM.load(cfg)
        check("the funnel taxonomy cache builds offline from the recorded GBIF answers",
              nm.provenance["funnel_cache"] is not None and nm.knows("Portulaca oleracea"))
        a = grid_img(TMP / "w" / "a.jpg", 1)
        b = grid_img(TMP / "w" / "b.jpg", 1, paint=(0, 1))
        c = grid_img(TMP / "w" / "c.jpg", 2)
        from weed_optimizer_framework.tools.inc import common as C
        near = bin(C.dhash(a) ^ C.dhash(b)).count("1")
        far = bin(C.dhash(a) ^ C.dhash(c)).count("1")
        check("a painted copy is 1..6 bits away, another seed far away", 1 <= near <= 6 and far > 6, (near, far))
        g = FakeGuard(eval_paths=[a])
        r, _m, _h = g.check_path(b)
        check("the fake guard refuses a near copy of an evaluation image", r == "near_eval_v2", r)
        r2, _m, _h = g.check_path(c)
        check("... and passes another picture", r2 is None, r2)
        net = make_net({("GET", "https://x/api"): (200, {"ok": 1})})
        st, body, sha, _h = net.get_json("https://x/api")
        check("the fake network answers its routes", st == 200 and body == {"ok": 1})
        e = raises(lambda: socket.create_connection(("example.org", 80), timeout=1), AssertionError)
        check("the real network is blocked", e is not None)
    finally:
        cleanup()
    print("\n%d failure(s)" % len(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
