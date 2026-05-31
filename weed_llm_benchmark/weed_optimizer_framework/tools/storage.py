"""
storage.py — abstract backend for image/label data so paths stop being
hardcoded to Lustre. Sibling to the MongoDB design (see
docs/mongodb_schema.md, Phase E1).

Why: the cluster goes away "in a few months" (per Prof Zhang, 2026-05-28).
When the labeler moves to the Uni server, paths like
`/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/datasets/<slug>/...`
won't exist. With a StorageBackend, the slug document carries
`storage_backend` + `storage_key`, and resolution happens at query time
in one place — not 50 places sprinkled through dashboard_server.py.

This is **Phase E2 skeleton**. Defines the protocol and a working
LustreBackend (matching current behavior). S3Backend + UniServerNASBackend
are documented stubs. No consumers refactored yet — that's E3, one
endpoint at a time, after this lands.

Default selection (when no per-slug override is set):

    weed_optimizer_framework.tools.storage.default_backend()

reads env `AGENTAI_STORAGE_BACKEND` (lustre | s3 | uniserv_nas), defaults
to "lustre". For Lustre, env `AGENTAI_LUSTRE_ROOT` overrides the project
root if needed.

Per-slug override (Mongo migration era): each slug document will carry
`storage_backend` and `storage_key` fields; call
`get_backend(slug_doc.storage_backend)` instead of `default_backend()`.

Public API (intentionally narrow):

    backend = default_backend()
    img_path = backend.get_image_path(slug, image_key)   # local path or URL
    lbl_dir  = backend.get_labels_dir(slug)
    with backend.open_image(slug, image_key) as f:
        ...                                              # bytes IO
    for key in backend.list_images(slug, max_n=10):
        ...

Currently `image_key` for Lustre is just the filename (matches today's
behavior). After Mongo migration, image_key becomes a backend-relative
path like "train/images/<file>.jpg" — so the protocol is
"key as backend interprets it."
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import IO, Iterator, Optional, Protocol, runtime_checkable


# ---- Defaults (resolved from env, but importable for tests) ----------------

DEFAULT_LUSTRE_ROOT = os.environ.get(
    "AGENTAI_LUSTRE_ROOT",
    "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark",
)


# ---- Protocol --------------------------------------------------------------

@runtime_checkable
class StorageBackend(Protocol):
    """One way to resolve image / label data for a slug. Each backend
    knows how to take (slug, image_key) and produce the actual bytes or a
    locally-resolvable path."""

    name: str

    def get_image_path(self, slug: str, image_key: str) -> Optional[str]:
        """Return a path or URL where the image is reachable. For Lustre
        this is an absolute local fs path. For S3, this could be a
        presigned URL (caller may need to download)."""
        ...

    def get_labels_dir(self, slug: str) -> Optional[str]:
        """Return a path/URL pointing at the slug's labels/ root.
        Returns None if the slug has no labels (classification-only)."""
        ...

    def open_image(self, slug: str, image_key: str) -> IO[bytes]:
        """Open and return a bytes-mode file-like object for the image.
        Caller should close (or use a `with` block)."""
        ...

    def list_images(self, slug: str, max_n: Optional[int] = None) -> Iterator[str]:
        """Yield image_keys (backend-relative) found for this slug, up to
        max_n. Bounded — must not enumerate huge dirs unconditionally."""
        ...

    def is_available(self, slug: str) -> bool:
        """Cheap presence check (e.g. directory exists / S3 prefix exists)."""
        ...


# ---- LustreBackend (real, default) -----------------------------------------

class LustreBackend:
    """Bridges-2 /ocean Lustre filesystem. Matches current behavior of
    dashboard_server.py / dataset_discovery.py.

    v3.0.56 (auto-loop iter 16): now registry-aware — `_slug_dir` reads
    the slug's `local_path` from `dataset_registry.json` when available,
    so canonical slugs (cottonweed_sp8 living under `downloads/`) resolve
    correctly without the defensive find_image_in_slug fallback.
    Lazy load of the registry; cached per backend instance + bound to
    the registry file's mtime so updates pick up automatically."""

    name = "lustre"

    def __init__(self, repo_root: Optional[str] = None):
        self.repo_root = Path(repo_root or DEFAULT_LUSTRE_ROOT)
        self.datasets_root = self.repo_root / "datasets"
        self._reg_cache: dict = {}  # slug → local_path str
        self._reg_mtime: float = 0.0

    def _registry_path(self) -> Path:
        return self.repo_root / "results" / "framework" / "dataset_registry.json"

    def _refresh_registry_if_stale(self) -> None:
        import json as _json
        rp = self._registry_path()
        try:
            mt = rp.stat().st_mtime
        except Exception:
            return
        if mt == self._reg_mtime and self._reg_cache:
            return
        try:
            with open(rp) as f:
                reg = _json.load(f)
        except Exception:
            return
        cache: dict = {}
        for slug, info in (reg.get("datasets") or {}).items():
            lp = info.get("local_path")
            if lp:
                cache[slug] = lp
        self._reg_cache = cache
        self._reg_mtime = mt

    def _slug_dir(self, slug: str) -> Path:
        # Sanity: refuse paths that escape datasets_root.
        if "/" in slug or "\\" in slug or slug in (".", ".."):
            raise ValueError(f"unsafe slug: {slug!r}")
        # Prefer the registry's local_path (handles canonical slugs that
        # live outside `datasets/`, e.g. cottonweed_sp8 under downloads/).
        self._refresh_registry_if_stale()
        lp = self._reg_cache.get(slug)
        if lp:
            p = Path(lp)
            # Defense: only accept paths within the repo root
            try:
                p.resolve().relative_to(self.repo_root.resolve())
            except Exception:
                # registry says path outside repo — refuse (safety)
                pass
            else:
                return p
        return self.datasets_root / slug

    def get_image_path(self, slug: str, image_key: str) -> Optional[str]:
        """image_key may be a bare filename (today's convention) or a
        backend-relative path. Both are accepted."""
        if "../" in image_key or image_key.startswith("/"):
            raise ValueError(f"unsafe image_key: {image_key!r}")
        slug_dir = self._slug_dir(slug)
        # Treat image_key as relative — try as-given first, then search
        # well-known subdirs for a bare filename.
        direct = slug_dir / image_key
        if direct.is_file():
            return str(direct)
        # Bare-filename fallback (matches dashboard_server.find_image_in_slug).
        # v3.0.61 (button-test iter 3): added test/images and test/ to
        # cover slugs like cottonweed_sp8 whose images live under test/.
        # Previously LustreBackend returned None and /api/img 404'd while
        # the file demonstrably existed (see _button_loop_state Bug 2).
        if "/" not in image_key:
            for sub in (slug_dir / "images",
                        slug_dir / "train" / "images",
                        slug_dir / "valid" / "images",
                        slug_dir / "test" / "images",
                        slug_dir / "train",
                        slug_dir / "valid",
                        slug_dir / "test",
                        slug_dir):
                cand = sub / image_key
                if cand.is_file():
                    return str(cand)
        return None

    def get_labels_dir(self, slug: str) -> Optional[str]:
        slug_dir = self._slug_dir(slug)
        if not slug_dir.is_dir():
            return None
        # Match dashboard's _find_label_dirs first hit.
        for cand in (slug_dir / "labels",
                     slug_dir / "train" / "labels",
                     slug_dir / "valid" / "labels"):
            if cand.is_dir():
                return str(cand)
        return None

    def open_image(self, slug: str, image_key: str) -> IO[bytes]:
        p = self.get_image_path(slug, image_key)
        if p is None:
            raise FileNotFoundError(f"{slug}:{image_key}")
        return open(p, "rb")

    def list_images(self, slug: str, max_n: Optional[int] = None) -> Iterator[str]:
        slug_dir = self._slug_dir(slug)
        if not slug_dir.is_dir():
            return
        exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
        n = 0
        # v3.0.61: include test/images for slugs like cottonweed_sp8.
        for sub in (slug_dir / "images", slug_dir / "train" / "images",
                    slug_dir / "valid" / "images",
                    slug_dir / "test" / "images", slug_dir):
            if not sub.is_dir():
                continue
            try:
                for p in sub.iterdir():
                    if p.suffix in exts:
                        # Yield the bare filename (today's contract).
                        yield p.name
                        n += 1
                        if max_n is not None and n >= max_n:
                            return
            except Exception:
                continue

    def is_available(self, slug: str) -> bool:
        return self._slug_dir(slug).is_dir()


# ---- S3Backend (stub) ------------------------------------------------------

class S3Backend:
    """boto3-backed object storage. Path → presigned URL for browsers;
    streamed bytes for server-side reads. STUB — not implemented yet.

    Image key = S3 object key under `<prefix>/<slug>/...`.

    Implementation outline:
      - Lazy boto3 client (only if used).
      - get_image_path: returns presigned URL (60-min TTL) OR streams
        bytes to a local cache under /tmp/<slug>/<key>.
      - list_images: paginated list_objects_v2 with cap.
    """
    name = "s3"

    def __init__(self, bucket: str, prefix: str = "weeds/"):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + "/"

    def get_image_path(self, slug: str, image_key: str) -> Optional[str]:
        raise NotImplementedError("S3Backend not yet implemented (Phase E2 stub)")

    def get_labels_dir(self, slug: str) -> Optional[str]:
        raise NotImplementedError("S3Backend not yet implemented (Phase E2 stub)")

    def open_image(self, slug: str, image_key: str) -> IO[bytes]:
        raise NotImplementedError("S3Backend not yet implemented (Phase E2 stub)")

    def list_images(self, slug: str, max_n: Optional[int] = None) -> Iterator[str]:
        raise NotImplementedError("S3Backend not yet implemented (Phase E2 stub)")

    def is_available(self, slug: str) -> bool:
        return False


# ---- UniServerNASBackend (stub, will mirror LustreBackend) ----------------

class UniServerNASBackend(LustreBackend):
    """Uni server (Tennessee Tech "Unie computer", per Prof Zhang) with
    NAS-mounted dataset root. Subclass of LustreBackend since the
    semantics are identical — just a different root path. The actual
    differentiator is the root, set via env `AGENTAI_UNISERV_ROOT`."""
    name = "uniserv_nas"

    def __init__(self, repo_root: Optional[str] = None):
        super().__init__(repo_root or os.environ.get(
            "AGENTAI_UNISERV_ROOT", "/srv/agentai/weed_llm_benchmark"))


# ---- Selection -------------------------------------------------------------

_BACKENDS: dict = {
    "lustre": LustreBackend,
    "s3": S3Backend,
    "uniserv_nas": UniServerNASBackend,
}

_default_singleton: Optional[StorageBackend] = None


def default_backend() -> StorageBackend:
    """Return the configured default backend (from env). Singleton."""
    global _default_singleton
    if _default_singleton is not None:
        return _default_singleton
    name = os.environ.get("AGENTAI_STORAGE_BACKEND", "lustre")
    _default_singleton = get_backend(name)
    return _default_singleton


def get_backend(name: str) -> StorageBackend:
    """Build a backend by name. For per-slug override (Mongo era), pass
    the slug document's storage_backend field."""
    if name == "lustre":
        return LustreBackend()
    if name == "s3":
        bucket = os.environ.get("AGENTAI_S3_BUCKET", "")
        if not bucket:
            raise RuntimeError("AGENTAI_S3_BUCKET env var required for S3Backend")
        return S3Backend(bucket=bucket)
    if name == "uniserv_nas":
        return UniServerNASBackend()
    raise ValueError(f"unknown storage backend: {name!r} "
                     f"(known: {list(_BACKENDS)})")


# ---- Convenience -----------------------------------------------------------

def list_known_backends() -> list:
    return list(_BACKENDS.keys())


# ---- Self-test -------------------------------------------------------------

def _self_test():
    """Minimal sanity check — verifies LustreBackend path resolution
    matches the dashboard's find_image_in_slug behavior. Run as:
        python -m weed_optimizer_framework.tools.storage
    """
    b = LustreBackend()
    print(f"default backend: {default_backend().name}")
    print(f"known backends: {list_known_backends()}")
    print(f"lustre root: {b.repo_root}")
    print(f"datasets root: {b.datasets_root}")
    # Try a known slug if available; just check it doesn't crash.
    test_slug = "cottonweed_sp8"
    if b.is_available(test_slug):
        print(f"slug {test_slug!r}: available")
        first_imgs = list(b.list_images(test_slug, max_n=3))
        print(f"  first {len(first_imgs)} images: {first_imgs}")
        if first_imgs:
            p = b.get_image_path(test_slug, first_imgs[0])
            print(f"  resolved: {p}")
        ld = b.get_labels_dir(test_slug)
        print(f"  labels dir: {ld}")
    else:
        print(f"slug {test_slug!r}: not available (running locally?)")


if __name__ == "__main__":
    _self_test()
