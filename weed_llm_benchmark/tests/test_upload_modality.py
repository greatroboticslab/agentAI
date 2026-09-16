#!/usr/bin/env python3
"""An upload is classified by what arrived, not by what the project declared.

Every project is created modality=["image"] unless somebody says otherwise
(dashboard_server.py:2158, :8036). The uploader used to fix its accepted
extensions to the project's declared modalities and drop everything else, so a
1.9 GB field video dropped into a fresh project came back as "no recognized
image files in the upload (accepted: .bmp, .jpeg, .jpg, .png, .webp)" and was
thrown away. The file was fine; the default chosen before the data existed was
the problem.

These pin the behaviour that replaced it:
  * a file whose extension belongs to ANY known modality is stored, whatever the
    project declared;
  * the project is widened to include what actually turned up;
  * the size cap is large enough for real field video, and is enforced by
    streaming rather than by buffering.

Run:  python3 tests/test_upload_modality.py
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRV = ROOT / "weed_optimizer_framework" / "tools" / "dashboard_server.py"
FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def main():
    if not SRV.exists():
        print("  skip  no dashboard_server.py in this checkout")
        return 0
    src = SRV.read_text()

    # --- the modality table, read out of the server itself -------------------
    ns = {}
    m = re.search(r"^_UPLOAD_IMG_EXT = (\{.*?\})", src, re.M | re.S)
    exec("_UPLOAD_IMG_EXT = " + m.group(1), ns)
    m = re.search(r"^_MODALITY_EXT = (\{.*?^\})", src, re.M | re.S)
    exec("_MODALITY_EXT = " + m.group(1), ns)
    MOD = ns["_MODALITY_EXT"]

    check("the server still declares a video modality", ".mp4" in MOD.get("video", set()))

    # The decision the uploader makes for a file that the project's declared
    # modalities do not cover. This mirrors the `_mod_of` line in the dispatch
    # loop: membership of ANY modality is enough to keep the file.
    def mod_of(ext):
        return next((k for k, exts in MOD.items() if ext in exts), None)

    for ext, want in ((".mp4", "video"), (".mov", "video"), (".csv", "sensor"),
                      (".wav", "audio"), (".pcd", "pointcloud")):
        check("%s is recognised as %s even in an image-only project" % (ext, want),
              mod_of(ext) == want,
              "got %r" % mod_of(ext))
    check("an extension we genuinely do not handle is still refused",
          mod_of(".exe") is None)

    # --- the dispatch loop actually uses it ----------------------------------
    check("the uploader classifies an unexpected file instead of skipping it",
          "_mod_of = next((m for m, exts in _MODALITY_EXT.items() if ext in exts), None)" in src,
          "the ingest-first branch is gone; an undeclared modality would be dropped again")
    check("what was stored widens the project's modality",
          "_dbw.update_domain(domain, {\"modality\": _new})" in src,
          "the project would stay image-only and the next upload would be refused")
    check("the failure message reports what arrived, not only what is allowed",
          "nothing in this upload could be stored. It held: " in src)

    # --- the cap -------------------------------------------------------------
    m = re.search(r"^_MAX_UPLOAD_BYTES = (.+?)\s*(?:#|$)", src, re.M)
    cap = eval(m.group(1), {})
    gb = cap / (1024 ** 3)
    check("the size cap admits real field video (>= 8 GB)", gb >= 8,
          "cap is %.1f GB; a single phone clip already reached 1.9 GB" % gb)
    # A cap is only safe to raise because neither path holds the file in memory.
    check("the multipart path streams to disk in chunks",
          "chunk = await uf.read(1024 * 1024)" in src)
    check("the raw-body path streams to disk in chunks",
          "async for chunk in request.stream():" in src)

    print("\n%d failure(s)" % len(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
