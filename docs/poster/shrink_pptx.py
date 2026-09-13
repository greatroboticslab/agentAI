#!/usr/bin/env python3
"""Recompress a poster .pptx for email without touching its geometry.

The plates that are photographs are written as RGBA PNG at 300 ppi, which is
the right thing on disk and the wrong thing inside a deck: the twelve-species
strip alone is 32 MB. They become JPEG at the same pixel size, so print
resolution is unchanged; the charts stay PNG because JPEG rings on hairlines
and type. The rels and [Content_Types].xml are patched to match, because a
renamed part that nothing points at is a deck that opens with a red X.
"""
import os, re, shutil, subprocess, sys, zipfile
from PIL import Image

src, dst = sys.argv[1], sys.argv[2]
QUALITY = int(os.environ.get("Q", "88"))
MIN_MB = float(os.environ.get("MIN_MB", "1.0"))

work = "/tmp/_pptx_shrink"
shutil.rmtree(work, ignore_errors=True)
os.makedirs(work)
with zipfile.ZipFile(src) as z:
    z.extractall(work)

media = os.path.join(work, "ppt", "media")
renamed = {}
for name in sorted(os.listdir(media)):
    p = os.path.join(media, name)
    if not name.lower().endswith(".png") or os.path.getsize(p) < MIN_MB * 1e6:
        continue
    im = Image.open(p)
    if im.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im.convert("RGBA"), mask=im.convert("RGBA").split()[-1])
        im = bg
    else:
        im = im.convert("RGB")
    out = os.path.splitext(name)[0] + ".jpeg"
    im.save(os.path.join(media, out), "JPEG", quality=QUALITY,
            optimize=True, progressive=True, subsampling=0)
    before, after = os.path.getsize(p), os.path.getsize(os.path.join(media, out))
    print("  %-16s %6.1f -> %5.1f MB  %s" % (name, before / 1e6, after / 1e6, im.size))
    os.remove(p)
    renamed[name] = out

if renamed:
    for root, _d, files in os.walk(work):
        for fn in files:
            if not fn.endswith((".xml", ".rels")):
                continue
            fp = os.path.join(root, fn)
            s = open(fp, encoding="utf-8").read()
            t = s
            for old, new in renamed.items():
                t = t.replace(old, new)
            if "[Content_Types]" in fn and "Extension=\"jpeg\"" not in t:
                t = t.replace("<Types ", "<Types ", 1)
                t = re.sub(r"(<Types[^>]*>)",
                           r'\1<Default Extension="jpeg" ContentType="image/jpeg"/>', t, count=1)
            if t != s:
                open(fp, "w", encoding="utf-8").write(t)

if os.path.exists(dst):
    os.remove(dst)
zf = zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=9)
for root, _d, files in os.walk(work):
    for fn in files:
        fp = os.path.join(root, fn)
        zf.write(fp, os.path.relpath(fp, work))
zf.close()
print("%s  %.1f MB -> %s  %.1f MB"
      % (os.path.basename(src), os.path.getsize(src) / 1e6,
         os.path.basename(dst), os.path.getsize(dst) / 1e6))
