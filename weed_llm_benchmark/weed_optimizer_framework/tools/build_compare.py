"""Build a 2x6 real-vs-FLUX Goosegrass comparison + a clean 6x6 FLUX montage.

Run on the cluster from $REPO_ROOT. Outputs to results/framework/:
  goosegrass_compare_2x6.jpg   — top row 6 REAL Goosegrass crops, bottom row 6 FLUX
  flux_montage_clean_6x6.jpg   — 36 FLUX images (full image, not crop), 6x6 grid
  perclass_montage_2x12.jpg    — top row 12 REAL representative crops (one per
                                 cwd12 species), bottom row 12 FLUX-generated
                                 crops at the bbox of that species
"""
import os
import random
from pathlib import Path

from PIL import Image, ImageDraw

REPO = Path(os.environ.get("REPO_ROOT",
                           "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"))
F = REPO / "results" / "framework"
BANK = F / "synth_cutpaste" / "object_bank"
FLUX_IMG = F / "synth_diffusion" / "images"
FLUX_LBL = F / "synth_diffusion" / "labels"
CANON = ["Carpetweeds","Crabgrass","Eclipta","Goosegrass","Morningglory",
         "Nutsedge","PalmerAmaranth","PricklySida","Purslane","Ragweed",
         "Sicklepod","SpottedSpurge"]

def flux_crops_for(class_id: int, max_n: int = 6):
    out = []
    for lbl in sorted(FLUX_LBL.glob("*.txt")):
        for ln in lbl.read_text().splitlines():
            parts = ln.split()
            if not parts:
                continue
            try:
                cid = int(parts[0])
            except ValueError:
                continue
            if cid != class_id:
                continue
            img = FLUX_IMG / (lbl.stem + ".jpg")
            if not img.is_file():
                continue
            try:
                cx, cy, bw, bh = map(float, parts[1:5])
            except ValueError:
                continue
            out.append((img, cx, cy, bw, bh))
            break
        if len(out) >= max_n:
            break
    return out

def paste_real(sheet, cell, x_off, y_off, p):
    try:
        im = Image.open(p).convert("RGB").resize((cell, cell), Image.BILINEAR)
        sheet.paste(im, (x_off, y_off))
        return True
    except Exception as e:
        print(f"real paste fail {p}: {e}")
        return False

def paste_flux_crop(sheet, cell, x_off, y_off, entry):
    p, cx, cy, bw, bh = entry
    try:
        im = Image.open(p).convert("RGB")
        W, H = im.size
        x1 = int(max(0, cx - bw / 2) * W)
        y1 = int(max(0, cy - bh / 2) * H)
        x2 = int(min(1, cx + bw / 2) * W)
        y2 = int(min(1, cy + bh / 2) * H)
        if x2 - x1 < 16 or y2 - y1 < 16:
            crop = im
        else:
            crop = im.crop((x1, y1, x2, y2))
        crop = crop.resize((cell, cell), Image.BILINEAR)
        sheet.paste(crop, (x_off, y_off))
        return True
    except Exception as e:
        print(f"flux crop fail {p}: {e}")
        return False

# ---------------------------------------------------------------- Goosegrass 2x6
def build_goosegrass_2x6():
    real_dir = BANK / "Goosegrass"
    real = sorted(real_dir.glob("*.png")) if real_dir.is_dir() else []
    flux = flux_crops_for(CANON.index("Goosegrass"), max_n=6)
    print(f"goosegrass: real_count={len(real)} flux_count={len(flux)}")
    rng = random.Random(0)
    real_pick = rng.sample(real, min(6, len(real)))
    flux_pick = flux[:6]
    cell, N, header = 256, 6, 36
    sheet = Image.new("RGB", (cell * N, cell * 2 + header), (240, 240, 245))
    d = ImageDraw.Draw(sheet)
    d.text((10, 4),
           "TOP: REAL Goosegrass crops (cwd12 / trusted slugs)   "
           "BOTTOM: FLUX-generated Goosegrass (bbox crop)",
           fill=(20, 20, 20))
    for i, p in enumerate(real_pick):
        paste_real(sheet, cell, i * cell, header, p)
    for i, e in enumerate(flux_pick):
        paste_flux_crop(sheet, cell, i * cell, cell + header, e)
    out = F / "goosegrass_compare_2x6.jpg"
    sheet.save(out, quality=92)
    print(f"WROTE {out} {out.stat().st_size}")

# ---------------------------------------------------------------- per-class 2x12
def build_perclass_2x12():
    rng = random.Random(1)
    cell, header = 192, 60
    sheet = Image.new("RGB", (cell * 12, cell * 2 + header), (240, 240, 245))
    d = ImageDraw.Draw(sheet)
    d.text((10, 4),
           "TOP: REAL crops (one random per cwd12 class)   "
           "BOTTOM: FLUX-generated crop of same class (NULL = FLUX didn't make any)",
           fill=(20, 20, 20))
    for i, cls in enumerate(CANON):
        # column header label
        d.text((i * cell + 6, header - 16), cls[:14], fill=(80, 80, 80))
        # real
        rd = BANK / cls
        cands = sorted(rd.glob("*.png")) if rd.is_dir() else []
        if cands:
            paste_real(sheet, cell, i * cell, header, rng.choice(cands))
        # flux
        flux = flux_crops_for(i, max_n=1)
        if flux:
            paste_flux_crop(sheet, cell, i * cell, cell + header, flux[0])
        else:
            d.rectangle([i * cell, cell + header,
                         (i + 1) * cell, cell * 2 + header],
                        fill=(60, 60, 60))
            d.text((i * cell + 30, cell + header + cell // 2 - 6),
                   "no FLUX", fill=(220, 220, 220))
    out = F / "perclass_compare_2x12.jpg"
    sheet.save(out, quality=92)
    print(f"WROTE {out} {out.stat().st_size}")

# ---------------------------------------------------------------- FLUX 6x6 clean
def build_flux_6x6():
    imgs = sorted(FLUX_IMG.glob("*.jpg"))
    print(f"flux total: {len(imgs)}")
    grid, cell = 6, 256
    pick = imgs[: grid * grid]
    sheet = Image.new("RGB", (cell * grid, cell * grid), (20, 20, 20))
    for idx, p in enumerate(pick):
        try:
            im = Image.open(p).convert("RGB").resize((cell, cell), Image.BILINEAR)
            sheet.paste(im, ((idx % grid) * cell, (idx // grid) * cell))
        except Exception:
            pass
    out = F / "flux_montage_clean_6x6.jpg"
    sheet.save(out, quality=92)
    print(f"WROTE {out} {out.stat().st_size}")

if __name__ == "__main__":
    print(f"REPO={REPO}")
    print(f"BANK exists: {BANK.is_dir()}, FLUX_IMG exists: {FLUX_IMG.is_dir()}")
    build_goosegrass_2x6()
    build_perclass_2x12()
    build_flux_6x6()
    print("DONE")
