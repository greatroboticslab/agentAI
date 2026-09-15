#!/usr/bin/env python3
"""Build the poster into versions/vNN_* and never overwrite an earlier one.

Every build so far wrote the same filename, so each round of feedback erased
the sheet it was feedback on and there was nothing to compare against. This
allocates the next number, writes the deck, the PDF, a PNG and a manifest
under docs/poster/versions/, and appends one line to versions/INDEX.md.

    python3 docs/poster/build_version.py "what changed in one line"
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
VER = os.path.join(HERE, "versions")
sys.path.insert(0, HERE)


def next_n():
    ns = [int(d[1:3]) for d in os.listdir(VER)
          if d.startswith("v") and d[1:3].isdigit()]
    return max(ns) + 1 if ns else 1


def main():
    note = sys.argv[1] if len(sys.argv) > 1 else ""
    os.makedirs(VER, exist_ok=True)
    n = next_n()
    tag = "v%02d" % n
    out = os.path.join(VER, tag)
    os.makedirs(out, exist_ok=True)

    import styles, gen
    lib = json.load(open(os.path.join(HERE, "sections.json")))["sections"]
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                         capture_output=True, text=True).stdout.strip()
    made = []
    for kind, label in (("logo-left", "navy"), ("pale-band", "pale")):
        st = styles.Style("four", "house", "house", kind, "none", "normal", "mtsu")
        p = gen.Poster(st, lib)
        pptx = os.path.join(out, "%s_%s.pptx" % (tag, label))
        dropped = p.build(pptx)
        kept = [s for _, ids, _k in gen.MTSU_COLUMNS for s in ids
                if s in p.lib and s not in dropped]
        small = pptx.replace(".pptx", "_small.pptx")
        subprocess.run([sys.executable, os.path.join(HERE, "shrink_pptx.py"), pptx, small],
                       capture_output=True)
        made.append({"look": label, "pptx": os.path.basename(pptx),
                     "small": os.path.basename(small),
                     "body_pt": p.st.d["body"], "heading_pt": p.st.d["heading"],
                     "type_scale": p.type_scale, "kept": kept, "dropped": dropped,
                     "offsheet": getattr(p, "offsheet", []),
                     "widows": getattr(p, "widows", [])})
        if label == "navy":
            subprocess.run(["soffice", "--headless", "--convert-to", "pdf",
                            "--outdir", out, pptx], capture_output=True)
            subprocess.run(["soffice", "--headless", "--convert-to", "png",
                            "--outdir", out, pptx], capture_output=True)
    json.dump({"version": tag, "note": note, "commit": rev, "decks": made},
              open(os.path.join(out, "manifest.json"), "w"), indent=1)

    idx = os.path.join(VER, "INDEX.md")
    if not os.path.exists(idx):
        open(idx, "w").write("# Poster versions\n\n"
                             "Newest last. Each folder holds both looks, a PDF, a PNG "
                             "and a manifest naming every section kept and dropped.\n\n")
    with open(idx, "a") as fh:
        fh.write("- **%s** (%s) — %s — body %.1f pt, %d blocks\n"
                 % (tag, rev, note or "no note", made[0]["body_pt"], len(made[0]["kept"])))
    print("%s  body %.1f pt  %d blocks  -> %s"
          % (tag, made[0]["body_pt"], len(made[0]["kept"]), out))


if __name__ == "__main__":
    main()
