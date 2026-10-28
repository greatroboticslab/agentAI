"""Read-only survey of the VIA JSON + VOC XML files of 2SeasonWeedDet8 (no image is opened)."""
import collections, hashlib, json, os, sys
import xml.etree.ElementTree as ET
ROOT = sys.argv[1]
DIRS = ["Year2021_Part1/Year2021_Part1", "Year2021_Part2/Year2021_Part2", "Year2022/Year2022"]
sig = collections.Counter(); regsig = collections.Counter(); rasig = collections.Counter(); shapes = collections.Counter()
names = collections.Counter(); voc_names = collections.Counter(); per_dir = collections.Counter()
exts = collections.Counter(); examples = {}; mism = []; nreg = collections.Counter(); multi_true = 0
attr_names = collections.Counter(); jpg_no_json = 0; json_no_jpg = 0; top_entries = collections.Counter()
def tsig(v):
    if isinstance(v, dict): return "dict"
    if isinstance(v, list): return "list[%s]" % ",".join(sorted(set(tsig(x) for x in v))) if v else "list[]"
    return type(v).__name__
def via_boxes(entry):
    regs = entry.get("regions")
    out = []
    if isinstance(regs, dict) and "shape_attributes" in regs:
        sa, ra = regs["shape_attributes"], regs.get("region_attributes")
        if isinstance(sa, dict): sa, ra = [sa], [ra]
        pairs = list(zip(sa, ra))
    elif isinstance(regs, list):
        pairs = [(r.get("shape_attributes"), r.get("region_attributes")) for r in regs]
    elif isinstance(regs, dict):
        pairs = [(r.get("shape_attributes"), r.get("region_attributes")) for r in regs.values()]
    else:
        pairs = []
    for sa, ra in pairs:
        labs = []
        for k, v in (ra or {}).items():
            if isinstance(v, dict):
                labs += [n for n, t in v.items() if t]
            elif isinstance(v, str):
                labs.append(v)
        out.append((tuple(sorted(labs)), sa.get("x"), sa.get("y"), sa.get("x") + sa.get("width"), sa.get("y") + sa.get("height")))
    return out
for d in DIRS:
    p = os.path.join(ROOT, d)
    files = sorted(os.listdir(p))
    stems = collections.defaultdict(set)
    for f in files:
        s, e = os.path.splitext(f); stems[s].add(e.lower()); exts[e.lower()] += 1
    for s, es in sorted(stems.items()):
        if ".jpg" in es and ".json" not in es: jpg_no_json += 1
        if ".json" in es and ".jpg" not in es: json_no_jpg += 1
        if ".json" not in es: continue
        per_dir[d] += 1
        with open(os.path.join(p, s + ".json"), "rb") as fh: raw = fh.read()
        doc = json.loads(raw)
        top_entries[len(doc) if isinstance(doc, dict) else -1] += 1
        for k, e in doc.items():
            regs = e.get("regions")
            if isinstance(regs, dict) and "shape_attributes" in regs:
                k2 = "dict{shape_attributes:%s,region_attributes:%s}" % (tsig(regs["shape_attributes"]), tsig(regs.get("region_attributes")))
                sas = regs["shape_attributes"] if isinstance(regs["shape_attributes"], list) else [regs["shape_attributes"]]
                ras = regs.get("region_attributes") if isinstance(regs.get("region_attributes"), list) else [regs.get("region_attributes")]
            else:
                k2 = tsig(regs); sas = [r.get("shape_attributes") for r in (regs if isinstance(regs, list) else (regs or {}).values())]
                ras = [r.get("region_attributes") for r in (regs if isinstance(regs, list) else (regs or {}).values())]
            regsig[k2] += 1
            examples.setdefault(k2, d + "/" + s)
            nreg[len(sas)] += 1
            sig[tuple(sorted(e.keys()))] += 1
            for sa in sas: shapes[(sa or {}).get("name")] += 1
            for ra in ras:
                for an, av in (ra or {}).items():
                    attr_names[an] += 1
                    rasig["%s:%s" % (an, tsig(av))] += 1
                    if isinstance(av, dict):
                        t = [n for n, v in av.items() if v]
                        if len(t) != 1: multi_true += 1
                        for n in t: names[n] += 1
                    elif isinstance(av, str): names[av] += 1
        xp = os.path.join(p, s + ".xml")
        vb = sorted(b for e in doc.values() for b in via_boxes(e))
        if os.path.exists(xp):
            r = ET.parse(xp).getroot()
            xb = []
            for ob in r.findall("object"):
                n = ob.findtext("name"); voc_names[n] += 1
                bb = ob.find("bndbox")
                xb.append(((n,), int(float(bb.findtext("xmin"))), int(float(bb.findtext("ymin"))), int(float(bb.findtext("xmax"))), int(float(bb.findtext("ymax")))))
            if sorted(xb) != vb: mism.append(d + "/" + s)
print(json.dumps({"exts": exts, "per_dir_json": per_dir, "jpg_no_json": jpg_no_json, "json_no_jpg": json_no_jpg,
                  "top_entries_per_file": {str(k): v for k, v in top_entries.items()},
                  "entry_keys": {"|".join(k): v for k, v in sig.items()}, "regions_layout": regsig, "layout_example": examples,
                  "regions_per_image": {str(k): v for k, v in sorted(nreg.items())}, "shapes": {str(k): v for k, v in shapes.items()},
                  "region_attribute_layout": rasig, "attr_names": attr_names, "class_names_via": names, "class_names_voc": voc_names,
                  "multi_or_zero_true": multi_true, "via_vs_voc_mismatch": len(mism), "mismatch_examples": mism[:20]}, indent=1, default=str))
