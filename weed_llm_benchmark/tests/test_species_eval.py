"""Species naming and id spaces in the evaluators (v3.60.0).

The evaluators import ultralytics / torch / pycocotools at module level, so the
pure functions are pulled out of their sources with ast and run against the
real cwd12_species module.

Run: cd weed_llm_benchmark && python3 tests/test_species_eval.py
"""
import ast
import json
import os
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from weed_optimizer_framework.tools import cwd12_species as cs  # noqa: E402
from weed_optimizer_framework.tools.cwd12_species import (  # noqa: E402
    CWD12_ID_TO_SLOT, CWD12_LEGACY_LABELS, CWD12_SPECIES, TRAINER_SLOT_LEGACY,
    TRAINER_SLOT_SPECIES, species_names_for)

TOOLS = HERE / "weed_optimizer_framework" / "tools"
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def extract(src, wanted, ns):
    """Exec the top-level defs / assignments named in `wanted` from `src` into ns."""
    tree = ast.parse(src)
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in wanted for t in node.targets):
            body.append(node)
    exec(compile(ast.Module(body=body, type_ignores=[]), "<extract>", "exec"), ns)
    return ns


def heredoc(sh_path, marker):
    s = Path(sh_path).read_text()
    m = re.search(r"<<'%s'\n(.*?)\n%s\n" % (marker, marker), s, re.S)
    return m.group(1)


def as_dict(seq):
    return dict(enumerate(seq))


# ------------------------------------------------------------ crossdataset_eval
xsrc = (TOOLS / "crossdataset_eval.py").read_text()
xns = extract(xsrc, {"ARM_SPECIES", "HOLDOUT_ARM_ID", "label_space",
                     "model_to_cwd12", "output_path"},
              {"CWD12_SPECIES": CWD12_SPECIES, "TRAINER_SLOT_SPECIES": TRAINER_SLOT_SPECIES,
               "species_names_for": species_names_for, "species_of": cs.species_of,
               "json": json, "Path": Path})
label_space, model_to_cwd12, output_path = (
    xns["label_space"], xns["model_to_cwd12"], xns["output_path"])
HOLD = xns["HOLDOUT_ARM_ID"]


def rag_ids(names):
    return sorted(k for k, c in model_to_cwd12(names).items() if c == HOLD)


check("holdout Ragweed id is raw cwd12 id 5", HOLD == 5, HOLD)

seed_head = as_dict(CWD12_LEGACY_LABELS)                 # s3 sealed seeds
check("s3 seed head: cwd12_id space", label_space(seed_head) == "cwd12_id")
check("s3 seed head: Ragweed is id 5 (stored 'Nutsedge')",
      rag_ids(seed_head) == [5] and seed_head[5] == "Nutsedge", rag_ids(seed_head))
check("s3 seed head: stored 'Ragweed' (id 9) maps to Sicklepod",
      model_to_cwd12(seed_head)[9] == CWD12_SPECIES.index("Sicklepod"))

ladder = as_dict(TRAINER_SLOT_LEGACY + [f"aux_{i}" for i in range(12, 100)])
check("ladder head (nc=100, legacy): trainer_slot space",
      label_space(ladder) == "trainer_slot")
check("ladder head: Ragweed is slot 11", rag_ids(ladder) == [11], rag_ids(ladder))
check("ladder head: slot 11 compared with raw id 5, slot 5 is Sicklepod",
      model_to_cwd12(ladder)[11] == 5 and model_to_cwd12(ladder)[5] == 9)
check("ladder head: aux slots do not map", all(k < 12 for k in model_to_cwd12(ladder)))
check("ladder head: every slot maps through CWD12_ID_TO_SLOT",
      all(model_to_cwd12(ladder)[CWD12_ID_TO_SLOT[i]] == i for i in range(12)))

new_slot = as_dict(TRAINER_SLOT_SPECIES + ["aux_12"])
check("new species slot head: Ragweed is slot 11", rag_ids(new_slot) == [11])
new_cwd = as_dict(CWD12_SPECIES)
check("new species cwd12 head: Ragweed is id 5", rag_ids(new_cwd) == [5])
check("species list not translated twice",
      label_space(new_cwd) == "cwd12_id" and model_to_cwd12(new_cwd) == {i: i for i in range(12)})

real = {0: "common ragweed", 1: "Crabgrass", 2: "Nutsedge", 3: "Giant ragweed"}
check("real-name head: 'common ragweed' is the arm, Crabgrass/Nutsedge are not cwd12",
      rag_ids(real) == [0] and set(model_to_cwd12(real)) == {0}, model_to_cwd12(real))
check("real-name head: label space other", label_space(real) == "other")
check("class-agnostic head: no species arm", rag_ids({0: "weed"}) == [])

with tempfile.TemporaryDirectory() as td:
    old = Path(td) / "s6.json"
    old.write_text(json.dumps({"per_seed": {"101": {"iw_ragweed": 0.0, "holdout_ragweed": 0.96}}}))
    check("pre-v3.60.0 artifact is not overwritten",
          output_path(old) == Path(td) / "s6_species.json", output_path(old))
    new = Path(td) / "n.json"
    new.write_text(json.dumps({"per_seed": {"101": {"iw_ragweed_species": 0.1}}}))
    check("v3.60.0 artifact is replaced in place", output_path(new) == new)
    check("missing artifact path used as is", output_path(Path(td) / "x.json") == Path(td) / "x.json")

check("docstring no longer claims the old same-species pairing",
      "IS cwd12's Ragweed" not in xsrc and "0.9767" not in xsrc)
check("species arm keys renamed", '"iw_ragweed_species"' in xsrc and '"iw_ragweed"' not in xsrc.split("def output_path")[1].split("return Path(out)\n\n")[1])

# ------------------------------------------------------------ eval_v3_0_23
esrc = (HERE / "eval_v3_0_23.py").read_text()
ens = extract(esrc, {"V3_NAMES", "SLOT_SPECIES", "check_cwd12_names", "check_slot_head"},
              {"CWD12_SPECIES": CWD12_SPECIES, "TRAINER_SLOT_LEGACY": TRAINER_SLOT_LEGACY,
               "TRAINER_SLOT_SPECIES": TRAINER_SLOT_SPECIES,
               "species_names_for": species_names_for, "species_of": cs.species_of})
old_join = {i: ens["V3_NAMES"].index(nm) for i, nm in enumerate(CWD12_LEGACY_LABELS)}
check("id map equals the old name join on today's legacy data.yaml",
      old_join == CWD12_ID_TO_SLOT)
check("staged names are the species of each slot",
      all(ens["SLOT_SPECIES"][CWD12_ID_TO_SLOT[i]] == CWD12_SPECIES[i] for i in range(12)))
check("legacy data.yaml names accepted", ens["check_cwd12_names"](CWD12_LEGACY_LABELS) == [])
check("species data.yaml names accepted", ens["check_cwd12_names"](CWD12_SPECIES) == [])
check("common-name data.yaml accepted",
      ens["check_cwd12_names"]([cs.CWD12_COMMON[s] for s in CWD12_SPECIES]) == [])
check("slot-order names in a cwd12 yaml are reported",
      len(ens["check_cwd12_names"](TRAINER_SLOT_SPECIES)) > 0)
check("short list reported", ens["check_cwd12_names"](CWD12_SPECIES[:4]) != [])
check("slot head recognised (legacy, nc=100)", ens["check_slot_head"](ladder))
check("slot head recognised (species)", ens["check_slot_head"](new_slot))
check("cwd12-order head is not a slot head", not ens["check_slot_head"](seed_head))
check("no name join left", "V3_NAMES.index" not in esrc and "in V3_NAMES" not in esrc)
check("per-class keys by species", "SLOT_SPECIES[i]: float(box.maps[i])" in esrc)
check("rounds-read keys kept",
      all(k in esrc for k in ('"mAP50_95"', '"mAP50"', '"n_classes_with_data"',
                              '"cwd12_test"', '"cwd12_valid"')))

# ------------------------------------------------------------ wbf_tta / pycoco
wsrc = (TOOLS / "wbf_tta_eval.py").read_text()
wns = extract(wsrc, {"_LEGACY_CANONICAL_12"}, {})
check("wbf fallback list is the trainer slot legacy list",
      wns["_LEGACY_CANONICAL_12"] == TRAINER_SLOT_LEGACY)
check("wbf fallback names translate to slot species",
      list(species_names_for(wns["_LEGACY_CANONICAL_12"])) == TRAINER_SLOT_SPECIES)
check("sealed yaml names translate to cwd12 species",
      list(species_names_for(CWD12_LEGACY_LABELS)) == CWD12_SPECIES)
check("wbf wrong comment removed",
      "id 2 is Eclipta in the data" not in wsrc and "alphabetically" not in wsrc)
check("wbf main names per-class by species",
      "names = list(species_names_for(names))" in wsrc and '"names_as_read"' in wsrc)
psrc = (TOOLS / "pycoco_eval.py").read_text()
check("pycoco categories named by slot species",
      "CANONICAL_12 = list(TRAINER_SLOT_SPECIES)" in psrc and '"Nutsedge"' not in psrc)

# ------------------------------------------------------------ run_s3_bestmodel_eval.sh
s3 = heredoc(HERE / "run_s3_bestmodel_eval.sh", "PY")
ast.parse(s3)
s3ns = extract(s3, {"head_species"}, {"cs": cs})
check("s3 per-class names are CWD12_SPECIES", "NAMES = list(cs.CWD12_SPECIES)" in s3
      and "'Carpetweeds'" not in s3)
check("s3 sealed seed head reads as CWD12_SPECIES (no warning)",
      s3ns["head_species"](seed_head)[:12] == CWD12_SPECIES)
check("s3 slot head would warn", s3ns["head_species"](ladder)[:12] != CWD12_SPECIES)
s3o = extract(s3, {"output_path"}, {"json": json})["output_path"]
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "s3_best_model_eval.json")
    check("s3 no existing file -> historical path", s3o(p) == p)
    Path(p).write_text(json.dumps({"per_species_map50_95": {"Carpetweeds": {"mean": 0.9}}}))
    check("s3 legacy-keyed artifact kept (writes _species)",
          s3o(p) == os.path.join(td, "s3_best_model_eval_species.json"))
    Path(p).write_text(json.dumps({"per_species_keys": "species by cwd12 id"}))
    check("s3 species artifact overwritten in place", s3o(p) == p)
check("s3 writes through output_path", "open(out_path,\"w\")" in s3)
check("eval_v3_0_23 falls back to loading cwd12_species by file",
      "except ImportError:" in esrc and "spec_from_file_location" in esrc)

# ------------------------------------------------------------ run_eval_generic.sh
gen = heredoc(HERE / "run_eval_generic.sh", "PYEOF")
ast.parse(gen)
gns = extract(gen, {"_load_cwd12_species", "_species_key", "class_mismatches"}, {"os": os})
cwd = os.getcwd()
try:
    os.chdir(HERE)
    loaded = gns["_load_cwd12_species"]()
finally:
    os.chdir(cwd)
check("generic eval loads cwd12_species by file",
      loaded is not None and loaded.CWD12_SPECIES == CWD12_SPECIES)
cm = gns["class_mismatches"]
check("seed head vs sealed legacy yaml: no mismatch", cm(cs, seed_head, CWD12_LEGACY_LABELS) == [])
check("species head vs legacy-named yaml in same id order: no mismatch",
      cm(cs, new_cwd, CWD12_LEGACY_LABELS) == [])
mm = cm(cs, ladder, CWD12_LEGACY_LABELS)
check("slot head vs cwd12-id yaml: mismatch reported",
      len(mm) > 0 and all(isinstance(x, tuple) and len(x) == 3 for x in mm), mm)
check("slot head vs slot-staged yaml: no mismatch", cm(cs, ladder, TRAINER_SLOT_SPECIES) == [])
check("cottonweed_sp8 resolved by id space, not stale stored names",
      cm(cs, ladder, ["Carpetweeds", "Crabgrass", "Eclipta", "Goosegrass"],
         slug="cottonweed_sp8") == [])
real_ds = ["Crabgrass", "Nutsedge"]
check("real Crabgrass/Nutsedge never match the legacy head's ids",
      len(cm(cs, seed_head, real_ds)) == 2)
check("dataset id beyond the head is reported", cm(cs, {0: "weed"}, ["weed", "corn"]) == [(1, None, "corn")])
check("generic eval does not refuse", "sys.exit" not in gen.split("class_mismatches(cs, model.names")[1].split("r = model.val(")[0])

print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
sys.exit(1 if FAILS else 0)
