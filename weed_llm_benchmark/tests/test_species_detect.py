#!/usr/bin/env python3
"""The detection API names species, not the legacy cwd12 labels of its checkpoint.

The served checkpoint carries the legacy labels in cwd12 id order ('Nutsedge' is
ragweed, 'Ragweed' is sicklepod). These pin that weed_detect translates them once
at load and sets them on the model so the drawn overlay agrees, that the model
card's reliability table is keyed by the species of each measured id, that a
checkpoint already naming species passes through untouched, and that
field_fire_sweep counts under the same names.

ultralytics and torch are replaced by stubs; no weights are loaded.

Run:  python3 tests/test_species_detect.py
"""
import io
import logging
import os
import pathlib
import sys
import tempfile
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools import cwd12_species as S  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


# ---------------------------------------------------------------- stubs
class _Boxes:
    def __init__(self, cls):
        self.cls = _T([float(c) for c in cls])
        self.conf = _T([0.9] * len(cls))
        self.xyxy = _T([[1.0, 2.0, 3.0, 4.0]] * len(cls))

    def __len__(self):
        return len(self.cls.v)


class _T:
    def __init__(self, v):
        self.v = v

    def tolist(self):
        return list(self.v)


class _Res:
    def __init__(self, cls, names):
        self.boxes = _Boxes(cls)
        self.names = names


class _Inner:
    def __init__(self, names):
        self.names = names


class FakeYOLO:
    NAMES = None
    CLS = []

    def __init__(self, path):
        self.model = _Inner(dict(FakeYOLO.NAMES))
        self.names_at_first_predict = None

    @property
    def names(self):
        return self.model.names

    def predict(self, *a, **k):
        if self.names_at_first_predict is None:
            self.names_at_first_predict = dict(self.model.names)
        return [_Res(FakeYOLO.CLS, self.names_at_first_predict)]


sys.modules["ultralytics"] = types.SimpleNamespace(YOLO=FakeYOLO)
sys.modules["torch"] = types.SimpleNamespace(
    cuda=types.SimpleNamespace(is_available=lambda: False))

from weed_optimizer_framework.tools import weed_detect as D  # noqa: E402

LEGACY = dict(enumerate(S.CWD12_LEGACY_LABELS))


def _fresh(names, cls, basename=D.CARD_CHECKPOINT):
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, basename)
    open(path, "wb").close()
    D.WEIGHTS = path
    D._CTX["log"] = logging.getLogger("t")
    D._MODEL.update(obj=None, err=None, loaded_at=None, device=None,
                    names=None, legacy_names=None, card=False)
    FakeYOLO.NAMES = names
    FakeYOLO.CLS = cls
    m = D._load()
    os.unlink(path)
    os.rmdir(tmpdir)
    return m


def _jpeg():
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, "JPEG")
    return buf.getvalue()


print("model card table")
check("PER_SPECIES keyed by the twelve species",
      sorted(D.PER_SPECIES) == sorted(S.CWD12_SPECIES), sorted(D.PER_SPECIES))
check("Sicklepod holds the card's 0.9767 (legacy 'Ragweed')",
      D.PER_SPECIES["Sicklepod"] == 0.9767)
check("Ragweed holds the card's 0.8585 (legacy 'Nutsedge')",
      D.PER_SPECIES["Ragweed"] == 0.8585)
check("Carpetweed holds the card's 0.7324 (legacy 'Morningglory')",
      D.PER_SPECIES["Carpetweed"] == 0.7324)
check("weak species are Purslane, SpottedSpurge, Carpetweed",
      sorted(D.WEAK) == ["Carpetweed", "Purslane", "SpottedSpurge"], D.WEAK)
check("no legacy-only name left in the table",
      not ({"Crabgrass", "Nutsedge", "Carpetweeds", "Morningglory"} & set(D.PER_SPECIES)))
check("same-name clause removed from the domain-gap warning",
      "same-name" not in D.MODEL_META["domain_gap_warning"]
      and "0.960" not in D.MODEL_META["domain_gap_warning"])
check("class-agnostic 0.873 -> 0.100 kept",
      "0.873 in-domain -> 0.100" in D.MODEL_META["domain_gap_warning"])
check("module docstring no longer uses legacy examples",
      "Morningglory (0.7324" not in D.__doc__ and "Ragweed (0.9767" not in D.__doc__)

print("legacy-labelled checkpoint")
m = _fresh(LEGACY, [5, 9, 1])
check("loaded", m["obj"] is not None, m["err"])
check("names translated to species by id",
      m["names"] == dict(enumerate(S.CWD12_SPECIES)), m["names"])
check("model names set before the first predict (overlay draws species)",
      m["obj"].names_at_first_predict == dict(enumerate(S.CWD12_SPECIES)))
check("legacy names remembered", m["legacy_names"] == LEGACY)
dets, _ = D._predict(_jpeg(), False)
check("id 5 reported as Ragweed, legacy_label Nutsedge",
      dets[0]["species"] == "Ragweed" and dets[0]["legacy_label"] == "Nutsedge", dets[0])
check("id 9 reported as Sicklepod with its card AP",
      dets[1]["species"] == "Sicklepod" and dets[1]["species_holdout_map50_95"] == 0.9767)
check("id 1 reported as MorningGlory, not Crabgrass",
      dets[2]["species"] == "MorningGlory" and dets[2]["legacy_label"] == "Crabgrass")
check("weak flag follows the species",
      [d["low_reliability_species"] for d in dets] == [False, False, False])

print("species-named checkpoint")
SP = dict(enumerate(S.CWD12_SPECIES))
m = _fresh(SP, [4])
check("species list passes through", m["names"] == SP and m["legacy_names"] is None)
dets, _ = D._predict(_jpeg(), False)
check("no legacy_label field", "legacy_label" not in dets[0], dets[0])
check("Carpetweed flagged weak",
      dets[0]["species"] == "Carpetweed" and dets[0]["low_reliability_species"])

print("trainer-slot legacy checkpoint")
TS = dict(enumerate(S.TRAINER_SLOT_LEGACY))
m = _fresh(TS, [11])
check("slot order translated", m["names"] == dict(enumerate(S.TRAINER_SLOT_SPECIES)))
dets, _ = D._predict(_jpeg(), False)
check("slot 11 (legacy Nutsedge) is Ragweed", dets[0]["species"] == "Ragweed", dets[0])

check("slot-order head is not the card's model: no card figures",
      not m["card"] and dets[0]["species_holdout_map50_95"] is None
      and dets[0]["low_reliability_species"] is False, dets[0])

print("card figures only for the card's checkpoint")
m = _fresh(LEGACY, [4], basename="cwd12_new_run.pt")
dets, _ = D._predict(_jpeg(), False)
check("other weights file: species named, reliability null",
      not m["card"] and dets[0]["species"] == "Carpetweed"
      and dets[0]["species_holdout_map50_95"] is None
      and dets[0]["low_reliability_species"] is False, dets[0])
REAL = dict(enumerate(["Carpet weed", "Crabgrass", "Eclipta", "Goosegrass",
                       "Morning glory", "Nutsedge"]))
m = _fresh(REAL, [2, 3, 4])
dets, _ = D._predict(_jpeg(), False)
check("card filename with a non-cwd12 head: no borrowed card numbers",
      not m["card"] and all(d["species_holdout_map50_95"] is None
                            and d["low_reliability_species"] is False for d in dets),
      dets)
m = _fresh(LEGACY, [9])
check("card checkpoint with its legacy head: card applies", m["card"])

print("field_fire_sweep")
from weed_optimizer_framework.tools import field_fire_sweep as F  # noqa: E402
src = pathlib.Path(F.__file__).read_text()
check("sweep translates model.names with species_names_for",
      "names = species_names_for(dict(raw_names))" in src)
check("sweep writes class_names and the legacy flag",
      '"class_names": names' in src and '"checkpoint_labels_legacy"' in src)
check("same translation gives species for a legacy head",
      F.species_names_for(dict(LEGACY))[5] == "Ragweed")

print()
if FAILURES:
    print("FAILED: %d" % len(FAILURES))
    sys.exit(1)
print("all passed")
