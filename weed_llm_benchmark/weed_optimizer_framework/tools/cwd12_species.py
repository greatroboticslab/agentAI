"""Which weed each cottonweeddet12 class id is, and how a class name from any
other dataset is matched to one of them.

The alphabetical name list in downloads/cottonweeddet12/data.yaml is not the
dataset's class list. CottonWeedDet12 has no Crabgrass and no Nutsedge; it has
Waterhemp and Cutleaf Groundcherry. The ids were established two independent
ways, both matching YOLO boxes to another source's named boxes:

  * the dataset's own VGG annotations (CottonWeedDet12/annotation_VGG_json,
    region_attributes {"CottonWeed": {"<species>": true}}), paired in file
    order over the 3,661 train images whose region count equals the YOLO line
    count: 6,090 boxes, every id 100 % one species;
  * AgML's 3SeasonWeedDet10 release, which shares 2,904 photographs with the
    cwd12 train split and names its own boxes: 4,849 boxes, 100 % agreement,
    none unmatched.

The trainer's merged class space (mega_trainer, slots 0-11) is a permutation of
these ids that was named with the legacy labels. The permutation is kept so
existing checkpoints stay valid; what changes is the species each slot is known
to hold, and therefore which external boxes may join it.

No third-party imports: the tests and the dashboard import this without numpy.
"""
import re
from pathlib import Path

# cwd12 YOLO label id -> species.
CWD12_SPECIES = [
    "Waterhemp", "MorningGlory", "Purslane", "SpottedSpurge",
    "Carpetweed", "Ragweed", "Eclipta", "PricklySida",
    "PalmerAmaranth", "Sicklepod", "Goosegrass", "CutleafGroundcherry",
]

# The labels this project attached to the same ids (data.yaml order). Only
# PricklySida is right. Kept to read old reports and label files, never to join.
CWD12_LEGACY_LABELS = [
    "Carpetweeds", "Crabgrass", "Eclipta", "Goosegrass",
    "Morningglory", "Nutsedge", "PalmerAmaranth", "PricklySida",
    "Purslane", "Ragweed", "Sicklepod", "SpottedSpurge",
]

# Normalised name (lower case, letters and digits only) -> species. Common and
# scientific names seen in public weed datasets. A name missing from this table
# does not join a cwd12 slot; it goes to an auxiliary slot instead, which costs
# far less than joining the wrong one.
_ALIASES = {
    "waterhemp": "Waterhemp", "commonwaterhemp": "Waterhemp",
    "tallwaterhemp": "Waterhemp", "amaranthustuberculatus": "Waterhemp",
    "amaranthusrudis": "Waterhemp",
    "morningglory": "MorningGlory", "ivyleafmorningglory": "MorningGlory",
    "tallmorningglory": "MorningGlory", "pittedmorningglory": "MorningGlory",
    "ipomoea": "MorningGlory", "ipomoeahederacea": "MorningGlory",
    "ipomoeapurpurea": "MorningGlory", "ipomoealacunosa": "MorningGlory",
    "purslane": "Purslane", "commonpurslane": "Purslane",
    "portulacaoleracea": "Purslane",
    "spottedspurge": "SpottedSpurge", "euphorbiamaculata": "SpottedSpurge",
    "carpetweed": "Carpetweed", "mollugoverticillata": "Carpetweed",
    "ragweed": "Ragweed", "commonragweed": "Ragweed",
    "ambrosiaartemisiifolia": "Ragweed",
    "eclipta": "Eclipta", "ecliptaprostrata": "Eclipta",
    "ecliptaalba": "Eclipta", "falsedaisy": "Eclipta",
    "pricklysida": "PricklySida", "sidaspinosa": "PricklySida",
    "palmeramaranth": "PalmerAmaranth", "amaranthuspalmeri": "PalmerAmaranth",
    "sicklepod": "Sicklepod", "sennaobtusifolia": "Sicklepod",
    "cassiaobtusifolia": "Sicklepod",
    "goosegrass": "Goosegrass", "eleusineindica": "Goosegrass",
    "cutleafgroundcherry": "CutleafGroundcherry",
    "physalisangulata": "CutleafGroundcherry",
}


def name_key(name):
    """'Carpet weed', 'carpet_weed' and 'CarpetWeed' all become 'carpetweed'."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def species_of(name):
    """The cwd12 species a class name refers to, or None if it names none of them.

    Exact alias match on the normalised name, then the same with one trailing
    's' removed ('Carpetweeds'). Nothing looser: 'Giant ragweed' is a different
    plant from common ragweed and must not join the Ragweed slot."""
    k = name_key(name)
    if k in _ALIASES:
        return _ALIASES[k]
    if k.endswith("s") and k[:-1] in _ALIASES:
        return _ALIASES[k[:-1]]
    return None


def cli_species(name):
    """The species a command-line class argument names, or None.

    v3.60.0: like species_of, but a legacy-only label (Carpetweeds, Crabgrass,
    Morningglory, Nutsedge: a legacy label that is not itself a species key)
    is refused. Old campaign scripts pass those meaning the legacy slot, and
    species_of would fold 'Carpetweeds' onto Carpetweed, while the slot held
    Waterhemp."""
    if name in CWD12_SPECIES:
        return name
    if name in CWD12_LEGACY_LABELS:
        return None
    return species_of(name)


def legacy_to_species(label):
    """Translate a legacy cwd12 label (as written in old reports) to the species."""
    return CWD12_SPECIES[CWD12_LEGACY_LABELS.index(label)]


def species_to_legacy(species):
    """The legacy label an old artifact uses for `species` (for writing into a
    store whose vocabulary is still legacy, such as the three Roboflow projects
    below, so one project never holds two names for one plant)."""
    return CWD12_LEGACY_LABELS[CWD12_SPECIES.index(species)]


# ---------------------------------------------------------------- slot space
# The trainer's merged class space (mega_trainer slots 0-11): a permutation of
# the cwd12 ids, first written with the legacy labels. Checkpoints trained
# before v3.60.0 carry TRAINER_SLOT_LEGACY in model.names; later ones carry
# TRAINER_SLOT_SPECIES. The ids are the same.
TRAINER_SLOT_LEGACY = [
    "Carpetweeds", "Crabgrass", "PalmerAmaranth", "PricklySida",
    "Purslane", "Ragweed", "Sicklepod", "SpottedSpurge",
    "Eclipta", "Goosegrass", "Morningglory", "Nutsedge",
]
TRAINER_SLOT_SPECIES = [legacy_to_species(n) for n in TRAINER_SLOT_LEGACY]
CWD12_ID_TO_SLOT = {i: TRAINER_SLOT_LEGACY.index(n) for i, n in enumerate(CWD12_LEGACY_LABELS)}

# ---------------------------------------------------------------- naming
CWD12_COMMON = {
    "Waterhemp": "Waterhemp", "MorningGlory": "Morning glory",
    "Purslane": "Purslane", "SpottedSpurge": "Spotted spurge",
    "Carpetweed": "Carpetweed", "Ragweed": "Ragweed", "Eclipta": "Eclipta",
    "PricklySida": "Prickly sida", "PalmerAmaranth": "Palmer amaranth",
    "Sicklepod": "Sicklepod", "Goosegrass": "Goosegrass",
    "CutleafGroundcherry": "Cutleaf groundcherry",
}
CWD12_BINOMIAL = {
    "Waterhemp": "Amaranthus tuberculatus", "MorningGlory": "Ipomoea spp.",
    "Purslane": "Portulaca oleracea", "SpottedSpurge": "Euphorbia maculata",
    "Carpetweed": "Mollugo verticillata", "Ragweed": "Ambrosia artemisiifolia",
    "Eclipta": "Eclipta prostrata", "PricklySida": "Sida spinosa",
    "PalmerAmaranth": "Amaranthus palmeri", "Sicklepod": "Senna obtusifolia",
    "Goosegrass": "Eleusine indica", "CutleafGroundcherry": "Physalis angulata",
}

# ---------------------------------------------------------------- provenance
# A string alone cannot say which vocabulary it is in: "Ragweed" is Sicklepod as
# a legacy label and ragweed as a real one, and a CottonWeedID15 copy
# (rf_zig-zag) uses "Crabgrass" and "Nutsedge" for real crabgrass and nutsedge.
# So legacy labels are recognised only by where they come from.

# Datasets that are copies of cwd12 itself, and the species of each class id IN
# THEIR LABEL FILES. Their stored class_names are never read: the registry held
# a four-name list for cottonweed_holdout, whose files use all twelve ids.
CWD12_ID_SPACE = {
    "cottonweeddet12": list(CWD12_SPECIES),          # original ids 0-11
    "cottonweed_holdout": list(CWD12_SPECIES),       # original ids 0-11
    "cottonweed_sp8": TRAINER_SLOT_SPECIES[:8],      # local ids 0-7 = slots 0-7
}

# Roboflow projects created from our own cwd12 uploads: their class list is the
# legacy labels. Two kinds of box live in them.
#   * Boxes our uploader wrote on cwd12 photographs carry legacy labels
#     (legacy_to_species reads them). They are never read back for training:
#     the verified cwd12 copies are the source of truth for those photographs,
#     holdout photographs must not train at all, and in weed-crop-agent-dataset
#     the cottonweed_holdout push went through a four-name labelmap over
#     twelve-id files, so its Eclipta / Goosegrass / Morningglory / Nutsedge
#     classes do not even follow the legacy vocabulary.
#   * Boxes a person drew on any other photograph were named from the list the
#     person saw, so "Ragweed" there means ragweed: species_of reads them.
# Uploading into these projects writes legacy labels (species_to_legacy) until
# their classes are renamed, so no project holds two names for one plant.
LEGACY_ROBOFLOW_PROJECTS = frozenset({
    "cwd12-multiclass-v1", "weed-crop-agent-dataset", "cwd12-weeds",
})


def _as_list(names):
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]
    return list(names or [])


def is_legacy_label_list(names):
    """True when a whole class list is one of this project's legacy lists: the
    twelve in data.yaml order or in trainer slot order (optionally followed by
    the aux placeholders of a trained head), or the eight of cottonweed_sp8
    (optionally followed by novel-class names that are neither a legacy label
    nor a species: the leave-4-out heads, e.g. [...8, 'novel_weed_llm'], and
    the pre-v3.60.0 Config.get_species_names() list [...8, 'novel_weed']).
    Only a whole-list match counts; a single string is ambiguous."""
    seq = [str(x) for x in _as_list(names)]
    if seq[:12] in (CWD12_LEGACY_LABELS, TRAINER_SLOT_LEGACY):
        return True
    if seq[:8] != TRAINER_SLOT_LEGACY[:8]:
        return False
    # v3.60.0: an sp8 head plus novel classes is still a legacy list.
    return all(n not in CWD12_LEGACY_LABELS and species_of(n) is None
               for n in seq[8:])


def species_names_for(names):
    """`names` with legacy labels replaced by species when the whole list is a
    legacy list (a checkpoint trained before v3.60.0, an old data.yaml);
    returned unchanged otherwise, so a list that already names species is never
    translated twice. Keeps the input's shape (list or {id: name})."""
    if not is_legacy_label_list(names):
        return names
    if isinstance(names, dict):
        return {k: (legacy_to_species(v) if v in CWD12_LEGACY_LABELS else v)
                for k, v in names.items()}
    return [legacy_to_species(v) if v in CWD12_LEGACY_LABELS else v for v in names]


def class_species(slug, cid, class_names=None):
    """The cwd12 species of class `cid` of dataset `slug`, or None.

    cwd12 copies resolve by id (their stored names are ignored); a whole legacy
    list by legacy_to_species; any other class by species_of(its real name)."""
    space = CWD12_ID_SPACE.get(slug)
    if space is not None:
        return space[cid] if 0 <= cid < len(space) else None
    names = _as_list(class_names)
    if not 0 <= cid < len(names):
        return None
    name = str(names[cid])
    if is_legacy_label_list(names):
        return legacy_to_species(name) if name in CWD12_LEGACY_LABELS else None
    return species_of(name)


def uploaded_label_species(project, class_name):
    """Species of a box OUR UPLOADER wrote into Roboflow `project` (per-class
    counts of cwd12 uploads, for display). Not for boxes a person drew; see
    LEGACY_ROBOFLOW_PROJECTS."""
    if project in LEGACY_ROBOFLOW_PROJECTS:
        return legacy_to_species(class_name) if class_name in CWD12_LEGACY_LABELS else None
    return species_of(class_name)


# ---------------------------------------------------------------- crop banks
# synth_cutpaste's object banks. object_bank/ (before v3.60.0) names its
# folders with the legacy labels; object_bank_species/ names them by species
# and is marked with BANK_VOCAB_FILE. Kept here, stdlib-only, so the dashboard
# and the pipeline pick the same bank and read its folders the same way.
BANK_VOCAB_FILE = ".vocabulary"


def bank_vocabulary(root):
    """'species' for a bank written since v3.60.0, else 'legacy'."""
    try:
        if (Path(root) / BANK_VOCAB_FILE).read_text().strip() == "species":
            return "species"
    except OSError:
        pass
    return "legacy"


def bank_folder_species(root, folder):
    """The species a bank folder holds, or None if it is not a cwd12 class."""
    if bank_vocabulary(root) == "species":
        return folder if folder in CWD12_SPECIES else None
    return legacy_to_species(folder) if folder in CWD12_LEGACY_LABELS else None


def bank_in_use(legacy_root, species_root):
    """The bank the pipeline reads: the species bank once it is marked, else
    the legacy one (synth_cutpaste.default_bank_dir)."""
    if bank_vocabulary(species_root) == "species":
        return Path(species_root)
    return Path(legacy_root)


def bank_folder_for(root, species):
    """The folder of `species` in bank `root` (legacy label in a legacy bank)."""
    if species not in CWD12_SPECIES:
        return None
    return species if bank_vocabulary(root) == "species" else species_to_legacy(species)
