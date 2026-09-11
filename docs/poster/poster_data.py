"""Every number the poster prints, in one place, each with the file it came from.

The poster is regenerated from this dict, so a number that lands later is a
one-line edit here and one command to re-render. Anything still running carries
PENDING and renders as a visible placeholder rather than as a blank or a guess.

Provenance strings are what a reviewer is handed when they ask "where is that
from"; keep them resolvable.
"""

PENDING = "PENDING"

MEETING = {
    "title": "What Actually Moves a Weed Detector",
    "subtitle": ("Six months of an autonomous data-harvesting pipeline, instrumented and audited: "
                 "which levers move accuracy, and which supervision catches failure"),
    "authors": [("Hongbo Zhang", "1"), ("Harry He", "1"), ("Song Cui", "2")],
    "affiliations": ["Department of Engineering Technology",
                     "School of Agriculture"],
    "institution": "Middle Tennessee State University, Murfreesboro, TN, USA",
    "size_in": (48, 24),
}

# --- the holdout every in-domain number is measured on -----------------------
HOLDOUT = {
    "name": "CottonWeedDet12 test+valid",
    "images": 1977,
    "instances": 3257,
    "classes": 12,
    "note": ("Used as the validation set during training, so every figure is a maximum over "
             "21-100 evaluations on the set it is reported on. The size of that optimism is "
             "measured and printed beside each row."),
    "source": "cwd12_sealed.yaml; instance count re-summed from s3_tta_ceiling/ens3_640.json per_class n_gt",
}

# --- F1 levers ---------------------------------------------------------------
LEVERS = [
    {"lever": "COCO pretraining",
     "delta": 0.0714, "hi": "0.8755 ± 0.0029", "lo": "0.8041 ± 0.0028", "n": 3,
     "detail": "YOLO11n pretrained vs random init, cwd12 train split (3,671 images), 100 epochs",
     "src": "results/framework/s3_yolo11n/s10{1,2,3}/results.csv and scratch_s10{1,2,3}/results.csv"},
    {"lever": "Architecture at equal init",
     "delta": 0.0225, "hi": "0.8266 ± 0.0064", "lo": "0.8041 ± 0.0028", "n": 3,
     "detail": "Mamba-YOLO-T over YOLO11n, both from random init, same data and schedule",
     "src": "s3_mamba_t_seed10{1,2,3}.json; YOLO11n from results.csv"},
    {"lever": "40,000 harvested web images",
     "delta": -0.0200, "hi": "0.8636", "lo": "0.8436", "n": 1,
     "detail": "added to the same clean in-domain core; seeds 102/103 in flight",
     "src": "results/framework/s3_tier_v2_{0,40000}.json"},
]

# --- F2 tier ladder ----------------------------------------------------------
LADDER = {
    "rungs": [0, 5000, 15000, 40000],
    "seed101": [0.8636, 0.8599, 0.8614, 0.8436],
    "seed102": PENDING, "seed103": PENDING,
    "jobs": "44397807_[0-3] (seed 101), 45790239_[0-3] (102), 45790291_[0-3] (103)",
    "src": "results/framework/s3_tier_v2_{0,5000,15000,40000}.json",
    "reading": ("Twelve times the training data is flat to +15,000 and then costs 0.020. "
                "The measured seed std for this family is 0.0029."),
}

# --- F3 the generalization wall ---------------------------------------------
WALL = {
    "in_domain": "0.8730 ± 0.0011", "out_domain": "0.1003 ± 0.0053", "n": 3,
    "best_species_in": "Ragweed 0.9604", "best_species_out": "Ragweed 0.0006",
    "target": "ImageWeeds, 3,208 images, CC BY 4.0 — the one harvested source that passed the audit",
    "note": "class-agnostic, same three checkpoints and the same matcher on both sides, so the evaluator offset cancels",
    "src": "results/framework/s6_crossdataset_imageweeds.json",
}

# --- F4 where the harvested data goes ---------------------------------------
FUNNEL = {
    "registry_labelled": 156521, "unique": 59134,
    "cross_dataset_dupes": 44750, "holdout_stems_dropped": 1977,
    "audited_sources": 6, "audited_images": 13527,
    "sources_passing": 1, "passing_images": 3208,
    "bar": 0.90, "failing_range": "0.18-0.74",
    "probe_calibration": "the audit probe reads 1.000 on human-labelled cwd12, so the low scores are the data",
    "src": "figures_data.json -> merge_funnel_2026_08_23, license_sweep_2026_08_23, s1_gate_verdict_2026_08_25",
}

# --- F5 the headline: what each kind of supervision catches ------------------
SUPERVISION = {
    "split": "dev, 149 cases", "incidents": 116, "controls": 33,
    "rows": [
        {"arm": "Scripted watchdog", "reads": "status fields",
         "tp": 0, "recall": 0.000, "fa": 0.000, "cases": 149, "partial": False,
         "note": "flags nothing on any case: \"no signal fired\", 149 times out of 149"},
        {"arm": "Deterministic signals", "reads": "12 pre-registered checks",
         "tp": 11, "recall": 0.095, "fa": 0.061, "cases": 149, "partial": False,
         "note": "17% of the 0.559 ceiling the corpus fixes for any signals-only arm"},
        {"arm": "Open model, raw artifacts", "reads": "artifact excerpts (DeepSeek-V4-Flash)",
         "tp": 45, "recall": 0.388, "fa": 0.091, "cases": 149, "partial": False,
         "note": "floor, not an estimate: a third of its prompts overran the context window"},
        {"arm": "Open model, raw artifacts", "reads": "artifact excerpts (Qwen3.8-27B)",
         "tp": 37, "recall": 0.841, "fa": 0.212, "cases": 107, "partial": True,
         "note": "job 45746472 still running"},
    ],
    "ceiling": 0.559,
    "ceiling_why": "56 of the 127 incidents declare no deterministic signal that could reach them",
    "src": "results/framework/supervision_bench/verdicts/*/ scored against cases/*/truth.json",
}

# --- F6 the fifteen rounds ---------------------------------------------------
ROUNDS = {
    "map": [0.6019, 0.5919, 0.5951, 0.5829, 0.5839, 0.5738, 0.5685, 0.5693,
            0.5672, 0.5711, 0.5589, 0.5665, 0.5521, 0.5594, 0.5607],
    "last": [0.57746, 0.57806, 0.57007, 0.56467, 0.56415, 0.55386, 0.54677, 0.55241,
             0.54768, 0.55501, 0.54593, 0.54534, 0.54687, 0.54985, 0.55146],
    "slope_best": -0.00302, "t_best": -8.86,
    "slope_last": -0.00213, "t_last": -5.52,
    "frozen_from": 3,
    "corpus": "24 datasets, 48,752 unique images, identical from round 3 to round 15",
    "collect_zero_rounds": 8,
    "cold": 0.58053, "warm": 0.55180, "campaign": "0.5577 ± 0.0040",
    "chain_effect": 0.0287, "chain_sigma": 5.0,
    "src": "Mongo round ledger + results/framework/mega_iter*/*/results.csv (41 curves, probe6.json)",
}

# --- the platform: robots in the field ---------------------------------------
# Counted off the live platform 2026-09-11, not from any document. Every number
# here is frames on disk in uploads/, and each is stated as what it is: a frame
# a robot recorded, none of them labelled.
PLATFORM = {
    "total_frames": 2686, "sessions": 26, "robots": 2,
    "labelled": 0,
    "r241_frames": 2259, "r241_field_frames": 1704,
    "r241_span": "2026-08-15 to 2026-09-05", "r241_res": "640 x 360", "r241_hz": 1.0,
    "lasercar_frames": 427, "lasercar_field_frames": 75,
    "lasercar_field_date": "2026-08-29",
    # The premise "GPS and IMU on both robots" does not survive the archive.
    "lasercar_sources": ["detections", "laser", "vehicle", "system", "camera"],
    "lasercar_has_gps": False, "lasercar_has_imu": False,
    "hero": {"slug": "ul_4_09test_49ea7a2a", "robot": "robot241",
             "date": "2026-08-29", "seconds": 213.1,
             "frames": 1013, "res": "640 x 360",
             "gps_fixes": 211, "imu_rows": 3154,
             "telemetry_rows": 3154, "control_rows": 3154,
             "track_m": 140.7,
             "what": "one pass through a crop plot: rows under black plastic mulch, "
                     "weeds between rows, bare soil, tree line"},
    "gps_bug": ("robot_ingest.py preferred the Pi fix, which is frozen to a single "
                "coordinate, over the board fix, which moves. Every trajectory the "
                "platform exposed was one motionless point: 17 sessions, 1,281 rows, "
                "0.0 m. Fixed 2026-09-11; 200.9 m of real track recovered across the "
                "live sessions and 140.7 m in the field drive."),
    "unitree": "no data on the platform yet",
    "src": "~/weed_llm_benchmark/uploads/ (26 session directories), counted on disk",
}

NEVER_PRINT = [
    "0.9033 and 0.8974 ± 0.0040 — no run artifact exists; --seed is a run label only",
    "0.907 — never measured, computed by hand from an evaluator offset",
    "the 15-round slope presented as compounding — n=1 per round, and it declines",
    "the 358-frame robot result as accuracy — those frames contain no weeds",
]
