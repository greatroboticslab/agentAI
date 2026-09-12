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
    # Seeds 102 and 103. The two completed rungs carry their own error bar now;
    # +15,000 and +40,000 are still running and stay PENDING.
    # Complete: three seeds at every rung, the same images at each seed.
    "seeds": {0: [0.8636, 0.8610, 0.8664], 5000: [0.8599, 0.8649, 0.8579],
              15000: [0.8614, 0.8526, 0.8600], 40000: [0.8436, 0.8439, 0.8469]},
    # Arm B of the class-space experiment, complete: every harvested box rewritten
    # to one shared class instead of a hash of its source dataset.
    "armB": [0.8636, 0.8538, 0.8451, 0.8252],
    "jobs": "44397807_[0-3] (seed 101), 45790239_[0-3] (102), 45790291_[0-3] (103)",
    "src": "results/framework/s3_tier_v2_{0,5000,15000,40000}.json",
    "reading": ("Twelve times the training data costs 0.0189 at +40,000 -- 8.2 pooled standard "
                "deviations, with three seeds at every rung."),
}

# --- the second exam --------------------------------------------------------
# The tier ladder is measured on CottonWeedDet12's own holdout, and that metric
# rewards training data that resembles CottonWeedDet12, so a greenhouse/aerial/
# three-season corpus can only ever cost. Scoring the SAME eight checkpoints on a
# second dataset removes that objection. Both exams fall.
SECOND_EXAM = {
    "target": "ImageWeeds, 3,208 images, class-agnostic, 0 excluded by the leak check",
    "armA": {0: 0.1026, 5000: 0.0725, 15000: 0.0825, 40000: 0.0728},
    "armB": {0: 0.1026, 5000: 0.0964, 15000: 0.0938, 40000: 0.0927},
    "cwd12_delta": -0.0189, "iw_delta_A": -0.0299, "iw_delta_B": -0.0100,
    "job": "45817696",
    "src": "results/framework/s6_crossdataset_ladder.json",
    "reading": ("Adding harvested data lowers both exams. The narrow metric is not what makes the "
                "ladder fall -- the data does not help on a broad one either."),
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
import json
import os
import statistics as st

# --- F5 the headline: what each kind of supervision catches ------------------
# Every number below is read from supervision_table.json, which is the project's
# own scorer (`bench reproduce`, no model call) re-scoring the committed verdicts
# on the dev split. An earlier draft used a counter written for the poster, which
# scored a case as detected whenever the arm raised anything. The committed scorer
# differs in two ways that both matter: a detection requires an `issue` verdict
# carrying a finding AT OR ABOVE A SEVERITY BAR, and a case that produced no
# answer leaves the denominator instead of counting as a miss. It reads between
# 0.06 and 0.29 lower per arm. `detection_grounded` is the same thing with the
# further requirement that the finding quote a line resolving in the artifact.
_ST = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "supervision_table.json")))


def _arm(key, field):
    a = _ST["arms"].get(key) or {}
    return (a.get(field) or {}).get("v"), (a.get(field) or {}).get("n"), a.get("counts", {})


SUPERVISION = {
    "split": "dev, 149 cases", "incidents": 116, "controls": 33,
    "scorer": "bench reproduce, run %s" % _ST["run_id"],
    # Tiers, in the order the poster reads them:
    #   A0   the scheduler's own watchdog, reading status fields
    #   A0p  twelve pre-registered deterministic checks
    #   L2   a model reading raw artifact excerpts
    #   L3   the same model, plus a retrieval round over the artifacts
    "tiers": [
        {"key": "A0", "label": "Scripted watchdog", "reads": "status fields"},
        {"key": "A0p", "label": "Deterministic signals", "reads": "12 pre-registered checks"},
    ],
    # Ordered small to large. The 7 B arm is the size class the campaign's own
    # reviewer ran on for a week, so it is the one that has to be on the figure.
    "models": [
        {"name": "Qwen2.5-7B", "size": "7 B",
         "l2": "L2@qwen2.5:7b", "l3": "L3@qwen2.5:7b"},
        {"name": "Qwen3-14B", "size": "14 B",
         "l2": "L2@qwen3:14b", "l3": "L3@qwen3:14b"},
        {"name": "Qwen3.8-27B", "size": "27 B",
         "l2": "L2@qwen3.8:27b", "l3": "L3@qwen3.8:27b"},
        {"name": "GLM-4.7-Flash", "size": "30 B",
         "l2": "L2@glm-4.7-flash", "l3": None},
    ],
    "table": _ST,
    # A0 is the one arm the scorer cannot score: "no signal fired" is not a
    # decision, so all 149 of its verdicts come back undecidable. That is the
    # finding, not a gap in the measurement -- the arm the pipeline actually ran
    # never produced a judgement about any of the 116 incidents.
    "a0_undecidable": _ST["arms"]["A0"]["counts"]["undecidable"],
    "a0_cases": _ST["arms"]["A0"]["counts"]["cases"],
    "ceiling": 0.559,
    "ceiling_why": "56 of the 127 incidents declare no deterministic signal that could reach them",
    "excluded": {"model": "DeepSeek-V4-Flash (284B)",
                 "why": "its verdicts predate the 2026-09-07 split re-cut, so the scorer "
                        "drops them rather than mixing two corpora"},
    "src": "results/framework/supervision_rescore/results/run_reproduce-*.json "
           "(bench reproduce over results/framework/supervision_bench/verdicts/)",
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
    # The matched control: same data, same 30 epochs, same complete cosine, seed
    # 101 both arms. Only the starting weights differ, so the gap between them is
    # the warm-start chain and nothing else.
    "cold": 0.58053,            # results/framework/ctl_chain_armA_45672628.json
    "warm": 0.55180,            # results/framework/ctl_chain_armB_45672628.json
    # The recipe the campaign actually ran -- warm start, epochs=60, patience=20,
    # time=10.0 -- repeated over three seeds. Its spread is the only seed-noise
    # measurement this project owns, so every sigma below is quoted against it.
    "recipe_seeds": [0.5607, 0.55915, 0.55309],   # seed 101 is round 15 itself
    "recipe_src": "round 15 + ctl_seed_s102/s103_45672672.json",
    "src": "Mongo round ledger + results/framework/mega_iter*/*/results.csv (41 curves, probe6.json)",
}

# Derived from the measurements above so the figure, the table and the prose
# cannot drift apart. sigma is always the round recipe's own seed spread.
_rs = ROUNDS["recipe_seeds"]
ROUNDS["recipe_mean"] = st.mean(_rs)
ROUNDS["recipe_sd"] = st.stdev(_rs)
# Start effect: two single runs, so the noise on their difference is sd * sqrt(2).
ROUNDS["chain_effect"] = ROUNDS["cold"] - ROUNDS["warm"]
ROUNDS["chain_sigma"] = ROUNDS["chain_effect"] / (ROUNDS["recipe_sd"] * 2 ** 0.5)
# Schedule effect: one run against the mean of three.
ROUNDS["sched_effect"] = ROUNDS["warm"] - ROUNDS["recipe_mean"]
ROUNDS["sched_sigma"] = ROUNDS["sched_effect"] / (
    (ROUNDS["recipe_sd"] ** 2 + (ROUNDS["recipe_sd"] / 3 ** 0.5) ** 2) ** 0.5)

# --- the audit, source by source --------------------------------------------
# The funnel says one of six sources clears the bar. This is which six, and by
# how much they miss, which is what makes the funnel an audit result rather than
# an assertion. The probe reads 1.000 on human-labelled CottonWeedDet12, so a
# source scoring 0.18 is the data and not the instrument.
SOURCES = {
    "bar": 0.90,
    "calibration": 1.000,
    "rows": [("ImageWeeds \u00b7 weed detection", 1.0000),
             ("AgML \u00b7 weed / crop detection", 0.7371),
             ("ImageWeeds \u00b7 aerial", 0.6496),
             ("Roboflow \u00b7 grass weeds", 0.5775),
             ("AgML \u00b7 MH weed16", 0.4519),
             ("Roboflow \u00b7 weed / crop aerial", 0.1845)],
    "src": "figures_data.json -> s1_gate_verdict_2026_08_25.per_source",
}

# --- what inference-time compute can buy ------------------------------------
# All six arms scored by ONE matcher, the plain baseline included, so the numbers
# are differences within one instrument rather than across two. The seed-noise bar
# is the same 0.006 the rest of the poster uses. The latency is why this is a
# ceiling and not a recommendation.
TTA = {
    "baseline": 0.8554,
    "arms": [("weighted box fusion alone", 0.8524),
             ("+ multi-scale", 0.8653),
             ("+ multi-scale and h-flip", 0.8728),
             ("3-seed ensemble", 0.8726),
             ("both, 18 views", 0.8830)],
    "seed_noise": 0.006,
    "s_per_image": 2.44, "deployed_ms": 3.7,
    "matcher": "wbf_tta_eval.compute_map, shared across all arms including the baseline",
    "src": "figures_data.json -> tta_ceiling_2026_08_26 (jobs 44463762, 44463922)",
}

# --- the field gap: does the deployed detector fire on our own frames? -------
# The cross-dataset wall is measured against another labelled corpus. This is the
# same question asked of the corpus the platform actually holds, which has no
# labels -- so it measures FIRING, not recall, and the poster says so. The
# vegetation rule exists so "there was nothing to detect" cannot explain the
# answer away. Restricted to robot-recorded sessions: the live uplink plus the
# bulk-uploaded field drive, with the drive's byte-identical second copy dropped.
_FF = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "s6_field_fire.json")))
_FF_DUP = "ul_4_09test_c5a52917"
_FF_ROBOT = {k: v for k, v in _FF["sessions"].items()
             if (k.startswith("rl_") or k.startswith("ul_4_09test")) and k != _FF_DUP}


def _ff_sum(field, conf=None):
    if conf is None:
        return sum(v[field] for v in _FF_ROBOT.values())
    return sum(v["at_conf_%.2f" % conf][field] for v in _FF_ROBOT.values())


FIELD = {
    "model": _FF["model"],
    "sessions": len(_FF_ROBOT),
    "frames": _ff_sum("frames"),
    "vegetated": _ff_sum("vegetated_frames"),
    "fired_25": _ff_sum("frames_firing", 0.25),
    "fired_40": _ff_sum("frames_firing", 0.40),
    "fired_60": _ff_sum("frames_firing", 0.60),
    "veg_fired_25": _ff_sum("veg_frames_firing", 0.25),
    "species": {"Purslane": 24, "Sicklepod": 1},
    "rule": _FF["vegetation_rule"],
    "caveat": "no ground truth exists for these frames, so this is firing behaviour, "
              "not recall and not precision",
    "src": "results/framework/s6_field_fire.json "
           "(weed_optimizer_framework/tools/field_fire_sweep.py, lab RTX 3060)",
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
