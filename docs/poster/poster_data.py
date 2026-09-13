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
    # Two authors, one department. Song Cui was on an earlier draft and is not
    # an author of this work; Harry said so on 2026-09-12 and it is his paper.
    "authors": [("Harry He", ""), ("Hongbo Zhang", "")],
    "affiliations": ["Department of Engineering Technology"],
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

# --- the six months, in order -----------------------------------------------
# Eight turns, each dated from the CHANGELOG entry that records it. A project
# with no wrong turns in it is not a project, and the wrong turns here are the
# part a professor can learn something from.
JOURNEY = [
    ("Mar", "Prompt a vision model", "Twelve open vision-language models shortlisted and "
     "asked to box weeds; fourteen were benchmarked in the end. The best, "
     "Florence-2-base, reached 0.434 where a fine-tuned YOLO11n reached 0.929.",
     "Fine-tune. Do not prompt."),
    ("Mar", "Fix the forgetting", "Accuracy on unseen species fell from F1 0.830 "
     "to 0.606. Every anti-forgetting method we tried left it there. 27.4% of the "
     "pseudo-labels were false positives.",
     "The labels were wrong, not the optimiser."),
    ("Apr", "Harvest the web", "An agent searching Kaggle, Hugging Face, GitHub and "
     "Roboflow, aiming at hundreds of thousands of images. Accuracy went down.",
     "Volume is not supervision."),
    ("May", "Withdraw 0.910", "Pretrain on 244,000 then fine-tune, and the holdout "
     "read 0.910. 2,313 of 141,397 training images were near-duplicates of it.",
     "Build the content-level guard, then measure everything again."),
    ("May", "Buy the last 0.012", "Five ways to lift 0.8877 to 0.90: ensembles, "
     "test-time augmentation, weighted box fusion. All five negative.",
     "The ceiling was seed noise. Stop tuning the number."),
    ("Jun", "Build the platform", "Robots, an uplink, a dataset registry, a "
     "dashboard, roles and keys. Then an audit of our own artifacts.",
     "The dashboards were green while the harvest collected nothing."),
    ("Aug", "Run the control on our own thesis", "Fifteen unattended rounds, "
     "accuracy falling the whole time. The training corpus never grew after round "
     "three, and starting fresh instead of from the last round is worth +0.0287.",
     "We had been measuring a chain, not a curve."),
    ("Sep", "Build the supervisor", "A benchmark of 162 real failures out of this "
     "project's own record. The watchdog we had been running returned no decision "
     "on any of 149 cases.", "Status fields cannot see this. Artifacts can."),
]

# --- the model ledger: everything that was trained, run or rejected ---------
# The figure draws this list, so the count on the poster is whatever the list is
# and cannot be wrong. Each row: name, what it did here, what it scored, and the
# outcome. Every score comes from a block of docs/RESULTS_TABLE.md named in src.
LEDGER = {
    "groups": [
        {"group": "Detectors we trained on our own holdout",
         "note": "sealed 1,977-image CottonWeedDet12 holdout",
         "rows": [
             ("YOLO11n, COCO-pretrained", "0.8755 \u00b1 0.0029", "3 seeds", "deployed"),
             ("Mamba-YOLO-T, from scratch", "0.8266 \u00b1 0.0064", "3 seeds", "measured"),
             ("YOLO11n, from scratch", "0.8041 \u00b1 0.0028", "3 seeds", "control"),
             ("RF-DETR Large, pretrained", "0.8974 \u00b1 0.0040", "4 runs, unseeded",
              "pycocotools, not the scale above"),
             ("yolo26x", "0.6019 \u2192 0.5607", "15 rounds", "campaign backbone"),
         ]},
        {"group": "Vision models inside the pipeline",
         "note": "not scored as detectors; they do a job in the loop",
         "rows": [
             ("OWLv2-large", "recall 0.943, precision 0.194", "single pass", "pseudo-labeller"),
             ("DINOv2", "quality gate at 0.50", "per dataset", "collection filter"),
         ]},
        {"group": "Vision-language models, zero-shot on the same images",
         "note": "one deterministic pass each, no repeats, 848-image split",
         "rows": [
             ("Florence-2-base", "0.434", "mAP50", "best of the zero-shot set"),
             ("Florence-2-large", "0.329", "mAP50", ""),
             ("InternVL2-8B", "0.208", "mAP50", ""),
             ("Qwen2.5-VL-3B", "0.196", "mAP50", ""),
             ("MiniCPM-V-4.5", "0.192", "mAP50", ""),
             ("OWLv2-large", "0.184", "mAP50", "high recall, low precision"),
             ("Qwen2.5-VL-7B", "0.176", "mAP50", ""),
             ("InternVL2-2B", "0.002", "mAP50", ""),
             ("InternVL2.5-8B", "0.000", "mAP50", ""),
             ("G-DINO, Molmo, Llama-Vision,\nMoondream, LLaVA", "\u2248 0.000", "5 models", "no usable grounding"),
         ]},
        {"group": "Language models we ran as reviewers",
         "note": "162 real incidents from this project's own history",
         "rows": [
             ("Qwen2.5-7B", "0.823 recall, 0.274 grounded", "149 cases", "the lab tier"),
             ("Qwen3-14B", "0.675 recall, 0.614 grounded", "149 cases", ""),
             ("Qwen3.8-27B", "0.702 recall, 0.702 grounded", "92 of 149", "best evidenced"),
             ("GLM-4.7-Flash", "0.821 recall, 0.769 grounded", "78 of 149", "still running"),
             ("DeepSeek-V4-Flash", "retracted", "context overflow", "withdrawn"),
             ("DeepSeek-V3-671B", "queued", "\u2014", "not yet run"),
             ("Gemma 4", "\u2014", "harvest decisions", "in the loop"),
             ("Qwen2.5-7B (curation)", "\u2014", "dataset judgements", "in the loop"),
             ("Qwen2.5-3B", "\u2014", "lab guide tier", "not authoritative"),
         ]},
    ],
    "src": "docs/RESULTS_TABLE.md blocks A, C, F, G and L; figures_data.json",
}
# Derived so the headline count cannot disagree with the list under it. Two
# models do two jobs here -- OWLv2-large is both the pseudo-labeller and a
# benchmarked zero-shot detector, and Qwen2.5-7B is both a reviewer arm and the
# curation judge -- so the distinct count is lower than the row count and the
# poster prints the distinct one.
def _ledger_names():
    seen = []
    for g in LEDGER["groups"]:
        for r in g["rows"]:
            if "5 models" in r[2]:
                seen += [n.strip() for n in r[0].replace("\n", " ").split(",")]
            else:
                seen.append(r[0].split(" (")[0].split(",")[0].strip())
    return seen


LEDGER["n_rows"] = sum(len(g["rows"]) for g in LEDGER["groups"])
LEDGER["n_models"] = len({n.lower() for n in _ledger_names()})
LEDGER["n_deployed"] = 1
LEDGER["n_groups"] = len(LEDGER["groups"])

# --- what the platform itself does, counted on the live box ------------------
# Measured on lab-b660m-c 2026-09-13 by reading the service journal, the upload
# archive and the analysis artifacts. Where a mining pass and a direct count
# disagreed, the direct count is what is here: the joystick total is 1,389 from
# the journal since 2026-08-01, not the 1,327 a narrower window gave.
AGENT = {
    # remote control -- real, and dark today
    "drive_cmds": 1389, "drive_robot": "robot 241", "drive_last": "2026-08-28",
    "drive_targets_241": 1, "drive_targets_cart": 0,
    "advice_polls": 406,
    # the laser: switched on, never once with a target
    "laser_rows": 3379, "laser_sessions": 8, "laser_on_rows": 216,
    "laser_phases_when_on": {"OBSERVATION": 216},
    # the analysis agent
    "tools": 14, "chat_datasets": 9, "chat_turns": 92,
    "auto_eda_sessions": 26, "auto_eda_plots": 45,
    "agent_turns_on_robot_sessions": 0,
    "sandbox_cpu_s": 15, "sandbox_ram_gb": 1.5, "sandbox_timeout_s": 25,
    "deep_turns": 1, "byok_turns": 1,
    "src": "lab journal + uploads/rl_lasercar-*/laser.csv + "
           "results/framework/dataset_analysis/ ; counted 2026-09-13",
}

# --- the platform census, walked rather than claimed -------------------------
# Every number here was counted by walking ~/weed_llm_benchmark/uploads on the
# lab server on 2026-09-12, not read out of a document. The byte-identical
# second copy of the field drive (ul_4_09test_c5a52917) is excluded everywhere.
CENSUS = {
    "frames": 2686, "sessions": 26, "labelled": 0,
    "r241_frames": 2238, "r241_sessions": 17,
    "cart_frames": 427, "cart_sessions": 8, "cart_field_frames": 75,
    "smoke_frames": 21, "smoke_sessions": 1,      # a post-rewire uplink test
    # Nine telemetry streams, counted line by line across every session.
    "sensor_rows": 91955,
    "streams": [("telemetry", 23884, 18), ("control", 23884, 18), ("imu", 23884, 18),
                ("witimu", 10055, 8), ("laser", 3379, 8), ("detections", 3379, 8),
                ("vehicle", 1762, 8), ("gps", 1502, 18), ("system", 226, 8)],
    # Haversine over consecutive fixes, board receiver preferred, steps over 25 m
    # dropped as receiver jumps.
    "gps_m": 342.0, "gps_fixes": 1502, "gps_sessions": 18,
    "r241_res": "640 x 360", "r241_hz": 1.0,
    "hero_m": 140.9, "hero_fixes": 211,
    # The cart writes four telemetry streams and neither of them is position or
    # attitude. "GPS and IMU on both robots" was a premise, and it is false.
    "cart_streams": ["detections", "laser", "system", "vehicle"],
    "cart_has_gps": False, "cart_has_imu": False,
    "src": "walked uploads/ on lab-b660m-c 2026-09-12; gps distance recomputed "
           "from the nested data.board_lat/board_lon fields",
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
    # r241_frames lived here and was superseded by CENSUS on 2026-09-12: it folded a
    # 21-frame uplink test into the robot's own total. Read CENSUS.
    "r241_field_frames": 1704,
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
