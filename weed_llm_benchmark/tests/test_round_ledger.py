#!/usr/bin/env python3
"""Round-ledger unit tests (no Mongo / cluster / Roboflow needed).

Exercises the v3.25.0 supervision additions to `tools/db.py`, all pure:
  * db._round_step_entry — the optional params / decided_by / review / su fields
  * db.merge_step_entry  — attempts history (the 2026-08-29 double TIMEOUT left
    one step entry for two burnt jobs) and the ownership rule that keeps a
    tier-1/tier-2/human entry from being overwritten by the scheduler
  * db.render_step_command — the step templates that replaced the scheduler's
    hardcoded sbatch literals; collect and filter must render byte-identical to
    those literals or the loop's behaviour has changed

Run:  python -m pytest tests/test_round_ledger.py   (or) python tests/test_round_ledger.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools import db  # noqa: E402

_fails = []


def ck(name, cond):
    print(("  ok   " if cond else "  FAIL ") + name)
    if not cond:
        _fails.append(name)


def entry(status, actor, now, **kw):
    return db._round_step_entry(status, actor=actor, now=now, **kw)


# ---- _round_step_entry: the old shape is unchanged -----------------------
plain = db._round_step_entry("done", now="T")
ck("entry with no new input keeps the three old keys",
   set(plain) == {"status", "actor", "at"})
ck("entry default actor unchanged", plain["actor"] == "user")


# ---- merge_step_entry: history ------------------------------------------
run1 = entry("running", "round-scheduler", "T1", job="j1")
head1 = db.merge_step_entry(None, run1, "round-scheduler")
ck("a None existing entry works", isinstance(head1, dict))
ck("first write has an empty attempts list", head1["attempts"] == [])
ck("first write keeps the incoming entry",
   head1["status"] == "running" and head1["job"] == "j1" and head1["at"] == "T1")

head2 = db.merge_step_entry(head1, entry("done", "round-scheduler", "T2", job="j1"),
                            "round-scheduler")
ck("running -> done keeps exactly one attempt", len(head2["attempts"]) == 1)
ck("the attempt is the previous head",
   head2["attempts"][0]["status"] == "running" and head2["attempts"][0]["at"] == "T1")
ck("an attempt carries no nested attempts key", "attempts" not in head2["attempts"][0])
ck("the head is the new entry", head2["status"] == "done" and head2["at"] == "T2")

legacy = {"status": "failed", "actor": "round-scheduler", "at": "T0"}   # pre-v3.25.0 doc
head_l = db.merge_step_entry(legacy, run1, "round-scheduler")
ck("a legacy entry without attempts still merges",
   [a["at"] for a in head_l["attempts"]] == ["T0"] and head_l["at"] == "T1")


# ---- merge_step_entry: ownership ----------------------------------------
t1 = entry("failed", "tier1:glm-4.7-flash", "T3", detail="walltime_bound")
t1_head = db.merge_step_entry(head2, t1, "tier1:glm-4.7-flash")
ck("a tier1 write becomes the head", t1_head["actor"] == "tier1:glm-4.7-flash")

after_sched = db.merge_step_entry(t1_head, entry("running", "round-scheduler", "T4", job="j2"),
                                  "round-scheduler")
ck("round-scheduler does not overwrite a tier1 head",
   after_sched["actor"] == "tier1:glm-4.7-flash" and after_sched["at"] == "T3"
   and after_sched["status"] == "failed")
ck("the refused scheduler write is kept as an attempt",
   after_sched["attempts"][-1]["at"] == "T4"
   and after_sched["attempts"][-1]["actor"] == "round-scheduler")
ck("the refused write does not lose the older history",
   len(after_sched["attempts"]) == len(t1_head["attempts"]) + 1)

after_t0 = db.merge_step_entry(t1_head, entry("running", "tier0:gemma4", "T5"), "tier0:gemma4")
ck("tier0 does not overwrite a tier1 head",
   after_t0["at"] == "T3" and after_t0["actor"] == "tier1:glm-4.7-flash")

t2_head = db.merge_step_entry(head2, entry("failed", "tier2:vllm", "T3b"), "tier2:vllm")
after_t2 = db.merge_step_entry(t2_head, entry("running", "round-scheduler", "T4b"),
                               "round-scheduler")
ck("round-scheduler does not overwrite a tier2 head", after_t2["at"] == "T3b")

after_hum = db.merge_step_entry(t1_head, entry("done", "human:owner", "T6"), "human:owner")
ck("human overwrites a tier1 head",
   after_hum["actor"] == "human:owner" and after_hum["status"] == "done"
   and after_hum["at"] == "T6")
ck("the tier1 entry survives as an attempt",
   after_hum["attempts"][-1]["actor"] == "tier1:glm-4.7-flash")

after_user = db.merge_step_entry(t1_head, entry("done", "user", "T6b"), "user")
ck("a plain user actor is not automation and overwrites", after_user["at"] == "T6b")

ck("incoming_actor defaults to the entry's own actor",
   db.merge_step_entry(t1_head, entry("running", "round-scheduler", "T7"))["at"] == "T3")


# ---- merge_step_entry: attempts cap --------------------------------------
h = db.merge_step_entry(None, entry("running", "round-scheduler", "T-0"), "round-scheduler")
for i in range(1, 30):
    h = db.merge_step_entry(h, entry("running", "round-scheduler", "T-%d" % i),
                            "round-scheduler")
ck("attempts cap at 20", len(h["attempts"]) == 20)
ck("the cap drops the oldest, keeps the newest",
   h["attempts"][0]["at"] == "T-9" and h["attempts"][-1]["at"] == "T-28")
ck("the head is still the latest write", h["at"] == "T-29")


# ---- supervision fields survive ------------------------------------------
rich = db._round_step_entry(
    "running", job="j9", actor="tier1:glm-4.7-flash", now="T8",
    params={"epochs": 30, "train_time_cap_h": 10.8},
    decided_by="tier1",
    review={"status": "awaiting", "review_id": "r1", "queued_at": "T8"},
    su=12.5)
ck("params recorded", rich["params"]["epochs"] == 30)
ck("decided_by recorded", rich["decided_by"] == "tier1")
ck("review recorded", rich["review"]["status"] == "awaiting")
ck("su recorded as a float", isinstance(rich["su"], float) and rich["su"] == 12.5)

m = db.merge_step_entry(head2, rich, "tier1:glm-4.7-flash")
ck("params, decided_by, review and su survive the merge",
   m["params"]["train_time_cap_h"] == 10.8 and m["decided_by"] == "tier1"
   and m["review"]["review_id"] == "r1" and m["su"] == 12.5)
mm = db.merge_step_entry(m, entry("running", "round-scheduler", "T9"), "round-scheduler")
ck("a refused scheduler write does not drop the review state",
   mm["review"]["review_id"] == "r1" and mm["decided_by"] == "tier1")
ck("su survives as an attempt field",
   db.merge_step_entry(m, entry("done", "human:owner", "T10"),
                       "human:owner")["attempts"][-1]["su"] == 12.5)

ck("a non-numeric su is dropped rather than stored",
   "su" not in db._round_step_entry("done", now="T", su="lots"))


# ---- default config: the new blocks --------------------------------------
D = db.DEFAULT_DOMAIN_CONFIG
ck("default config carries step templates",
   set(D["steps"]) == {"collect", "filter", "train"})
ck("default round_params carry the trainer time cap",
   D["round_params"]["train_time_cap_h"] == 10.8)
ck("default round_params keep today's epochs/imgsz/patience",
   D["round_params"]["epochs"] == 60 and D["round_params"]["imgsz"] == 640
   and D["round_params"]["patience"] == 20)
ck("brain policy defaults to the scripted baseline", D["brain"]["policy"] == "scripted")
ck("no tier is wired by default", set(D["brain"]["tiers"].values()) == {""})
ck("budget envelope present", D["budget"]["su_envelope"] == 1500)
ck("noise floor is per recipe",
   D["noise_floor"]["merged_curated"] == 0.005 and D["noise_floor"]["merged_raw"] == 0.009
   and D["noise_floor"]["cwd12_core"] == 0.006)


# ---- render_step_command: byte-identical to the v3.24 literals ------------
# These two strings are round_scheduler._WEED_STEPS as of v3.24.11. If a render
# stops matching, the unattended loop's submitted command has changed.
COLLECT = ("sbatch --time=10:00:00 --export=ALL,BRAIN_MAX_NEW=3 "
           "run_v3_0_43_brain_harvest_oneshot.sh")
FILTER = "sbatch run_s2_dino_scores.sh"
ck("collect renders byte-identical to the hardcoded literal",
   db.render_step_command(D, "collect") == COLLECT)
ck("filter renders byte-identical to the hardcoded literal",
   db.render_step_command(D, "filter") == FILTER)

train = db.render_step_command(D, "train", iter_name="rnd5_train_44900001",
                               domain="weed", round=5,
                               trace="/x/_brain/weed/trace/t.jsonl")
ck("train keeps the h100 gres and the array", "--gres=gpu:h100-80:1" in train
   and "--array=1-1" in train)
ck("train keeps the 12 h walltime", "--time=12:00:00" in train)
ck("train carries the trainer time cap", "TRAIN_TIME_H=10.8" in train)
ck("train carries the epoch budget", "TRAIN_EPOCHS=60" in train)
ck("train carries the job-scoped iteration name",
   "ITER_NAME=rnd5_train_44900001" in train)
ck("train carries the trace path",
   "BRAIN_TRACE=/x/_brain/weed/trace/t.jsonl" in train and "BRAIN_DOMAIN=weed" in train)
ck("train ends with the job script", train.endswith(" run_m1_merged_seeds.sh"))
# The job script has always read BRAIN_ROUND and the template never exported it,
# so every per-epoch trace record carried an empty round and could only be grouped
# by job id. The field is bounded in the policy catalogue for the same reason it is
# rendered here: the gate authorises exactly what a template substitutes.
ck("train carries the round number the job script reads",
   "BRAIN_ROUND=5" in train)

_missing = ""
try:
    db.render_step_command(D, "train")            # no iter_name / domain / trace
except KeyError as e:
    _missing = str(e)
ck("a missing placeholder raises KeyError naming the field", "iter_name" in _missing)
ck("the KeyError message names the step", "'train'" in _missing)

_unknown = False
try:
    db.render_step_command(D, "notastep")
except KeyError:
    _unknown = True
ck("an unknown step raises KeyError", _unknown)

_nocfg = False
try:
    db.render_step_command({}, "collect")
except KeyError:
    _nocfg = True
ck("a config without templates raises KeyError", _nocfg)

cfg2 = db._deep_merge(D, {"round_params": {"collect_time_h": 4}})
ck("a config override changes exactly one field",
   db.render_step_command(cfg2, "collect") == COLLECT.replace("10:00:00", "4:00:00"))
ck("overriding does not mutate the defaults",
   db.render_step_command(D, "collect") == COLLECT)


# ---- the step-command allow-list is the choke point to a cluster shell ----
# It has to hold on any machine: the scheduler renders on the always-on server
# while the scripts live in the cluster checkout, so a check that reads the local
# filesystem refused every legitimate template there and silently stopped the loop.
ck("a rendered collect command is submittable",
   db.validate_step_command(db.render_step_command(D, "collect"))[0])
ck("a rendered filter command is submittable",
   db.validate_step_command(db.render_step_command(D, "filter"))[0])
ck("a rendered train command is submittable",
   db.validate_step_command(db.render_step_command(
       D, "train", round=9, domain="weed", iter_name="rnd9_train", trace=""))[0])
for _cmd, _why in (
        ("sbatch run_m1_merged_seeds.sh; rm -rf /", "chaining"),
        ("sbatch --export=ALL,X=$(curl evil) run_m1_merged_seeds.sh", "substitution"),
        ("sbatch run_m1_merged_seeds.sh | tee /tmp/x", "piping"),
        ("sbatch run_m1_merged_seeds.sh > /tmp/x", "redirection"),
        ("python run_m1_merged_seeds.sh", "a non-sbatch submitter"),
        ("sbatch run_evil.sh", "a script outside the allow-list"),
        ("sbatch run_deploy_model.sh", "a real script the scheduler may not submit"),
        ("sbatch --export=ALL run_m1_merged_seeds.SH", "a name outside the shape"),
        ("", "an empty command"),
):
    ck("refuses %s" % _why, not db.validate_step_command(_cmd)[0])
ck("every allow-listed script matches the script shape",
   all(db._STEP_SCRIPT_RE.fullmatch(x) for x in db._STEP_ALLOWED_SCRIPTS))
ck("a steps patch naming a script outside the allow-list is refused",
   not db._validate_steps_patch(D, {"train": "sbatch run_evil.sh"})[0])
ck("a steps patch choosing its script by field is refused",
   not db._validate_steps_patch(D, {"train": "sbatch {tier}"})[0])
ck("a legitimate steps patch is accepted",
   db._validate_steps_patch(D, {"filter": "sbatch run_s2_dino_scores.sh"})[0])


# ---- record_round_step still guards its inputs without Mongo -------------
ck("record_round_step rejects an unknown step",
   db.record_round_step("coral", "notastep", "done") is None)


if _fails:
    print(f"\nFAILED: {len(_fails)} -> {_fails}")
    sys.exit(1)
print("\nALL PASS")
