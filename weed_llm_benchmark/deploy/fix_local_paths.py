"""Rewrite registry local_path from the cluster repo prefix to this host's
REPO_ROOT, then re-mirror to Mongo. Idempotent. Run after pulling the registry
from the cluster so the lab dashboard finds images at lab paths.
  REPO_ROOT=~/weed_llm_benchmark CLUSTER_REPO=/ocean/.../weed_llm_benchmark python deploy/fix_local_paths.py
"""
import json, os
from weed_optimizer_framework.tools import db

REPO = os.environ.get("REPO_ROOT", os.path.expanduser("~/weed_llm_benchmark"))
CLUSTER = os.environ.get("CLUSTER_REPO",
                         "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark")
regp = os.path.join(REPO, "results", "framework", "dataset_registry.json")
reg = json.load(open(regp))
ds = reg.get("datasets") or {}
n = 0
for slug, info in ds.items():
    lp = info.get("local_path", "") or ""
    if lp.startswith(CLUSTER):
        info["local_path"] = REPO + lp[len(CLUSTER):]
        n += 1

# v3.0.125 (Z1): re-inject lab-local MANUAL UPLOADS. The cluster→lab sync just
# overwrote dataset_registry.json with the cluster's copy, which has no manual
# uploads (those originate on the lab). Merge them back from the durable
# manual_uploads.json so user-uploaded datasets survive every sync. Existing
# registry entries win only for cluster-origin slugs; manual_upload slugs are
# authoritative from the json file.
mu_path = os.path.join(REPO, "results", "framework", "manual_uploads.json")
n_manual = 0
try:
    if os.path.isfile(mu_path):
        manual = json.load(open(mu_path)) or {}
        for slug, info in manual.items():
            # keep the upload's lab-local path as-is (already a lab path)
            ds[slug] = {**ds.get(slug, {}), **info}
            n_manual += 1
        reg["datasets"] = ds
        reg["total_downloaded"] = sum(
            1 for v in ds.values() if v.get("status") == "downloaded")
except Exception as e:
    print(f"manual_uploads merge skipped: {e}")

# v3.24.6: count existence over the FINAL dataset dict, after the manual-upload
# merge. It used to be counted inside the rewrite loop above — before the merge
# added the manual slugs — while the denominator was taken after, so the two
# halves of "X/Y exist" came from different populations. Within one sync run the
# first invocation (fresh cluster registry) skipped the 15 manual slugs and the
# second (already-merged file) included them, which made the printed count jump
# 63/78 → 78/78 → 63/78 across runs and read like data appearing and vanishing
# during the 2026-08-25 outage audit. (Note `isdir` is a weak signal either way:
# rsync creates a dataset's directory before its contents finish arriving.)
exists = sum(1 for v in ds.values()
             if os.path.isdir(v.get("local_path", "") or ""))
json.dump(reg, open(regp, "w"))
print(f"rewrote {n}/{len(ds)} local_path -> lab; now {exists}/{len(ds)} exist on disk; "
      f"re-injected {n_manual} manual upload(s)")
res = db.mirror_registry_to_mongo(reg)
print("mongo re-mirror:", res)
