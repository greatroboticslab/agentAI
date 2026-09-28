#!/usr/bin/env python3
"""run_inc_funnel.sh, the funnel audit's sbatch wrapper
(docs/FUNNEL_AUDIT_RUNNER.md 5.6.2).

Static checks read the script; the rest run it with bash in a temporary REPO
whose weed_llm_benchmark/ is a link to this nested copy, with the test hooks
the script offers (INC_FUNNEL_REPO, INC_FUNNEL_CONDA_SH, INC_FUNNEL_DRY_RUN=1:
every check runs, then the command that would run is printed and no server
starts), a fake conda.sh, a fake nvidia-smi on PATH (its GPU list set per
case) and `python` pointing at this interpreter. A module the import check
needs that is not installed here (open_clip, as a rule) is stubbed for the
GPU cases, and the test says so.

Pinned:
  * bash -n passes; the #SBATCH lines are the runner's (GPU-shared, one
    V100-32, 5 CPUs, 45G, 8 h, the log under funnel/logs);
  * the verb case list is the CLI's VERBS without fetch and sbatch-args; the
    verb class comes from the CLI's VERB_CLASS;
  * a dry run of census: exit 0, verb class cpu, the sha256 of every file
    under the funnel package (as walked here) and of the fixed INC and tools
    modules (mega_trainer.py, the dHash, included), each equal to the file's,
    and the python command with the arguments unchanged; recover also checks
    torch and transformers (its H6 detector);
  * refusals (exit 2): no verb, an unknown verb, fetch, sbatch-args; a GPU
    verb with no GPU listed, printing the sbatch line to use; rl-b on a 32 GB
    GPU; an array job of any verb but embed-judges --stage embed; a missing
    contract; an outer $REPO/run_inc_funnel.sh that differs from the nested
    copy (an identical one runs);
  * an array task of embed-judges --stage embed gets --shard <task id>
    (also with --stage=embed), and an explicit --shard is kept;
  * rl-b on an 80 GB GPU: the model tag is the domain config's RL-B model
    (read here from the config), added as --model, and FUNNEL_OLLAMA_ENDPOINT
    is the per-job port 8000 + job id % 1000;
  * the ollama block mirrors run_inc_plan.sh: the same exported OLLAMA_*
    settings, host and port formula, serve binary, and two warm-up attempts
    bounded at 1200 s;
  * the 80 GB threshold in MiB is ceil(80e9 / 2**20).

No network, no GPU, no Slurm.

Run:  python3 tests/test_funnel_job_script.py
"""
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
GIT = ROOT.parent
SCRIPT = ROOT / "run_inc_funnel.sh"
PLAN = ROOT / "run_inc_plan.sh"
FUN = ROOT / "weed_optimizer_framework" / "tools" / "funnel"
TMP = pathlib.Path(tempfile.mkdtemp(prefix="funnel_job_"))
os.environ["INC_DIR"] = str(TMP / "inc_unused")
os.environ["REPO"] = str(TMP / "repo_unused")
sys.path.insert(0, str(ROOT))

from weed_optimizer_framework.tools.funnel import __main__ as M  # noqa: E402

FAILURES = []
SKIPS = []
MOD = "weed_optimizer_framework.tools.funnel"
FIXED_MODULES = ("tools/inc/__init__.py", "tools/inc/common.py", "tools/inc/verify.py", "tools/inc/select.py",
                 "tools/inc/relevance.py", "tools/inc/driver.py", "tools/inc/gate.py", "tools/cwd12_species.py",
                 "tools/near_dup.py", "tools/semisup_labeler.py", "tools/mega_trainer.py")


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------- the sandbox
REPO = TMP / "repo"
(REPO / "docs").mkdir(parents=True)
os.symlink(ROOT, REPO / "weed_llm_benchmark")
shutil.copyfile(GIT / "docs" / "FUNNEL_AUDIT.md", REPO / "docs" / "FUNNEL_AUDIT.md")
INC = REPO / "results" / "framework" / "inc"
(INC / "funnel").mkdir(parents=True)
PREREG = INC / "funnel" / "prereg_v1.json"
shutil.copyfile(ROOT / "results" / "framework" / "inc" / "funnel" / "prereg_v1.json", PREREG)
BIN = TMP / "bin"
BIN.mkdir()
os.symlink(sys.executable, BIN / "python")
(BIN / "nvidia-smi").write_text("#!/bin/bash\n[ -n \"${FAKE_GPUS:-}\" ] || exit 9\nprintf '%s\\n' \"$FAKE_GPUS\"\n")
(BIN / "nvidia-smi").chmod(0o755)
CONDA = TMP / "conda.sh"
CONDA.write_text("conda() { return 0; }\n")
STUBS = TMP / "stubs"
STUBBED = []
for m in ("torch", "transformers", "open_clip"):
    if importlib.util.find_spec(m) is None:
        (STUBS / m).mkdir(parents=True)
        (STUBS / m / "__init__.py").write_text("__version__ = 'stub for test_funnel_job_script'\n")
        STUBBED.append(m)
V100 = "Tesla V100-SXM2-32GB, 32768"
H100 = "NVIDIA H100 80GB HBM3, 81559"
RUN_SCRIPT = REPO / "weed_llm_benchmark" / "run_inc_funnel.sh"


def job(args, gpus="", env=None, script=RUN_SCRIPT):
    """(exit code, stdout, stderr) of bash run_inc_funnel.sh args in the sandbox."""
    e = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    e.update({"PATH": "%s:%s" % (BIN, os.environ.get("PATH", "")), "INC_FUNNEL_REPO": str(REPO),
              "INC_FUNNEL_CONDA_SH": str(CONDA), "INC_FUNNEL_DRY_RUN": "1", "FAKE_GPUS": gpus,
              "PYTHONPATH": str(STUBS)})
    e.pop("INC_DIR", None)
    e.pop("REPO", None)
    e.update(env or {})
    r = subprocess.run(["bash", str(script)] + list(args), capture_output=True, text=True, env=e, timeout=300)
    return r.returncode, r.stdout, r.stderr


def dry_line(out):
    lines = [ln for ln in out.splitlines() if ln.startswith("DRY RUN: ")]
    return lines[-1][len("DRY RUN: "):] if lines else None


# ----------------------------------------------------------------- static
def test_static():
    print("static")
    r = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    check("bash -n passes", r.returncode == 0, r.stderr)
    text = SCRIPT.read_text()
    sb = [ln for ln in text.splitlines() if ln.startswith("#SBATCH")]
    check("the #SBATCH lines are the runner's",
          sb == ["#SBATCH --job-name=inc_funnel", "#SBATCH --partition=GPU-shared", "#SBATCH --gres=gpu:v100-32:1",
                 "#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=5", "#SBATCH --mem=45G", "#SBATCH --time=08:00:00",
                 "#SBATCH --output=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/inc/"
                 "funnel/logs/%x_%j.out"], sb)
    m = re.search(r'case "\$VERB" in\s*\n\s*([a-z|-]+)\) shift ;;', text)
    listed = m.group(1).split("|") if m else []
    check("the verb case list is the CLI's VERBS without fetch and sbatch-args, in order",
          listed == [v for v in M.VERBS if v not in ("fetch", "sbatch-args")], listed)
    check("the verb class comes from the CLI's VERB_CLASS",
          "from weed_optimizer_framework.tools.funnel.__main__ import VERB_CLASS" in text)
    check("it imports the nested copy (PYTHONPATH=$CODE) and runs the CLI module",
          'CODE="$REPO/weed_llm_benchmark"' in text and 'export PYTHONPATH="$CODE' in text
          and 'python -u -m "$MOD" "$VERB"' in text and "MOD=%s" % MOD in text)
    m = re.search(r"GPU_LARGE_MIB=(\d+)", text)
    check("the 80 GB threshold is ceil(80e9 / 2**20) MiB",
          m and int(m.group(1)) == math.ceil(80e9 / 2 ** 20), m and m.group(1))
    plan = PLAN.read_text()
    exports = ("OLLAMA_MODELS", "OLLAMA_KEEP_ALIVE", "OLLAMA_NUM_PARALLEL", "OLLAMA_FLASH_ATTENTION",
               "OLLAMA_LOAD_TIMEOUT", "OLLAMA_HOST")

    def export_line(t, var):
        got = [ln.strip() for ln in t.splitlines() if ln.strip().startswith("export %s=" % var)]
        return got[0] if len(got) == 1 else None
    same = {v: (export_line(text, v), export_line(plan, v)) for v in exports}
    check("the ollama exports mirror run_inc_plan.sh (models store, keep-alive, one slot, flash attention, load "
          "timeout, host)", all(a is not None and a == b for a, b in same.values()), same)
    port = "PORT=$(( 8000 + ${SLURM_JOB_ID:-0} % 1000 ))"
    serve = "/ocean/projects/cis240145p/byler/ollama/bin/ollama serve"
    check("the per-job port formula, the serve binary and two warm-ups bounded at 1200 s mirror run_inc_plan.sh",
          port in text and port in plan and serve in text and serve in plan
          and "for attempt in 1 2; do" in text and "for attempt in 1 2; do" in plan
          and "curl -sf -m 1200 -X POST" in text and "curl -sf -m 1200 -X POST" in plan)
    check("the OLLAMA_MODELS store is the one the task pins",
          same["OLLAMA_MODELS"][0] == "export OLLAMA_MODELS=/ocean/projects/cis240145p/byler/ollama/models")


# ---------------------------------------------------------------- dry runs
def funnel_files():
    return sorted(p.relative_to(ROOT / "weed_optimizer_framework").as_posix()
                  for p in FUN.rglob("*") if p.is_file() and p.suffix in (".py", ".json")
                  and "__pycache__" not in p.parts)


def test_runs():
    print("dry runs")
    args = ["--prereg", str(PREREG), "--out", str(INC / "funnel")]
    rc, out, err = job(["census"] + args)
    logged = dict(re.findall(r"^module (\S+): ([0-9a-f]{64})$", out, flags=re.M))
    want = funnel_files() + list(FIXED_MODULES)
    wrong = [m for m in want if logged.get(m) != sha(ROOT / "weed_optimizer_framework" / m)]
    check("census dry run: exit 0, verb class cpu, the command with its arguments unchanged",
          rc == 0 and "verb class: cpu" in out and dry_line(out) == "python -u -m %s census %s" % (MOD, " ".join(args)),
          (rc, err[-500:], dry_line(out)))
    check("every file under the funnel package (%d) and the fixed INC and tools modules are logged with their "
          "sha256" % len(funnel_files()),
          not wrong and sorted(logged) == sorted(want), (wrong[:3], sorted(set(logged) ^ set(want))[:5]))
    check("the imports checked for a CPU verb: numpy, sklearn, joblib", "imports: numpy, sklearn, joblib" in out)
    rc, out, err = job(["recover", "--prereg", str(PREREG), "--audit", "a", "--maps", "m", "--policy", "R-A"])
    check("recover (class cpu) also checks torch and transformers (its H6 detector embeds with DINOv2) and PIL",
          rc == 0 and "verb class: cpu" in out and "imports: numpy, sklearn, joblib, torch, transformers, PIL" in out,
          (rc, err[-300:]))

    for argv, what in (([], "no verb"), (["bogus"] + args, "an unknown verb"), (["fetch"] + args, "fetch"),
                       (["sbatch-args", "leak"], "sbatch-args")):
        rc, out, err = job(argv)
        check("refuses %s (exit 2, usage)" % what, rc == 2 and "usage:" in err, (rc, err[-300:]))
    rc, out, err = job(["leak"] + args)
    check("a GPU verb with no GPU listed refuses (exit 2) and prints the sbatch line to use",
          rc == 2 and "sbatch $(python -m %s sbatch-args leak) run_inc_funnel.sh leak --prereg" % MOD in err,
          (rc, err[-400:]))
    if STUBBED:
        print("       (stubbed for the GPU import check, not installed here: %s)" % ", ".join(STUBBED))
    rc, out, err = job(["leak"] + args, gpus=V100)
    check("leak on a V100 passes the GPU check; imports torch, transformers, open_clip and PIL",
          rc == 0 and "verb class: gpu" in out and "imports: numpy, sklearn, joblib, torch, transformers, open_clip, "
          "PIL" in out and dry_line(out).startswith("python -u -m %s leak" % MOD), (rc, err[-400:]))
    rc, out, err = job(["rl-b"] + args, gpus=V100)
    check("rl-b on a 32 GB GPU refuses (needs 80 GB), printing the sbatch line with sbatch-args",
          rc == 2 and "at least 80 GB" in err and "sbatch-args rl-b" in err, (rc, err[-400:]))

    cfg = FUN / "domains" / ("%s.json" % json.loads(PREREG.read_text())["domain"])
    if cfg.is_file():
        tag = json.loads(cfg.read_text())["reference_labeller"]["backends"]["RL-B"]["model"]
        rc, out, err = job(["rl-b"] + args, gpus=H100, env={"SLURM_JOB_ID": "45123"})
        check("rl-b on an 80 GB GPU: the config's RL-B model %s is added as --model; the endpoint is the per-job "
              "port 8123; no server in a dry run" % tag,
              rc == 0 and "verb class: gpu_large" in out
              and "[cfg] model=%s endpoint=http://127.0.0.1:8123" % tag in out
              and dry_line(out) == "python -u -m %s rl-b %s --model %s" % (MOD, " ".join(args), tag)
              and "[ollama] serving" not in out, (rc, err[-400:], dry_line(out)))
        rc, out, err = job(["rl-b"] + args + ["--model", "other:tag"], gpus=H100)
        check("rl-b keeps an explicit --model (the CLI refuses any tag but the config's)",
              rc == 0 and dry_line(out).endswith("--model other:tag") and dry_line(out).count("--model") == 1)
    else:
        SKIPS.append("rl-b model tag (no domain config)")
        print("  SKIP %s is not there: the rl-b model tag cannot be resolved" % cfg)

    emb = ["embed-judges", "--stage", "embed", "--nshards", "4"] + args
    rc, out, err = job(emb, gpus=V100, env={"SLURM_ARRAY_TASK_ID": "3", "SLURM_JOB_ID": "7"})
    check("an array task of embed-judges --stage embed gets --shard <task id>",
          rc == 0 and dry_line(out) == "python -u -m %s %s --shard 3" % (MOD, " ".join(emb)), (rc, err[-300:]))
    emb2 = ["embed-judges", "--stage=embed", "--nshards", "4"] + args
    rc, out, err = job(emb2, gpus=V100, env={"SLURM_ARRAY_TASK_ID": "2"})
    check("... also with --stage=embed", rc == 0 and dry_line(out).endswith("--shard 2"), (rc, err[-300:]))
    rc, out, err = job(emb + ["--shard", "1"], gpus=V100, env={"SLURM_ARRAY_TASK_ID": "3"})
    check("an explicit --shard is kept", rc == 0 and dry_line(out).endswith("--shard 1")
          and "--shard 3" not in dry_line(out))
    for what, argv in (("census", ["census"] + args), ("embed-judges --stage judges",
                                                      ["embed-judges", "--stage", "judges"] + args)):
        rc, out, err = job(argv, gpus=V100, env={"SLURM_ARRAY_TASK_ID": "0"})
        check("an array job of %s refuses (exit 2)" % what, rc == 2 and "array job" in err)

    outer = REPO / "run_inc_funnel.sh"
    outer.write_text(SCRIPT.read_text() + "\n# edited\n")
    rc, out, err = job(["census"] + args)
    check("an outer $REPO/run_inc_funnel.sh that differs from the nested copy refuses (exit 2)",
          rc == 2 and "differs" in err, (rc, err[-300:]))
    shutil.copyfile(SCRIPT, outer)
    rc, out, err = job(["census"] + args)
    check("an identical outer copy runs", rc == 0, err[-300:])
    outer.unlink()
    edited = TMP / "edited_run_inc_funnel.sh"
    edited.write_text(SCRIPT.read_text() + "\n# edited\n")
    rc, out, err = job(["census"] + args, script=edited)
    check("a running script that differs from the nested copy refuses (exit 2)", rc == 2 and "differs" in err)
    (REPO / "docs" / "FUNNEL_AUDIT.md").rename(TMP / "contract.md")
    try:
        rc, out, err = job(["census"] + args)
        check("a missing contract refuses (exit 2)", rc == 2 and "FUNNEL_AUDIT.md is missing" in err, err[-300:])
    finally:
        (TMP / "contract.md").rename(REPO / "docs" / "FUNNEL_AUDIT.md")


def main():
    test_static()
    test_runs()


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("\n%d failure(s), %d skipped: %s" % (len(FAILURES), len(SKIPS), SKIPS))
    sys.exit(1 if FAILURES else 0)
