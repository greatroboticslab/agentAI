#!/usr/bin/env python3
"""run_inc_collect.sh, the collector's sbatch wrapper (docs/CONTINUOUS_LOOP.md
§3.1, §6.3 L16, §7.2; group D acceptance "Script").

Static checks read the script; the rest run it with bash in a temporary REPO
whose weed_llm_benchmark/ is a link to this nested copy, through the test
hooks the script offers (INC_COLLECT_REPO, INC_COLLECT_CONDA_SH,
INC_COLLECT_DRY_RUN=1: every check runs, then the command that would run is
printed), a fake conda.sh and `python` pointing at this interpreter.

Pinned:
  * bash -n passes; the #SBATCH lines (GPU-shared with one V100: RM-shared is
    refused by the allocation; the log under results/framework/inc/logs, the
    directory the autopilot's stream-submit creates before it submits: Slurm
    fails a job whose log directory is missing, and nothing creates
    intake/logs before the first job);
  * it accepts exactly plan, fetch, intake, summary and probe (the CLI's
    JOB_VERBS); names, any other verb and no verb refuse (exit 2);
  * it calls logging.basicConfig before the collector's main;
  * it has no git reset, no copy of the nested package over the outer one and
    no labelling-service sync or upload (no AUTO_SYNC, no roboflow);
  * a dry run of each verb exits 0, logs the sha256 of every collector file
    and of the modules it calls, and prints the verb with its arguments;
  * an outer $REPO/run_inc_collect.sh that differs from the nested copy
    refuses (an identical one runs); intake refuses when the nested copy lacks
    the copy guard module (fail closed);
  * intake, and only intake, runs with HF_HUB_OFFLINE=1 (compute nodes have no
    internet; D28-v2 describes its dHash hits with DINOv2 from the Hugging
    Face cache), while fetch keeps the network; the script no longer says the
    collector never uses the GPU (intake's DINOv2 runs on it).

No network, no GPU, no Slurm.

Run:  python3 tests/test_collect_job_script.py
"""
import hashlib
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_inc_collect.sh"
COL = ROOT / "weed_optimizer_framework" / "tools" / "collect"
TMP = pathlib.Path(tempfile.mkdtemp(prefix="collect_job_"))
os.environ["INC_DIR"] = str(TMP / "inc_unused")
os.environ["REPO"] = str(TMP / "repo_unused")
sys.path.insert(0, str(ROOT))
from weed_optimizer_framework.tools.collect import __main__ as M  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, str(detail)[:1500]))
        FAILURES.append(name)


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


REPO = TMP / "repo"
REPO.mkdir()
os.symlink(ROOT, REPO / "weed_llm_benchmark")
BIN = TMP / "bin"
BIN.mkdir()
os.symlink(sys.executable, BIN / "python")
CONDA = TMP / "conda.sh"
CONDA.write_text("conda() { return 0; }\n")
RUN = REPO / "weed_llm_benchmark" / "run_inc_collect.sh"


def job(args, env=None, script=RUN, repo=REPO):
    e = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    e.update({"PATH": "%s:%s" % (BIN, os.environ.get("PATH", "")), "INC_COLLECT_REPO": str(repo),
              "INC_COLLECT_CONDA_SH": str(CONDA), "INC_COLLECT_DRY_RUN": "1"})
    e.pop("INC_DIR", None)
    e.pop("REPO", None)
    e.pop("HF_HUB_OFFLINE", None)
    e.update(env or {})
    r = subprocess.run(["bash", str(script)] + list(args), capture_output=True, text=True, env=e, timeout=300)
    return r.returncode, r.stdout, r.stderr


def test_static():
    print("static")
    r = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    check("bash -n passes", r.returncode == 0, r.stderr)
    text = SCRIPT.read_text()
    sb = [ln for ln in text.splitlines() if ln.startswith("#SBATCH")]
    check("GPU-shared with one V100 (RM-shared is refused), the log under inc/logs",
          "#SBATCH --partition=GPU-shared" in sb and "#SBATCH --gres=gpu:v100-32:1" in sb
          and any(ln.startswith("#SBATCH --output=") and ln.endswith("results/framework/inc/logs/%x_%j.out")
                  for ln in sb), sb)
    import weed_optimizer_framework.tools.inc_autopilot.stream_remote as SR
    src = pathlib.Path(SR.__file__).read_text()
    code_lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#") or ln.startswith("#SBATCH")]
    check("... which is the directory the autopilot creates before every stream sbatch (and no line of code "
          "names intake/logs)", '(R.inc_dir() / "logs").mkdir(parents=True, exist_ok=True)' in src
          and not any("intake/logs" in ln for ln in code_lines), [ln for ln in code_lines if "intake/logs" in ln])
    m = re.search(r'case "\$VERB" in\s*\n\s*([a-z|-]+)\) shift ;;', text)
    listed = m.group(1).split("|") if m else []
    check("the verb list is exactly plan, fetch, intake, summary, probe (the CLI's JOB_VERBS)",
          listed == list(M.JOB_VERBS) == ["plan", "fetch", "intake", "summary", "probe"], listed)
    check("it calls logging.basicConfig before the collector's main", "logging.basicConfig(" in text
          and text.index("logging.basicConfig(") < text.index("from weed_optimizer_framework.tools.collect.__main__ "
                                                               "import main"))
    code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    check("no git reset, no rsync, no copying of the nested package over the outer one",
          "git reset" not in code and "rsync" not in code and not re.search(r"\bcp\b", code), None)
    check("no labelling-service sync or upload", "AUTO_SYNC" not in text and "roboflow" not in text.lower()
          and "upload" not in code.lower(), None)
    check("the script no longer claims the collector never uses the GPU (intake's DINOv2 runs on it, D28-v2)",
          "never uses the GPU" not in text and "export HF_HUB_OFFLINE=1" in code, None)


def test_runs():
    print("dry runs")
    files = sorted(p.relative_to(ROOT / "weed_optimizer_framework").as_posix() for p in COL.rglob("*")
                   if p.is_file() and p.suffix in (".py", ".json") and "__pycache__" not in p.parts)
    for verb, args in (("plan", ["--out", "/tmp/c.json"]), ("fetch", ["--source", "x", "--max-bytes", "5"]),
                       ("intake", ["--source", "x"]), ("summary", []), ("probe", [])):
        rc, out, err = job([verb] + args)
        logged = dict(re.findall(r"^module (\S+): ([0-9a-f]{64})$", out, flags=re.M))
        wrong = [f for f in files if logged.get(f) != sha(ROOT / "weed_optimizer_framework" / f)]
        dry = [ln for ln in out.splitlines() if ln.startswith("DRY RUN: ")]
        check("%s: exit 0, every collector file logged with its sha256, the verb and its arguments" % verb,
              rc == 0 and not wrong and dry and dry[-1].endswith(("%s %s" % (verb, " ".join(args))).rstrip()),
              (rc, err[-400:], wrong[:3], dry))
    offline = {}
    for verb, args in (("intake", ["--source", "x"]), ("fetch", ["--source", "x", "--max-bytes", "5"]),
                       ("probe", [])):
        rc, out, err = job([verb] + args)
        offline[verb] = re.findall(r"^HF_HUB_OFFLINE: (\S+)$", out, flags=re.M)
    check("intake runs with HF_HUB_OFFLINE=1 (DINOv2 from the Hugging Face cache, no internet on compute nodes); "
          "fetch and probe keep the network", offline == {"intake": ["1"], "fetch": ["unset"], "probe": ["unset"]},
          offline)
    rc, out, err = job(["summary"])
    for m in ("tools/mega_trainer.py", "tools/inc/common.py", "tools/funnel/taxonomy.py", "tools/license_audit.py"):
        check("the called module %s is logged" % m, ("module %s: %s" % (m, sha(ROOT / "weed_optimizer_framework" / m)))
              in out)
    for args in ([], ["names", "--source", "x"], ["train"], ["--help"]):
        rc, out, err = job(args)
        check("%r refuses (exit 2)" % (args[:1] or ["no verb"]), rc == 2 and "usage" in err, (rc, err[-300:]))


def test_copies():
    print("the nested copy")
    outer = REPO / "run_inc_collect.sh"
    shutil.copyfile(SCRIPT, outer)
    rc, out, err = job(["summary"])
    check("an identical outer copy runs", rc == 0, err[-300:])
    outer.write_text(SCRIPT.read_text() + "\n# drift\n")
    rc, out, err = job(["summary"])
    check("an outer copy that differs from the nested one refuses", rc == 2 and "differs" in err, (rc, err[-300:]))
    outer.unlink()
    repo2 = TMP / "repo2"
    code = repo2 / "weed_llm_benchmark"
    shutil.copytree(ROOT / "weed_optimizer_framework", code / "weed_optimizer_framework",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pt", "*.npz"), symlinks=True)
    shutil.copyfile(SCRIPT, code / "run_inc_collect.sh")
    (code / "weed_optimizer_framework" / "tools" / "inc2" / "guard.py").unlink(missing_ok=True)
    rc, out, err = job(["intake", "--source", "x"], script=code / "run_inc_collect.sh", repo=repo2)
    check("intake refuses when the nested copy lacks the copy guard module (fail closed)",
          rc == 2 and "inc2/guard.py" in err, (rc, err[-300:]))
    rc, out, err = job(["summary"], script=code / "run_inc_collect.sh", repo=repo2)
    check("... while a verb that needs no guard still runs", rc == 0, err[-300:])


def main():
    try:
        test_static()
        test_runs()
        test_copies()
    finally:
        shutil.rmtree(str(TMP), ignore_errors=True)
    print("\n%d failure(s)" % len(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
