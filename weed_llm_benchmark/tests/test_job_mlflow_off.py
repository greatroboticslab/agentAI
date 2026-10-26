#!/usr/bin/env python3
"""The training jobs run with Ultralytics' MLflow logging off, and nothing else of Ultralytics' settings changes
(run_inc_job.sh and run_inc2_job.sh, the block between '# >>> mlflow-off' and '# <<< mlflow-off').

Ultralytics' MLflow callback copies every run's weights directory into runs/mlflow (about 0.08 GB a run; 337 GB by
2026-10-26). The block gives the job its own copy of the shared settings with only "mlflow" set false and exports
YOLO_CONFIG_DIR only once Ultralytics, imported under it, reads every other key (datasets_dir first) equal to the
shared file's, rewrites nothing and registers no MLflow hook; otherwise the job runs as before with a WARNING.

Checked here:
  * the block is the same text in both scripts, after `conda activate bench` and before the trainer starts;
  * without Ultralytics (or with a shared settings file that carries no boolean mlflow) the block warns, exports
    nothing and the job goes on;
  * with Ultralytics importable: on a shared settings file of this Ultralytics' own schema with mlflow true and a
    datasets_dir of its own, the job's copy differs only in mlflow, YOLO_CONFIG_DIR is exported, and a process under
    it reads datasets_dir, runs_dir and weights_dir as the shared file has them, mlflow false, no MLflow hook, the
    font found through the job's link, the shared file unchanged (sha256 and mtime); the control under the shared
    settings registers the MLflow hooks when mlflow is installed;
  * a chained array (the in-job advance submits the next array from inside the job, and the driver's sbatch
    environment keeps every variable): the advance hands the next array the caller's YOLO_CONFIG_DIR (or none),
    never the job's copy; and a job that inherits another job's copy (ultralytics_cfg_*, gone with that job's
    node-local disk) drops it and reads the shared settings, so Ultralytics never falls back to fresh defaults
    (another datasets_dir, MLflow on). Both scripts' advance and block are run;
  * MLFLOW_OFF_REAL=1 (the cluster's bench env): the same on the account's real shared settings, read only.

Run:  python3 tests/test_job_mlflow_off.py        (MLFLOW_OFF_REAL=1 also checks the real shared settings)
"""
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = ("run_inc_job.sh", "run_inc2_job.sh")
BEGIN, END = "# >>> mlflow-off", "# <<< mlflow-off"
FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, str(detail)[:1500]))
        FAILURES.append(name)


def block(script):
    text = (ROOT / script).read_text()
    i, j = text.find(BEGIN), text.find(END)
    return text[i:text.find("\n", j) + 1] if i >= 0 and j > i else None


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def has_ultralytics(py):
    return subprocess.run([py, "-c", "import ultralytics"], capture_output=True).returncode == 0


def has_mlflow(py):
    return subprocess.run([py, "-c", "import mlflow"], capture_output=True).returncode == 0


def advance_fn(script):
    """The advance() function of a job script (the in-job advance of the experiment's next array)."""
    text = (ROOT / script).read_text()
    i = text.find("advance() {")
    return text[i:text.find("\n}\n", i) + 3] if i >= 0 else None


def run_block(tmp, env_extra, py, script=SCRIPTS[0], advance=False, job="4242"):
    """The block in bash with `python` on PATH, then the environment it leaves (YOLO_CONFIG_DIR) printed; with
    `advance`, the script's advance() run after it, the driver replaced by a probe that prints the YOLO_CONFIG_DIR
    the next array's sbatch would carry (DRIVER=...)."""
    bindir = tmp / "bin"
    bindir.mkdir(exist_ok=True)
    if not (bindir / "python").exists():
        # a wrapper, never a symlink: a venv's python reached through a symlink in another directory loses its venv
        (bindir / "python").write_text('#!/bin/bash\ncase "$*" in *"tools.inc.driver advance"*)\n'
                                       '    echo "DRIVER=${YOLO_CONFIG_DIR-<unset>}"; exit 0 ;;\nesac\n'
                                       'exec "%s" "$@"\n' % py)
        (bindir / "python").chmod(0o755)
    sh = tmp / "block.sh"
    sh.write_text("set -uo pipefail\n" + block(script) + 'echo "EXPORTED=${YOLO_CONFIG_DIR:-}"\n'
                  + (advance_fn(script) + 'EXP=exp_x\nadvance\n' if advance else ""))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("YOLO_", "SLURM_", "LOCAL"))}
    env.update(PATH="%s:%s" % (bindir, env.get("PATH", "")), TMPDIR=str(tmp / "local"), SLURM_JOB_ID=job,
               SLURM_ARRAY_TASK_ID="3", YOLO_OFFLINE="true")
    (tmp / "local").mkdir(exist_ok=True)
    env.update(env_extra)
    p = subprocess.run(["bash", str(sh)], capture_output=True, text=True, env=env, timeout=600)
    exported = [x.split("=", 1)[1] for x in p.stdout.splitlines() if x.startswith("EXPORTED=")]
    if advance:
        drv = [x.split("=", 1)[1] for x in p.stdout.splitlines() if x.startswith("DRIVER=")]
        return p, (exported[-1] if exported else None), env, (drv[-1] if drv else None)
    return p, (exported[-1] if exported else None), env


def probe(env, py, font=True):
    """What a process under env reads: Ultralytics' settings, its config dir, its MLflow hooks, the font path (font:
    check_font, which downloads a missing font into the config dir; never asked of a shared config dir)."""
    code = ("import json; from ultralytics.utils import SETTINGS, USER_CONFIG_DIR; "
            "from ultralytics.utils.callbacks import mlflow as m; "
            "from ultralytics.utils.checks import check_font; "
            "f = check_font('Arial.ttf') if %s else None; "
            "print(json.dumps({'settings': dict(SETTINGS), 'dir': str(USER_CONFIG_DIR), "
            "'hooks': sorted(m.callbacks), 'font': str(f)}))" % bool(font))
    p = subprocess.run([py, "-c", code], capture_output=True, text=True, env=env, timeout=600)
    lines = [x for x in p.stdout.splitlines() if x.startswith("{")]
    return json.loads(lines[-1]) if lines else {"error": p.stderr[-800:]}


def test_text():
    print("the block in both job scripts")
    a, b = (block(s) for s in SCRIPTS)
    check("both scripts hold the block, the same text", a and b and a == b, (bool(a), bool(b)))
    for s in SCRIPTS:
        t = (ROOT / s).read_text()
        act = t.find("conda activate bench")
        train = max(t.find("python -u -m weed_optimizer_framework.tools.inc.train --spec"),
                    t.find("python -u -m weed_optimizer_framework.tools.inc2.train --spec"))
        check("%s: after conda activate bench, before the trainer" % s, 0 <= act < t.find(BEGIN) < train,
              (act, t.find(BEGIN), train))
        check("%s: the block never writes the shared settings or runs/" % s,
              "> \"$UL_SHARED" not in (a or "") and "runs/mlflow" not in (a or "").split("never touched")[-1])


def shared_dir(tmp, py, mlflow=True, datasets="/data/elsewhere"):
    """A shared settings directory of this Ultralytics' own schema (written by Ultralytics itself), with mlflow
    and datasets_dir set, and a font file."""
    root = tmp / "shared_root"
    (root / "Ultralytics").mkdir(parents=True)   # an existing directory is used as it is (never a fallback)
    env = {k: v for k, v in os.environ.items() if not k.startswith("YOLO_")}
    env.update(YOLO_CONFIG_DIR=str(root), YOLO_OFFLINE="true")
    subprocess.run([py, "-c", "from ultralytics.utils import SETTINGS"], env=env, capture_output=True, timeout=600)
    f = root / "Ultralytics" / "settings.json"
    s = json.loads(f.read_text())
    s.update(mlflow=mlflow, datasets_dir=datasets)
    f.write_text(json.dumps(s, indent=2))
    (root / "Ultralytics" / "Arial.ttf").write_bytes(b"not really a font, only its path is read")
    return root


def test_no_ultralytics():
    print("without Ultralytics, or without a boolean mlflow: a warning, nothing exported, the job goes on")
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="mlflow_off_"))
    try:
        root = tmp / "shared_root" / "Ultralytics"
        root.mkdir(parents=True)
        (root / "settings.json").write_text(json.dumps({"mlflow": True, "datasets_dir": "/data/x"}))
        fake = tmp / "nopy"
        fake.mkdir()
        (fake / "ultralytics").mkdir()
        (fake / "ultralytics" / "__init__.py").write_text("raise ImportError('no ultralytics here')\n")
        p, exp, _ = run_block(tmp, {"YOLO_CONFIG_DIR": str(tmp / "shared_root"), "PYTHONPATH": str(fake)},
                              sys.executable)
        check("Ultralytics not importable: exit 0, a WARNING, YOLO_CONFIG_DIR stays the caller's",
              p.returncode == 0 and "WARNING: MLflow logging stays on" in p.stderr and exp == str(tmp / "shared_root"),
              (p.returncode, exp, p.stderr[-400:]))
        (root / "settings.json").write_text(json.dumps({"datasets_dir": "/data/x"}))
        tmp2 = tmp / "second"
        tmp2.mkdir()
        p, exp, _ = run_block(tmp2, {"YOLO_CONFIG_DIR": str(tmp / "shared_root")}, sys.executable)
        check("shared settings with no boolean mlflow: a WARNING, no copy used",
              p.returncode == 0 and "WARNING: MLflow logging stays on" in p.stderr and exp == str(tmp / "shared_root"),
              (p.returncode, exp, p.stderr[-400:]))
        check("the shared settings were not written", json.loads((root / "settings.json").read_text())
              == {"datasets_dir": "/data/x"})
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)


def test_with_ultralytics(py):
    print("with Ultralytics (%s): MLflow off, every other setting the shared file's" % py)
    if not has_ultralytics(py):
        print("  skip: Ultralytics is not importable by %s" % py)
        return
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="mlflow_off_"))
    try:
        root = shared_dir(tmp, py)
        f = root / "Ultralytics" / "settings.json"
        before = (sha(f), f.stat().st_mtime_ns)
        p, exp, env = run_block(tmp, {"YOLO_CONFIG_DIR": str(root)}, py)
        job = tmp / "local" / "ultralytics_cfg_4242_3"
        check("the block exports YOLO_CONFIG_DIR = the job's directory and says so",
              p.returncode == 0 and exp == str(job) and "MLflow logging off" in p.stdout,
              (p.returncode, exp, p.stdout[-600:], p.stderr[-600:]))
        mine = json.loads((job / "Ultralytics" / "settings.json").read_text()) if (job / "Ultralytics").is_dir() else {}
        shared = json.loads(f.read_text())
        check("the job's copy differs from the shared settings in mlflow only",
              mine.get("mlflow") is False and {k: v for k, v in mine.items() if k != "mlflow"}
              == {k: v for k, v in shared.items() if k != "mlflow"}, (mine, shared))
        env2 = dict(env, YOLO_CONFIG_DIR=str(job))
        got = probe(env2, py)
        st = got.get("settings") or {}
        check("under it Ultralytics reads datasets_dir, runs_dir and weights_dir as the shared file has them",
              all(st.get(k) == shared.get(k) for k in ("datasets_dir", "runs_dir", "weights_dir"))
              and st.get("datasets_dir") == "/data/elsewhere", got)
        check("under it mlflow is false and no MLflow hook is registered", st.get("mlflow") is False
              and got.get("hooks") == [], got)
        check("the font is found through the job's link (no download)",
              pathlib.Path(str(got.get("font"))).parent == job / "Ultralytics"
              and os.path.realpath(str(got.get("font"))) == str((root / "Ultralytics" / "Arial.ttf").resolve()), got)
        check("the shared settings file is unchanged (sha256 and mtime)", (sha(f), f.stat().st_mtime_ns) == before)
        # a shared settings file this Ultralytics would reset (an unknown key): nothing exported, MLflow left on
        bad = tmp / "bad"
        bad.mkdir()
        broot = shared_dir(bad, py)
        bf = broot / "Ultralytics" / "settings.json"
        bs = json.loads(bf.read_text())
        bs["not_a_setting"] = 1
        bf.write_text(json.dumps(bs, indent=2))
        p2, exp2, _ = run_block(bad, {"YOLO_CONFIG_DIR": str(broot)}, py)
        check("a shared settings file Ultralytics would reset: a WARNING, YOLO_CONFIG_DIR stays the caller's",
              p2.returncode == 0 and exp2 == str(broot) and "WARNING: MLflow logging stays on" in p2.stderr,
              (exp2, p2.stdout[-300:], p2.stderr[-500:]))
        if has_mlflow(py):
            ctl = probe(dict(env, YOLO_CONFIG_DIR=str(root)), py, font=False)
            check("control: under the shared settings (mlflow true) the MLflow hooks are registered",
                  "on_train_end" in (ctl.get("hooks") or []), ctl)
        else:
            print("  skip: mlflow is not installed for %s (the control needs it)" % py)
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)


def test_chain(py):
    print("a chained array: the advance hands on the caller's settings; an inherited job copy is dropped")
    for script in SCRIPTS:
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="mlflow_off_chain_"))
        try:
            # no Ultralytics needed: the advance's environment and the inherited copy's fate
            fake = tmp / "nopy"
            (fake / "ultralytics").mkdir(parents=True)
            (fake / "ultralytics" / "__init__.py").write_text("raise ImportError('no ultralytics here')\n")
            xdg = tmp / "xdg"
            (xdg / "Ultralytics").mkdir(parents=True)
            (xdg / "Ultralytics" / "settings.json").write_text(json.dumps({"mlflow": True, "datasets_dir": "/d"}))
            dead = tmp / "parent_local" / "ultralytics_cfg_111_0"       # a parent job's copy, its disk gone
            for caller, want in ((None, "<unset>"), (str(tmp / "callers_root"), str(tmp / "callers_root"))):
                extra = {"PYTHONPATH": str(fake), "XDG_CONFIG_HOME": str(xdg)}
                if caller:
                    extra["YOLO_CONFIG_DIR"] = caller
                p, exp, _env, drv = run_block(tmp, extra, sys.executable, script=script, advance=True)
                check("%s: the advance hands the next array the caller's YOLO_CONFIG_DIR (%s)" % (script, want),
                      p.returncode == 0 and drv == want, (drv, p.stdout[-400:], p.stderr[-400:]))
            p, exp, _env = run_block(tmp, {"PYTHONPATH": str(fake), "XDG_CONFIG_HOME": str(xdg),
                                           "YOLO_CONFIG_DIR": str(dead)}, sys.executable, script=script)
            cp = tmp / "local" / "ultralytics_cfg_4242_3" / "Ultralytics" / "settings.json"
            cdoc = json.loads(cp.read_text()) if cp.is_file() else {}
            check("%s: an inherited job copy (gone) is dropped: the job's copy is taken from the shared settings "
                  "(datasets_dir /d), nothing exported without Ultralytics, the gone path never named" % script,
                  p.returncode == 0 and exp == "" and cdoc == {"mlflow": False, "datasets_dir": "/d"}
                  and str(dead) not in p.stderr + p.stdout, (exp, cdoc, p.stderr[-500:]))
            if not has_ultralytics(py):
                print("  skip: Ultralytics is not importable by %s (the chained run end to end)" % py)
                continue
            # end to end: the parent job exports its copy and advances; the child inherits the parent's environment
            # after the parent's node-local disk is gone
            root = shared_dir(tmp, py, datasets="/data/shared_ds")
            f = root / "Ultralytics" / "settings.json"
            before = (sha(f), f.stat().st_mtime_ns)
            ptmp = tmp / "parent"
            ptmp.mkdir()
            p, exp, penv, drv = run_block(ptmp, {"YOLO_CONFIG_DIR": str(root)}, py, script=script, advance=True,
                                          job="111")
            pjob = ptmp / "local" / "ultralytics_cfg_111_3"
            check("%s: the parent exports its own copy and hands the shared root to the next array" % script,
                  exp == str(pjob) and drv == str(root), (exp, drv, p.stderr[-400:]))
            shutil.rmtree(str(ptmp / "local"), ignore_errors=True)
            ctmp = tmp / "child"
            ctmp.mkdir()
            # the old advance's environment (before this change): the parent's copy inherited
            c, cexp, cenv = run_block(ctmp, {"YOLO_CONFIG_DIR": str(pjob), "XDG_CONFIG_HOME": str(root)}, py,
                                      script=script, job="222")
            cjob = ctmp / "local" / "ultralytics_cfg_222_3"
            got = probe(dict(cenv, YOLO_CONFIG_DIR=cexp) if cexp else {k: v for k, v in cenv.items()
                                                                      if k != "YOLO_CONFIG_DIR"}, py, font=False)
            st = got.get("settings") or {}
            check("%s: a child that inherited the parent's gone copy reads the shared settings through its own copy: "
                  "datasets_dir %s, mlflow false (never Ultralytics' fresh defaults)" % (script, st.get("datasets_dir")),
                  c.returncode == 0 and cexp == str(cjob) and st.get("datasets_dir") == "/data/shared_ds"
                  and st.get("mlflow") is False and got.get("hooks") == [], (cexp, got, c.stderr[-400:]))
            check("%s: the shared settings file is unchanged (sha256 and mtime)" % script,
                  (sha(f), f.stat().st_mtime_ns) == before)
        finally:
            shutil.rmtree(str(tmp), ignore_errors=True)


def test_real(py):
    print("the account's real shared settings (MLFLOW_OFF_REAL=1), read only")
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="mlflow_off_real_"))
    try:
        env0 = {k: v for k, v in os.environ.items() if not k.startswith("YOLO_")}
        env0["YOLO_OFFLINE"] = "true"
        base = probe(env0, py, font=False)
        shared = pathlib.Path(base.get("dir") or "") / "settings.json"
        files = [shared] + [x for x in [shared.parent / "persistent_cache.json"] if x.is_file()]
        before = {str(x): (sha(x), x.stat().st_mtime_ns) for x in files}
        p, exp, env = run_block(tmp, {}, py)
        job = tmp / "local" / "ultralytics_cfg_4242_3"
        check("the block exports the job's directory", p.returncode == 0 and exp == str(job),
              (p.stdout[-600:], p.stderr[-600:]))
        got = probe(dict(env, YOLO_CONFIG_DIR=str(job)), py)
        st, sh = got.get("settings") or {}, base.get("settings") or {}
        check("datasets_dir, runs_dir, weights_dir and every other key as the shared settings (mlflow aside): %s"
              % st.get("datasets_dir"), st and {k: v for k, v in st.items() if k != "mlflow"}
              == {k: v for k, v in sh.items() if k != "mlflow"}, (st, sh))
        check("mlflow false, no MLflow hook (the shared settings register %s)" % (base.get("hooks") or "none"),
              st.get("mlflow") is False and got.get("hooks") == [], got)
        check("the shared settings files are unchanged (sha256 and mtime)",
              {str(x): (sha(x), x.stat().st_mtime_ns) for x in files} == before)
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)


def main():
    py = os.environ.get("MLFLOW_OFF_PYTHON") or sys.executable
    test_text()
    test_no_ultralytics()
    test_with_ultralytics(py)
    test_chain(py)
    if os.environ.get("MLFLOW_OFF_REAL") == "1":
        test_real(py)
    print("\n%d failure(s)" % len(FAILURES))
    for x in FAILURES:
        print("  FAILED: %s" % x)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
