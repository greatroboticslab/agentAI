"""Code-writing analyst — the agent writes analysis code on demand (v3.7).

The prof's ask, verbatim: arbitrary analysis methods + plots that follow the
question, with the agent's code visible in the browser. Architecture follows the
Code-Interpreter / Microsoft-LIDA pattern: LLM writes a small script → static
safety check → sandboxed execution → on error, the traceback is fed back for
self-repair (≤2 rounds) → return the plot + printed results + THE CODE ITSELF.

Honesty: results are labeled as computed by the generated code (shown to the
user); if every attempt fails we say so — we never fake an answer.

Sandbox (no docker/nsjail on the lab box; threat model = accidental bad code
from our own codegen, not an adversarial attacker):
  layer 1: AST whitelist — only numpy/pandas/scipy/matplotlib/math/json/etc.,
           no socket/subprocess/os/sys/eval/exec/open-for-write/dunder tricks.
  layer 2: subprocess with POSIX rlimits (CPU, memory, file size), cleaned env,
           throwaway working dir, hard wall-clock timeout, headless matplotlib.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# layer 1 — static safety check (AST)
# ---------------------------------------------------------------------------
_ALLOWED_IMPORTS = {
    "numpy", "np", "pandas", "pd", "math", "json", "statistics", "collections",
    "itertools", "functools", "datetime", "re", "io",
    "matplotlib", "matplotlib.pyplot", "scipy", "scipy.signal", "scipy.stats",
    "scipy.fft", "scipy.cluster", "scipy.optimize", "sklearn", "sklearn.cluster",
    "sklearn.linear_model", "sklearn.decomposition", "sklearn.preprocessing",
    "seaborn", "sns",
    # v3.9: image datasets are staged into the sandbox too — reading pixels
    # needs PIL (Pillow). Still no network / no file writes.
    "PIL", "PIL.Image", "PIL.ImageOps", "PIL.ImageStat", "glob", "pathlib",
}
_BANNED_NAMES = {"eval", "exec", "compile", "__import__", "globals", "locals",
                 "vars", "getattr", "setattr", "delattr", "input", "breakpoint",
                 "memoryview", "exit", "quit"}
_BANNED_ATTRS = {"system", "popen", "fork", "kill", "remove", "unlink", "rmdir",
                 "rmtree", "chmod", "chown", "environ", "putenv"}


def check_code(code: str) -> str | None:
    """Return None if the code passes the whitelist, else a human-readable reason."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"syntax error: {e}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = ([a.name for a in node.names] if isinstance(node, ast.Import)
                    else [node.module or ""])
            for m in mods:
                root = m.split(".")[0]
                if m not in _ALLOWED_IMPORTS and root not in _ALLOWED_IMPORTS:
                    return f"import of '{m}' is not allowed"
        elif isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            return f"use of '{node.id}' is not allowed"
        elif isinstance(node, ast.Attribute) and node.attr in _BANNED_ATTRS:
            return f"attribute '{node.attr}' is not allowed"
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id == "open":
                # writing files is banned; reading the provided csvs is fine
                for a in list(node.args[1:2]) + [k.value for k in node.keywords
                                                 if k.arg == "mode"]:
                    if isinstance(a, ast.Constant) and any(c in str(a.value) for c in "wax+"):
                        return "opening files for writing is not allowed (only savefig to out.png)"
        # dunder access like x.__class__ etc.
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"dunder attribute '{node.attr}' is not allowed"
        # v3.9: with pathlib/glob whitelisted (image staging) keep reads INSIDE
        # the staged working dir: no absolute paths, no '..', no '~' in string
        # literals. (Heuristic guard on top of the existing layers, not a VM.)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            s = node.value
            if s.startswith(("/", "~")) or "/../" in s or s.startswith("../") or s == "..":
                return (f"path '{s[:40]}' is not allowed — read only the staged "
                        f"dataset copies in the working directory")
    return None


# ---------------------------------------------------------------------------
# layer 2 — sandboxed execution
# ---------------------------------------------------------------------------
_PRELUDE = (
    "import matplotlib\n"
    "matplotlib.use('Agg')\n"
)

# v3.9: users kept running code that plotted but never called savefig — and saw
# nothing. This trusted footer (OUR code, appended AFTER the AST check of the
# user code) captures the FINAL open figure state — always, not only when
# out.png is missing: a user who edits code and draws a NEW figure after the
# original savefig must see the new figure, not the stale file (v3.9.1 fix).
_FOOTER = (
    "\n\ntry:\n"
    "    import matplotlib.pyplot as _plt_cap\n"
    "    if _plt_cap.get_fignums():\n"
    "        _plt_cap.gcf().savefig('out.png', dpi=110, bbox_inches='tight')\n"
    "except Exception:\n"
    "    pass\n"
)


def run_sandboxed(code: str, workdir: Path, timeout_s: int = 25) -> dict:
    """Run `code` in a subprocess inside `workdir` with rlimits. Returns
    {ok, stdout, stderr, plot: bool}. The script may read the staged dataset
    copies; figures are auto-captured to ./out.png (explicit savefig also fine);
    findings go to stdout."""
    script = workdir / "_script.py"
    script.write_text(_PRELUDE + code + _FOOTER, encoding="utf-8")

    def _limits():
        import resource
        # apply best-effort per-limit (some limits are unsupported/stricter on macOS;
        # the wall-clock timeout below is the universal backstop)
        for lim, val in ((resource.RLIMIT_CPU, 15),
                         (resource.RLIMIT_AS, 1_500_000_000),
                         (resource.RLIMIT_FSIZE, 25_000_000),
                         (resource.RLIMIT_NOFILE, 256)):
            try:
                resource.setrlimit(lim, (val, val))
            except (ValueError, OSError):
                pass

    env = {"PATH": "/usr/bin:/bin", "HOME": str(workdir), "MPLCONFIGDIR": str(workdir),
           "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "2",
           "OMP_NUM_THREADS": "2"}
    try:
        p = subprocess.run([sys.executable, str(script)], cwd=str(workdir), env=env,
                           capture_output=True, text=True, timeout=timeout_s,
                           preexec_fn=_limits)
        out = (p.stdout or "")[-4000:]
        err = (p.stderr or "")[-4000:]
        return {"ok": p.returncode == 0, "stdout": out, "stderr": err,
                "plot": (workdir / "out.png").is_file()}
    except subprocess.TimeoutExpired:
        return {"ok": False, "stdout": "", "stderr": f"timed out after {timeout_s}s", "plot": False}
    except Exception as e:
        return {"ok": False, "stdout": "", "stderr": f"{type(e).__name__}: {e}", "plot": False}


# ---------------------------------------------------------------------------
# codegen + self-repair loop (LIDA-style)
# ---------------------------------------------------------------------------
_CODE_SYS = (
    "You write ONE short self-contained Python script for data analysis. Rules:\n"
    "- The working directory contains the dataset EXACTLY as listed in the prompt: "
    "CSV file(s) in ./ (read with pandas), and/or a sample of images in ./images/ "
    "(open with PIL.Image; convert to numpy with np.asarray) with optional YOLO "
    "label files in ./labels/ (same stem, .txt: 'class cx cy w h' normalized 0-1).\n"
    "- Allowed libraries ONLY: pandas, numpy, scipy, sklearn, matplotlib, seaborn, PIL, "
    "math, json, statistics, collections, datetime, re, io, glob, pathlib.\n"
    "- Use ONLY relative paths inside the working directory (never absolute, never ..).\n"
    "- If a figure helps, save it with plt.savefig('out.png', dpi=110, bbox_inches='tight'). "
    "Never plt.show().\n"
    "- print() the key numeric findings as short labeled lines (these are shown to the user).\n"
    "- No network, no file writes except out.png, no subprocess/os/sys.\n"
    "- Keep it under ~60 lines. Output ONLY the code, no markdown fences, no explanations."
)


def _strip_fences(txt: str) -> str:
    t = (txt or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if "```" in t:
            t = t[:t.rindex("```")]
    return t.strip()


def data_brief(files: list, max_cols: int = 24) -> str:
    """Compact description of the CSVs for the codegen prompt:
    name, rows, columns with dtype-ish info and ranges."""
    lines = []
    for f in files:
        cols = []
        for c, vals in list(f["cols"].items())[:max_cols]:
            if vals:
                cols.append(f"{c} (numeric, {min(vals):.4g}..{max(vals):.4g})")
        n = max((len(v) for v in f["cols"].values()), default=0)
        lines.append(f"- {f['name']}: {n} rows; columns: {', '.join(cols)}")
    return "\n".join(lines)


def write_and_run(question: str, files: list, workdir: Path, llm_code,
                  max_repairs: int = 2, extra_brief: str = "") -> dict:
    """Full loop: prompt the coder LLM, safety-check, execute, self-repair on error.
    `llm_code(prompt, system) -> str`. `extra_brief` describes non-tabular staged
    data (e.g. the image sample). Returns
    {ok, code, stdout, stderr, plot, attempts, refusal_reason?}."""
    brief = (data_brief(files) + ("\n" + extra_brief if extra_brief else "")).strip()
    prompt = (f"DATASET FILES in the working directory:\n{brief}\n\n"
              f"USER REQUEST: {question}\n\nWrite the script now.")
    code, last = "", {}
    for attempt in range(1 + max_repairs):
        raw = llm_code(prompt, _CODE_SYS) or ""
        code = _strip_fences(raw)
        if not code:
            last = {"ok": False, "stderr": "model returned no code", "stdout": "", "plot": False}
            continue
        why = check_code(code)
        if why:
            prompt += (f"\n\nYour previous script was rejected by the safety checker: {why}. "
                       f"Rewrite the whole script avoiding that.")
            last = {"ok": False, "stderr": f"safety: {why}", "stdout": "", "plot": False}
            continue
        last = run_sandboxed(code, workdir)
        if last["ok"]:
            return {"ok": True, "code": code, "stdout": last["stdout"],
                    "stderr": "", "plot": last["plot"], "attempts": attempt + 1}
        prompt += (f"\n\nYour previous script failed with this error:\n{last['stderr'][-1200:]}\n"
                   f"Fix the problem and output the FULL corrected script.")
    return {"ok": False, "code": code, "stdout": last.get("stdout", ""),
            "stderr": last.get("stderr", ""), "plot": False, "attempts": 1 + max_repairs}


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def stage_images(src_root: Path, workdir: Path, cap: int = 40,
                 scan_bound: int = 3000) -> dict:
    """v3.9: copy a bounded SAMPLE of an image dataset into the sandbox:
    up to `cap` images into ./images/, matching YOLO label .txt (same stem)
    into ./labels/, plus class names from data.yaml if present. The walk stops
    after `scan_bound` files so huge datasets can't stall the request.
    Returns {staged: [names], n_seen, truncated, classes, labels_staged}."""
    import shutil as _sh
    imgs, labels, seen, truncated = [], {}, 0, False
    for base, dirs, fnames in os.walk(src_root):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for fn in sorted(fnames):
            seen += 1
            if seen > scan_bound:
                truncated = True
                break
            p = Path(base) / fn
            ext = p.suffix.lower()
            if ext in _IMG_EXTS and len(imgs) < cap:
                imgs.append(p)
            elif ext == ".txt":
                labels[p.stem] = p
        if truncated:
            break
    (workdir / "images").mkdir(exist_ok=True)
    staged, labels_staged = [], 0
    for p in imgs:
        name = p.name
        if (workdir / "images" / name).exists():  # name collision across subdirs
            name = f"{p.parent.name}_{p.name}"
        _sh.copy(p, workdir / "images" / name)
        staged.append(name)
        lb = labels.get(p.stem)
        if lb:
            (workdir / "labels").mkdir(exist_ok=True)
            _sh.copy(lb, workdir / "labels" / (Path(name).stem + ".txt"))
            labels_staged += 1
    classes = []
    for yml in ("data.yaml", "data.yml"):
        y = src_root / yml
        if y.is_file():
            try:
                import re as _re
                m = _re.search(r"names:\s*\[(.*?)\]", y.read_text(errors="replace"),
                               _re.S)
                if m:
                    classes = [c.strip().strip("'\"") for c in m.group(1).split(",")
                               if c.strip()]
            except Exception:
                pass
            break
    return {"staged": staged, "n_seen": seen, "truncated": truncated,
            "classes": classes, "labels_staged": labels_staged}


def image_brief(m: dict) -> str:
    """Compact description of the staged image sample for the codegen prompt."""
    if not m or not m.get("staged"):
        return ""
    lines = [f"- ./images/: {len(m['staged'])} image file(s) staged "
             f"(a sample{' of a larger dataset' if m.get('truncated') else ''}), "
             f"e.g. {', '.join(m['staged'][:6])}"]
    if m.get("labels_staged"):
        lines.append(f"- ./labels/: YOLO .txt for {m['labels_staged']} of them "
                     f"(same stem; lines are 'class cx cy w h', normalized)")
    if m.get("classes"):
        lines.append(f"- classes: {m['classes']}")
    return "\n".join(lines)


def stage_dataset(files: list, workdir: Path) -> list:
    """Materialize the parsed tables as CSVs in the sandbox dir (the generated
    script reads copies — never the original dataset)."""
    import csv as _csv
    names = []
    for f in files:
        cols = f["cols"]
        keys = list(cols.keys())
        n = max((len(v) for v in cols.values()), default=0)
        out = workdir / f["name"]
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = _csv.writer(fh)
            w.writerow(keys)
            for i in range(n):
                w.writerow([cols[k][i] if i < len(cols[k]) else "" for k in keys])
        names.append(f["name"])
    return names
