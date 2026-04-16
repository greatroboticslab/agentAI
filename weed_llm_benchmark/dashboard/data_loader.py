"""
Data Loader — Load and parse all framework data for the dashboard.

Reads JSON files from results/framework/ and results/.
Supports local files (primary) and optional SSH fetch from cluster.
"""

import json
import os
import glob
from pathlib import Path
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FRAMEWORK_DIR = os.path.join(RESULTS_DIR, "framework")


def load_all_data():
    """Load all available data into a single dict."""
    return {
        "memory": load_memory(),
        "run_log": load_run_log(),
        "results": load_all_results(),
        "benchmark": load_benchmark(),
        "slurm_logs": find_slurm_logs(),
        "base_dir": BASE_DIR,
    }


def load_memory():
    """Load memory.json (experiments, lessons, baseline)."""
    path = os.path.join(FRAMEWORK_DIR, "memory.json")
    if not os.path.exists(path):
        return {"experiments": [], "hard_lessons": [], "learned_lessons": [],
                "baseline": {}, "current_best": {}, "meta": {}}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, KeyError):
        return {"experiments": [], "hard_lessons": [], "learned_lessons": [],
                "baseline": {}, "current_best": {}, "meta": {}}


def load_run_log():
    """Load run_log.json (agent rounds with actions)."""
    path = os.path.join(FRAMEWORK_DIR, "run_log.json")
    if not os.path.exists(path):
        return {"rounds": [], "tool_stats": {}}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, KeyError):
        return {"rounds": [], "tool_stats": {}}


def load_all_results():
    """Auto-discover and load all JSON result files."""
    results = {}
    for json_file in glob.glob(os.path.join(RESULTS_DIR, "**/*.json"), recursive=True):
        key = os.path.relpath(json_file, RESULTS_DIR).replace("/", "_").replace(".json", "")
        try:
            with open(json_file) as f:
                results[key] = json.load(f)
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def load_benchmark():
    """Load the main benchmark results (Phase 2: 15 models)."""
    path = os.path.join(RESULTS_DIR, "final_verified_results.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, KeyError):
        return {}


def find_slurm_logs():
    """Find all SLURM log files."""
    logs = []
    for log_file in sorted(glob.glob(os.path.join(FRAMEWORK_DIR, "slurm_*.out"))):
        stat = os.stat(log_file)
        logs.append({
            "path": log_file,
            "name": os.path.basename(log_file),
            "size_mb": round(stat.st_size / 1024 / 1024, 2),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })
    return logs


def parse_slurm_actions(log_path):
    """Parse Brain actions from a SLURM log file."""
    actions = []
    if not os.path.exists(log_path):
        return actions
    with open(log_path) as f:
        for line in f:
            if "Action:" in line and "orchestrator" in line:
                parts = line.strip().split("Action: ", 1)
                if len(parts) == 2:
                    action = parts[1].strip()
                    timestamp = line[:8] if len(line) > 8 else ""
                    actions.append({"time": timestamp, "action": action})
            elif "Reasoning:" in line and "orchestrator" in line:
                parts = line.strip().split("Reasoning: ", 1)
                if len(parts) == 2 and actions:
                    actions[-1]["reasoning"] = parts[1].strip()
            elif "Evaluation:" in line and "evaluator" in line:
                parts = line.strip().split("Evaluation: ", 1)
                if len(parts) == 2:
                    actions.append({"time": line[:8], "action": "EVAL_RESULT",
                                    "detail": parts[1].strip()})
            elif "AGENT ROUND" in line:
                parts = line.strip().split("AGENT ROUND ", 1)
                if len(parts) == 2:
                    actions.append({"time": line[:8], "action": "ROUND_START",
                                    "detail": parts[1].strip()})
            elif "[Filter]" in line:
                actions.append({"time": line[:8], "action": "FILTER_RESULT",
                                "detail": line.strip().split("[Filter] ", 1)[-1]})
            elif "[LoRA]" in line and "Inject" in line:
                actions.append({"time": line[:8], "action": "LORA_INJECT",
                                "detail": line.strip().split("[LoRA] ", 1)[-1]})
    return actions


def get_experiments_df():
    """Get experiments as a list of flat dicts for easy DataFrame creation."""
    memory = load_memory()
    rows = []
    for e in memory.get("experiments", []):
        r = e.get("result", {})
        s = e.get("strategy", {})
        rows.append({
            "iteration": e.get("iteration", 0),
            "name": s.get("name", "?"),
            "mode": s.get("mode", "strategy"),
            "old_f1": r.get("old_f1", 0),
            "new_f1": r.get("new_f1", 0),
            "old_map50": r.get("old_map50", 0),
            "new_map50": r.get("new_map50", 0),
            "old_map50_95": r.get("old_map50_95", 0),
            "new_map50_95": r.get("new_map50_95", 0),
            "forgetting": r.get("forgetting", False),
            "timestamp": e.get("timestamp", ""),
        })
    return rows
