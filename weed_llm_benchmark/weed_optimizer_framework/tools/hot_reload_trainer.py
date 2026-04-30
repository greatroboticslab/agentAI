"""
v3.0.26 Hot-Reload Trainer — TRUE PARALLEL training that re-reads the
registry mid-run.

Architecture per professor's directive (REQ-1):
  - Job-D writes new datasets to dataset_registry.json (atomic rename)
  - Job-T (this) trains in K-epoch mini-phases
  - Between phases, re-snapshot registry, re-merge data, continue training
    from previous phase's best.pt (progressive transfer-learning chain)
  - Plateau detection across phases triggers early exit

Why mini-phases instead of subclassing ultralytics BaseTrainer:
  - BaseTrainer internals change between ultralytics versions; subclasses
    break on every upgrade.
  - Mini-phases use stable public APIs (`YOLO(weights).train(...)`).
  - Each phase's `model.train()` overhead is small (~30s-2min) vs. K
    epochs of training time (hours).
  - Same logical effect as a true mid-epoch hot-reload: training run sees
    growing data without restart.

Trade-off:
  - Optimizer momentum (Adam state) is reset each phase. Mitigated by
    starting each phase from the previous phase's best.pt and using a
    lower lr (0.0005) so the model fine-tunes rather than re-learns.
  - Acceptable per published continual-learning practice (CL-DETR, ERD)
    which use episodic phases for the same reason.

References:
  - CL-DETR (arXiv:2304.03110) — episodic continual detection
  - ERD (arXiv:2204.02136) — elastic response distillation across phases
  - Unbiased Teacher (arXiv:2102.09480) — pseudo-label SSL with phases
"""

import glob
import logging
import os
import time
from pathlib import Path

from ..config import Config
from . import registry_lock
from .mega_trainer import (
    _merge_datasets,
    _resolve_best_pt,
)

logger = logging.getLogger(__name__)


def _read_latest_metric(best_pt_path):
    """Read the latest mAP50-95 from results.csv next to best.pt."""
    if not best_pt_path:
        return None
    res = Path(best_pt_path).parent.parent / "results.csv"
    if not res.exists():
        return None
    try:
        lines = res.read_text().strip().splitlines()
        if len(lines) < 2:
            return None
        # Header: epoch,time,train/box_loss,...,metrics/mAP50-95(B),...
        header = lines[0].split(",")
        last = lines[-1].split(",")
        for i, h in enumerate(header):
            if "mAP50-95" in h:
                try:
                    return float(last[i])
                except (IndexError, ValueError):
                    return None
    except Exception as e:
        logger.warning(f"[hot-reload] failed to read {res}: {e}")
    return None


def _count_datasets_in_snapshot(snap_path):
    data = registry_lock.safe_read_json(snap_path) or {}
    return len(data.get("datasets", {}))


def hot_reload_train(strategy):
    """K-epoch mini-phase training with registry re-snapshot between phases.

    Strategy keys (extends mega_trainer.train_yolo_mega's):
      epochs_per_phase    int — K, default 5
      max_phases          int — cap, default 50 (safety)
      walltime_soft_sec   float — exit cleanly N seconds before SLURM cap,
                                  default 47*3600 (= 47h, 1h before 48h cap)
      base_model          str — initial weights path
      val_dataset_root    str — cwd12 holdout root
      include_autolabel   bool — default True
      epochs              ignored (use epochs_per_phase * num_phases)
      imgsz, batch_size, lr, workers — passed through to model.train

    Returns:
      (final_best_pt, summary_dict)
    """
    from ultralytics import YOLO
    import torch

    K = int(strategy.get("epochs_per_phase", 5))
    MAX_PHASES = int(strategy.get("max_phases", 50))
    WALLTIME_SOFT = float(strategy.get("walltime_soft_sec", 47 * 3600))

    # Resolve initial weights
    base_pt = strategy.get("base_model")
    if not base_pt or not os.path.exists(base_pt):
        candidates = sorted(
            glob.glob(f"{Config.FRAMEWORK_DIR}/mega_iterv3_0_*/train*/weights/best.pt"),
            key=os.path.getmtime, reverse=True,
        )
        base_pt = candidates[0] if candidates else None
        if base_pt:
            logger.info(f"[hot-reload] auto-detected base_model: {base_pt}")
    if not base_pt or not os.path.exists(base_pt):
        base_pt = Config.DETECTION_MODEL  # final fallback
        logger.info(f"[hot-reload] no prior best.pt; starting from {base_pt}")

    registry_path = f"{Config.FRAMEWORK_DIR}/dataset_registry.json"
    snap_dir = f"{Config.FRAMEWORK_DIR}/snapshots"
    os.makedirs(snap_dir, exist_ok=True)

    start = time.time()
    metrics_history = []  # list of (phase, mAP50_95, n_datasets)
    plateau_count = 0
    last_n_datasets = -1

    for phase in range(1, MAX_PHASES + 1):
        elapsed = time.time() - start
        if elapsed > WALLTIME_SOFT:
            logger.info(f"[hot-reload] walltime soft limit reached ({elapsed/3600:.1f}h); exiting at phase {phase-1}")
            break

        # 1) Snapshot the registry (atomic), so we have a frozen view for this phase
        snap_path = registry_lock.snapshot_registry(registry_path, snap_dir)
        n_datasets_now = _count_datasets_in_snapshot(snap_path)
        new_datasets_added = (n_datasets_now > last_n_datasets) and (last_n_datasets >= 0)
        logger.info(
            f"[hot-reload] phase {phase}: registry has {n_datasets_now} datasets "
            f"(prev={last_n_datasets}, new={'YES' if new_datasets_added else 'no change' if last_n_datasets >= 0 else 'first'})"
        )
        last_n_datasets = n_datasets_now

        # 2) Build merged dataset (fresh each phase — picks up new data)
        iteration = f"v3_0_26_phase_{phase}"
        merged_dir = f"{Config.FRAMEWORK_DIR}/mega_iter{iteration}"
        try:
            _, data_yaml, stats, used_datasets, names_list = _merge_datasets(
                merged_dir,
                include_autolabel=bool(strategy.get("include_autolabel", True)),
                val_dataset_root=strategy.get("val_dataset_root"),
            )
        except Exception as e:
            logger.exception(f"[hot-reload] phase {phase} merge failed: {e}")
            continue

        if stats["images"] < 100:
            logger.warning(f"[hot-reload] phase {phase} merge produced too few images ({stats['images']}); skipping phase")
            time.sleep(60)
            continue

        logger.info(
            f"[hot-reload] phase {phase}: merged {stats['images']} imgs from {stats['datasets']} datasets, "
            f"weed instances={stats.get('weed_class_instances', {})}"
        )

        # 3) Train K epochs from latest best.pt
        try:
            model = YOLO(base_pt)
        except Exception as e:
            logger.exception(f"[hot-reload] phase {phase}: YOLO({base_pt}) failed: {e}")
            base_pt = Config.DETECTION_MODEL
            model = YOLO(base_pt)

        try:
            model.train(
                data=data_yaml,
                epochs=K,
                imgsz=int(strategy.get("imgsz", 1024)),
                batch=int(strategy.get("batch_size", 5)),
                device=0,
                project=merged_dir,
                name="train",
                patience=999,  # don't early-stop within phase
                lr0=float(strategy.get("lr", 0.0005)),
                workers=int(strategy.get("workers", 4)),
                verbose=False,
                save_period=1,
                cos_lr=True,
                mosaic=1.0,
                mixup=0.1,
            )
        except Exception as e:
            logger.exception(f"[hot-reload] phase {phase}: model.train failed: {e}")
            del model
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            continue

        # 4) Resolve new best.pt and read mAP
        new_best = _resolve_best_pt(model, merged_dir)
        del model
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        if new_best and os.path.exists(new_best):
            base_pt = new_best
            mAP = _read_latest_metric(new_best)
            metrics_history.append({
                "phase": phase,
                "mAP50_95": mAP,
                "n_datasets": stats["datasets"],
                "n_images": stats["images"],
                "elapsed_h": round(elapsed / 3600, 2),
            })
            logger.info(
                f"[hot-reload] phase {phase} done: mAP50-95={mAP}, "
                f"new base_pt={new_best}"
            )
        else:
            logger.warning(f"[hot-reload] phase {phase}: no best.pt found")

        # 5) Plateau check — sustained plateau across last 3 phases AND no new datasets
        if len(metrics_history) >= 3:
            recent = [m["mAP50_95"] for m in metrics_history[-3:] if m["mAP50_95"] is not None]
            if len(recent) == 3:
                spread = max(recent) - min(recent)
                if spread < 0.005 and not new_datasets_added:
                    plateau_count += 1
                    logger.info(
                        f"[hot-reload] plateau hit (spread={spread:.4f}, no new data); "
                        f"plateau_count={plateau_count}/2"
                    )
                    if plateau_count >= 2:
                        logger.info(f"[hot-reload] sustained plateau + saturated registry, exiting at phase {phase}")
                        break
                else:
                    plateau_count = 0

    summary = {
        "phases_completed": phase,
        "final_base_pt": base_pt,
        "metrics_history": metrics_history,
        "elapsed_h": round((time.time() - start) / 3600, 2),
    }
    logger.info(f"[hot-reload] DONE: {summary}")
    return base_pt, summary
