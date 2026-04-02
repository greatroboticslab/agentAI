#!/usr/bin/env python3
"""
CLI entry point for the Weed Optimizer Framework.

Usage:
    python -m weed_optimizer_framework.run                                    # agent mode (default)
    python -m weed_optimizer_framework.run --mode strategy                    # rigid pipeline mode
    python -m weed_optimizer_framework.run --brain deepseek_r1_7b --rounds 5  # stronger brain
    python -m weed_optimizer_framework.run --list-brains
    python -m weed_optimizer_framework.run --list-vlms
"""

import argparse
import logging
import sys
from .config import Config
from .orchestrator import Orchestrator


def setup_logging(log_file=None):
    """Configure logging to both console and file."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


def list_brains():
    """Print available brain models."""
    print("\nAvailable Brain models:")
    print("-" * 70)
    for key, info in Config.BRAIN_MODELS.items():
        print(f"  {key:<20s} {info['hf_id']}")
        print(f"  {'':20s} VRAM: ~{info['vram_gb']}GB — {info['description']}")
        print()


def list_vlms():
    """Print available VLM tools."""
    print("\nAvailable VLM tools (read-only):")
    print("-" * 70)
    for key, info in Config.VLM_REGISTRY.items():
        print(f"  {key:<18s} prec={info['precision']:.3f}  rec={info.get('recall', '?'):<5}  "
              f"mAP50={info['mAP50']:.3f}  — {info['description']}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Weed Optimizer Framework — SuperBrain + Tools + Memory"
    )
    parser.add_argument("--brain", default="qwen25_7b",
                        help=f"Brain model key ({', '.join(Config.BRAIN_MODELS.keys())})")
    parser.add_argument("--brain-id", default=None,
                        help="Custom HuggingFace model ID for Brain (overrides --brain)")
    parser.add_argument("--rounds", type=int, default=3,
                        help="Max optimization rounds (default: 3)")
    parser.add_argument("--no-improve-limit", type=int, default=2,
                        help="Stop after N rounds without improvement (default: 2)")
    parser.add_argument("--list-brains", action="store_true",
                        help="List available brain models")
    parser.add_argument("--list-vlms", action="store_true",
                        help="List available VLM tools")
    parser.add_argument("--mode", default="agent", choices=["agent", "strategy"],
                        help="agent=Brain decides each step; strategy=rigid pipeline (default: agent)")
    parser.add_argument("--log-file", default=None,
                        help="Save log to file")

    args = parser.parse_args()

    if args.list_brains:
        list_brains()
        return

    if args.list_vlms:
        list_vlms()
        return

    # Resolve brain model
    if args.brain_id:
        brain_model_id = args.brain_id
    elif args.brain in Config.BRAIN_MODELS:
        brain_model_id = Config.BRAIN_MODELS[args.brain]["hf_id"]
    else:
        print(f"Unknown brain key: {args.brain}")
        list_brains()
        sys.exit(1)

    # Setup logging
    log_file = args.log_file or str(Config.FRAMEWORK_DIR) + "/framework.log"
    setup_logging(log_file)

    # Run
    orchestrator = Orchestrator(
        brain_model_id=brain_model_id,
        max_rounds=args.rounds,
        max_no_improve=args.no_improve_limit,
        mode=args.mode,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()
