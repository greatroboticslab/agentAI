#!/usr/bin/env python3
"""
EMACF — Embodied Multi-Agent Cognitive Framework

Entry point for the cloud-side multi-agent system.
Initializes all agents, starts EventBus, EdgeBridge, and Dashboard.

Usage:
    python main.py
    python main.py --config config/default.yaml
    python main.py --no-dashboard
"""

import argparse
import asyncio
import sys


def main():
    parser = argparse.ArgumentParser(description="EMACF - Embodied Multi-Agent Cognitive Framework")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to configuration file")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Start without the web dashboard")
    args = parser.parse_args()

    try:
        from core.embodied_team import EmbodiedTeam
        team = EmbodiedTeam(config_path=args.config)
        asyncio.run(team.start(dashboard=not args.no_dashboard))
    except KeyboardInterrupt:
        print("\n[*] Shutting down EMACF...")
    except ImportError as e:
        print(f"[!] Missing dependency: {e}")
        print("    Run: pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
