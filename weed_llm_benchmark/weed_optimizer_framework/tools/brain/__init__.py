"""Tiered-supervision support code (v3.25.0).

Home of the deterministic pieces the supervision layer is built on: the
append-only artifact trace, the signal functions over those artifacts, the
evidence bundle, the policy gate and the scheduler health alarm. The legacy
`brain.py` / `orchestrator.py` research scripts are frozen and nothing here
imports them.

This file stays a docstring on purpose. Modules here are imported from inside
SLURM job bodies, from an Ultralytics callback and from the dashboard's HTML
middleware; a re-export in the package `__init__` would make every one of those
callers pay for the imports of all the others (FastAPI, Mongo, torch) and would
let one broken module take the rest down with it. Import the module you need.
"""
