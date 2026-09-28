"""Domain adapters (runner §5.2.1): the one place where a domain's filter
pipeline meets the domain-free funnel engine.

A domain config names its adapter ("adapter": "<module>" in
domains/<domain>.json); the module lives here and must provide every function
of INTERFACE. load() imports it and refuses (AdapterError) a module that lacks
any of them, so a partial adapter fails at load time, not in the middle of a
job.
"""
from __future__ import annotations

import importlib
import re

from .. import AdapterError

INTERFACE = ("census", "ledger_from_summaries", "crop_table", "known_truth", "unit_keys", "guard_pairs",
             "pool_rows", "base_rows", "increment_rows", "eval_rows", "never_train_guard", "text_encoder",
             "step1_features", "j1_scores", "bioclip_embedder", "label_rows", "name_status_v1")

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def check_interface(module):
    """The INTERFACE names the module lacks, or does not define as callables."""
    return [f for f in INTERFACE if not callable(getattr(module, f, None))]


def load(name):
    """The adapter module `name` (a module under this package, or an already
    imported module object), checked against INTERFACE."""
    if not isinstance(name, str):
        module = name
    else:
        if not _NAME_RE.match(name):
            raise AdapterError("adapter name %r is not a module name" % (name,))
        try:
            module = importlib.import_module("%s.%s" % (__name__, name))
        except ImportError as e:
            raise AdapterError("no adapter %r (%s)" % (name, e))
    missing = check_interface(module)
    if missing:
        raise AdapterError("adapter %s lacks %s" % (getattr(module, "__name__", module), ", ".join(missing)))
    return module


def for_domain(domain):
    """The adapter a loaded domain config names."""
    return load(domain.adapter)
