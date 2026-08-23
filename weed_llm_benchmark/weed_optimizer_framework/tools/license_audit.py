"""License capture for harvested datasets (SUPERWEED_PLAN §1.6, M0 hygiene).

The 2026-08-23 audit found 0/45 registry datasets with a recorded license — nothing
harvested may be redistributed until that is fixed. This module resolves a dataset's
license from its source API by slug convention and writes it into the registry's
`provenance` block:

    kg_<owner>__<name>    Kaggle    datasets/view -> licenseName
    rf_<ws>__<proj>       Roboflow  universe project -> license
    gh_<owner>__<repo>    GitHub    /repos -> license.spdx_id (repo license; the
                                    DATA inside may differ — recorded as repo-level)
    hf-style <owner>__<name> or plain names: Hugging Face datasets API -> license tag

Used two ways:
  * at harvest time — dataset_discovery calls `detect_license()` on registration
  * retroactively — `python -m weed_optimizer_framework.tools.license_audit backfill`
    fills every registry entry that has no license yet (locked registry update),
    and `... report` prints the mix.

"unresolved" means the source API answered but names no license; "unreachable"
means the API could not be queried — both stay non-redistributable.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REG = os.path.join(REPO, "results", "framework", "dataset_registry.json")


def _http_json(url, headers=None, timeout=20):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _kaggle_token():
    for p in ("~/.kaggle_token", "~/.kaggle/access_token"):
        p = os.path.expanduser(p)
        if os.path.isfile(p):
            return open(p).read().strip()
    return os.environ.get("KAGGLE_API_TOKEN", "")


def _hf_token():
    for p in ("~/.hf_token", "~/.cache/huggingface/token"):
        p = os.path.expanduser(p)
        if os.path.isfile(p):
            return open(p).read().strip()
    return os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")


def _roboflow_key():
    p = os.path.expanduser("~/.roboflow_key")
    if os.path.isfile(p):
        return open(p).read().strip()
    return os.environ.get("ROBOFLOW_API_KEY", "")


def _norm(lic):
    lic = (lic or "").strip()
    return lic if lic else "unresolved"


def detect_license(slug: str) -> dict:
    """Return {"license": str, "license_source": str} — never raises."""
    try:
        if slug.startswith("kg_"):
            ref = slug[3:].replace("__", "/", 1)
            tok = _kaggle_token()
            if not tok:
                return {"license": "unreachable", "license_source": "kaggle:no-token"}
            d = _http_json("https://www.kaggle.com/api/v1/datasets/view/" + ref,
                           {"Authorization": "Bearer " + tok})
            lic = d.get("licenseName") or (d.get("licenses") or [{}])[0].get("name")
            return {"license": _norm(lic), "license_source": "kaggle:datasets/view"}
        if slug.startswith("rf_"):
            ws, _, proj = slug[3:].partition("__")
            key = _roboflow_key()
            if not key:
                return {"license": "unreachable", "license_source": "roboflow:no-key"}
            d = _http_json("https://api.roboflow.com/%s/%s?api_key=%s" % (ws, proj, key))
            lic = (d.get("project") or {}).get("license") or d.get("license")
            return {"license": _norm(lic), "license_source": "roboflow:project"}
        if slug.startswith("gh_"):
            owner, _, repo = slug[3:].partition("__")
            d = _http_json("https://api.github.com/repos/%s/%s" % (owner, repo),
                           {"Accept": "application/vnd.github+json",
                            "User-Agent": "weed-llm-benchmark-license-audit"})
            lic = (d.get("license") or {}).get("spdx_id")
            if lic == "NOASSERTION":
                lic = "unresolved"
            return {"license": _norm(lic), "license_source": "github:repo (repo-level)"}
        # Hugging Face style: owner__name (e.g. francesco__grass_weeds)
        if "__" in slug:
            ref = slug.replace("__", "/", 1)
            hdrs = {}
            tok = _hf_token()
            if tok:
                hdrs["Authorization"] = "Bearer " + tok
            d = _http_json("https://huggingface.co/api/datasets/" + ref, hdrs)
            lic = d.get("cardData", {}).get("license") or next(
                (t.split(":", 1)[1] for t in d.get("tags", [])
                 if isinstance(t, str) and t.startswith("license:")), None)
            if isinstance(lic, list):
                lic = ",".join(str(x) for x in lic)
            return {"license": _norm(lic), "license_source": "huggingface:datasets-api"}
        return {"license": "unresolved", "license_source": "local:no-source-api"}
    except urllib.error.HTTPError as e:
        return {"license": "unreachable", "license_source": "http-%d" % e.code}
    except Exception as e:
        return {"license": "unreachable", "license_source": type(e).__name__}


def backfill(only_missing=True, sleep_s=0.6):
    from .registry_lock import update_registry

    reg = json.load(open(REG))
    ds = reg.get("datasets", {})
    todo = []
    for slug, info in ds.items():
        if not isinstance(info, dict):
            continue
        have = ((info.get("provenance") or {}).get("license")
                or info.get("license"))
        if only_missing and have and have not in ("unresolved", "unreachable"):
            continue
        todo.append(slug)
    print("resolving %d/%d datasets..." % (len(todo), len(ds)))
    resolved = {}
    for i, slug in enumerate(todo):
        resolved[slug] = detect_license(slug)
        print("  %-52s -> %-28s (%s)" % (slug[:52], resolved[slug]["license"],
                                         resolved[slug]["license_source"]))
        time.sleep(sleep_s)                      # be polite to the source APIs

    def _apply(r):
        for slug, res in resolved.items():
            info = r.get("datasets", {}).get(slug)
            if isinstance(info, dict):
                prov = info.setdefault("provenance", {})
                prov["license"] = res["license"]
                prov["license_source"] = res["license_source"]
                prov["license_checked_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        return r

    update_registry(REG, _apply)
    print("registry updated (locked write).")
    report()


def report():
    from collections import Counter
    reg = json.load(open(REG))
    mix = Counter()
    for slug, info in reg.get("datasets", {}).items():
        if isinstance(info, dict):
            mix[(info.get("provenance") or {}).get("license") or "MISSING"] += 1
    print("license mix:", dict(mix.most_common(20)))
    bad = sum(v for k, v in mix.items() if k in ("MISSING", "unresolved", "unreachable"))
    print("non-redistributable (missing/unresolved/unreachable): %d" % bad)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "backfill":
        backfill()
    elif cmd == "report":
        report()
    else:
        print("usage: license_audit.py [backfill|report]")
