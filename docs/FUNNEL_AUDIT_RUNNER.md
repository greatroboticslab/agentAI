# FUNNEL runner: modules, formats and commands

Companion to docs/FUNNEL_AUDIT.md (the contract, pre-registration v1, commit 35d3bd5) and its transcription `INC_DIR/funnel/prereg_v1.json`. The contract says what is measured and why. This file says how the code that measures it is laid out, what every module exposes, what every file holds, and in which order the commands run.

Six builder groups implement it in parallel: G-stats, G-data, G-vision, G-labels, G-autopilot and G-loop. A group builds against the interfaces pinned here, not against another group's code. If this file and the contract disagree, the contract wins and this file is amended. Where the contract left a choice open, §9 records the choice made here and why.

This file covers steps F3 to F10 of the contract (§10) and the platform code of F1 (§8). It does not cover:
- F0 (done: the contract and `prereg_v1.json` are committed) or F11 (the H8 verifier refit, card X11, which gets its own runner amendment);
- F12 (claim resolution and records);
- scoring of the H10d domain dev (§9, item 13);
- the review-surface UI behind lever L14.

Paths. `INC_DIR` is `$REPO/results/framework/inc` on the cluster. On the lab it is `LAB_INC = <lab repo>/weed_llm_benchmark/results/framework/inc` (the tree `inc_autopilot/model.py` calls `LAB_REPO/results/framework/inc`). `PKG` is `weed_optimizer_framework/tools/`. `FUN` is `PKG/funnel/`. Section numbers in square brackets, such as [§5.4], refer to the contract.

---

## 1. Conventions every group follows

### 1.1 Package rules

- **Imports.** Inside `FUN`, imports are relative (`from ..inc import common as C`, `from . import domain`). No file in `FUN` spells the package name: the domain-free test (§7.5) strips it, but relative imports keep engine files free of any path that could carry a domain term.
- **Light modules.** `FUN/__init__.py`, `domain.py`, `ledger.py` and `claims.py` import the standard library only, because the lab ticker imports them. Every other module may import `numpy` at module level. `torch`, `open_clip`, `transformers`, `PIL`, `sklearn` and `joblib` are imported inside the functions that need them.
- **No scipy.** `estimate.py` implements the beta and binomial quantiles it needs (§5.1.6), so the bench env needs nothing new.
- **Pinned INC modules are never edited:** `inc/driver.py`, `gate.py`, `splits.py`, `common.py`, `scorer.py`, `lora.py`, `train.py`, `verify.py`, `select.py`, `relevance.py`, `audit.py` and `PKG/cwd12_species.py`. They are imported and called, including their underscore helpers (`verify._cut_task`, `verify._read_jsonl`, `verify._load_registry`, `verify._resolve_dir`, `verify._layout`, `verify.read_source_label`, `verify.class_join`, `select.dup_groups`, `select.read_pool_dhash`). `inc/realloop.py` and `inc/pilot.py` are not pinned; G-loop changes them (§5.6).
- **Domain-free engine.** Every `.py` under `FUN` outside `FUN/adapters/` and `FUN/domains/` holds no domain term (§7.5). Everything domain-specific comes from `FUN/domains/<domain>.json` or from the adapter the config names.
- **English only,** in code, comments, prompts and docs.

### 1.2 Determinism, hashing, atomic writes, refusals

- **Seeds.** Every seed is `C.stable_int(text)`. The seed text is recorded next to each draw, in the file the draw wrote. `numpy.random.default_rng(C.stable_int(text))` is the only generator. Iteration is over sorted keys.
- **Atomic writes.** Every file is written to `<name>.tmp` in the same directory and moved with `os.replace`. Helpers are in `FUN/__init__.py` (§5.1.1).
- **Header.** Every JSON output carries these top-level keys (the funnel-ledger/1 of [§8.2] already has `format`, `domain` and `inputs` at top level):
  ```
  "format": "funnel-<kind>/1", "domain": "<name>", "built_utc": "YYYY-MM-DDTHH:MM:SSZ",
  "prereg":  {"path": str, "sha256": str, "core_sha256": str},
  "contract": {"path": str, "sha256": str},
  "domain_config": {"path": str, "sha256": str},
  "code": {"<path relative to PKG>": sha256, ...},        # every module that produced the file
  "inputs": {"<logical name>": {"path": str, "sha256": str, "bytes": int}, ...},
  "seeds": {"<purpose>": "<seed text>", ...},
  "testing": bool
  ```
- **Freshness.** A consumer re-hashes every input its producer recorded that it also reads. On a mismatch it raises `StaleInput` naming the file. `VOLATILE_KEYS = ("built_utc", "seconds", "hostname", "slurm_job_id")` are left out of every determinism comparison.
- **Reruns.** The same inputs and parameters over an existing output are a no-op and return the existing record. Other inputs or parameters refuse unless `--force`. After the sample lock (§4.9), `--force` is refused for `name_status_v2.json`, `frames_v1*` and `sample_v1*` (`SampleLocked`).
- **Refuse, do not guess.** A missing key, an unknown id, an input from another build or a count that does not reconcile raises a named error. Nothing falls back to a default that changes a number.
- **CLI exit codes.** 0 done; 2 refused (any `FunnelError`, with the message on stderr); 1 crash.

### 1.3 Identifiers

Every unit, item and stratum has one string id, used in every file.

| Id | Grammar | Meaning |
|---|---|---|
| box | `b:<pool key>#<box>` | Box `<box>` (0-based index into `pool_meta.jsonl` `boxes`, = `crops.csv` `box`) of a pool image |
| image | `i:<pool key>` | A pool image |
| pre-pool image | `d:<slug>\|<rel>` | An image dropped at S2–S5; `<rel>` is verify's `rel` (`<split>/images/<name>` or `images/<name>`) |
| class | `c:<slug>\|<src_id>` | A source class; `<src_id>` is the registry class id as a decimal string, `*` for a wildcard source |
| cluster | `k:<slug>\|<src_id>\|<j>` | Visual cluster `j` (0–7) of a no-name class |
| source | `s:<slug>` | A source (registry slug) |
| pair | `p:<slug>\|<rel>\|<split>\|<eval key>` | A near_eval or cwd12_copy guard pair; `p:<slug>\|<rel>\|dup\|<kept pool key>` for an exact_dup twin |
| KT1 crop | `t1:<train_core key>#<box>` | A train_core box |
| KT2/KT3 crop | `t2:<copy key>#<box>` | A box of a copy in `cwd12_copies.jsonl` |
| KT7 photo | `t7:<observation id>/<photo id>` | An independent-truth photo |
| item | `C.sha256_text("funnel/v1/item/" + unit_id + "/" + group)[:16]` | The opaque id a labeller sees |
| stratum | `<group>/<key>=<value>/...` | Keys in the fixed order of §4.8 |

KT4, KT5 and KT6 members are pool boxes (`b:` ids) and carry a `kt` membership field.

**Disjointness keys** (contract §4.1). Every unit and every known-truth item has four keys:
- `source`: the slug, or `train_core`, or `kt7`.
- `near_dup3`: a 3-bit near-duplicate group from `select.dup_groups(hashes, bits=3)`, computed once over train_core, the copies, the pool and the KT7 photos. Its id is `n:<first member key>`.
- `provenance`: `prov:<train_core key>` for a copy and its twin; `prov:<slug>|<stem>` when the domain config declares a capture-stem regex for the source; otherwise `prov:<own key>`.
- `lab`: the name of the lab group the domain config lists the source in, or `src:<slug>` when it is in none.

Two things "share" when any of the four keys coincide. The **reference lab group** is the lab group that holds `sources.reference` (LuLab for the weed domain). "Outside the reference lab group" is how the engine says the contract's "sources not derived from cwd12" and "outside the Lu-lab group".

### 1.4 Errors

`FUN/__init__.py` defines `FunnelError(RuntimeError)` and these subclasses: `StaleInput`, `SampleLocked`, `PreregError`, `DomainError`, `LedgerError`, `StrataError`, `DrawError`, `EstimateError`, `ClaimsError`, `AdapterError`, `TaxonomyError`, `NamesError`, `RelationError`, `FetchError`, `EmbedError`, `JudgeError`, `LeakError`, `LeakCalibrationError(LeakError)`, `SheetError`, `RLError`, `QualifyError`, `DisjointnessError(QualifyError)`, `CircularityError(QualifyError)`, `RecoverError`, `NeverTrainHit(RecoverError)`, `GateNotMet(RecoverError)`, `CLIError`.

Each module raises only its own class or `StaleInput`/`SampleLocked`. The CLI maps any `FunnelError` to exit 2.

### 1.5 Tests

- **Form.** Tests are plain scripts in `weed_llm_benchmark/tests/`, run as `cd weed_llm_benchmark && python3 tests/test_X.py`. They copy the INC style: `check(name, cond, detail)`, a `FAILURES` list, a `SKIPS` list, and the closing line `"\n%d failure(s), %d skipped: %s"`. Exit 1 on any failure.
- **Isolation.** Before any import, each test sets `os.environ["INC_DIR"] = TMP/inc` and `os.environ["REPO"] = TMP/repo`, and copies `docs/FUNNEL_AUDIT.md` and `prereg_v1.json` into that tree.
- **No network, no GPU.** Model and HTTP clients are injected (a fake embedder, a fake text tower, a fake transport).
- **Optional dependencies.** A test that needs one of them prints `SKIP: <reason>` and adds the name to `SKIPS`.
- **Real data.** Numbers about real data in a test are read from the local artifacts under `LAB_INC` (`step1/*_summary.json`, `calibration.json`, `realloop_v1/*`, `funnel/census_v0.json`), never typed in. The crop sheets in `funnel/sheets/*.jpg` are not in git and are never fixtures.
- **Shared synthetic world.** `tests/funnel_world.py` (owned by G-data, §5.2.7) builds a small Step 1 world in verify's own formats. Until it lands, a group may write its own minimal fixtures in the formats of §4.

---

## 2. Package layout and ownership

```
PKG/funnel/
  __init__.py            G-stats   errors, paths, header, atomic writers, seeds
  __main__.py            G-loop    CLI: python -m <package>.tools.funnel <verb> ...
  domain.py              G-stats   domain config + prereg loader, schema, exam list
  ledger.py              G-stats   funnel-ledger/1 and ledger.jsonl rows
  strata.py              G-stats   stratum frames, allocation, visual clusters
  draw.py                G-stats   seeded stratified draw -> sample_v1.csv, sample lock
  estimate.py            G-stats   every estimator of [§5.4], hypothesis evaluators -> audit_v1.json
  claims.py              G-stats   claims register [§8.3]
  taxonomy.py            G-data    authority resolver (GBIF backbone kind) with on-disk cache (L12)
  names.py               G-data    name status v2 from the resolver and config patterns
  relation.py            G-data    relation audit (H0b, H0c, G3 visual), geometry match (H3a, H1-pre)
  fetch.py               G-data    lab fetches: cards, archives, KT7, known items, taxonomy, refetch
  embed.py               G-vision  DINOv2 ViT-B/14 crop and image features
  judges.py              G-vision  J-zs, J-knn1, J-knn2 score files
  leak.py                G-vision  copy detector (H6), calibration, scans -> leak_v1.json
  sheets.py              G-labels  blind multiple-choice sheets, boards, answer key
  rl.py                  G-labels  RL-A ingestion, RL-B client, gold rows
  qualify.py             G-labels  machine-judge and RL qualification, circularity, disjointness
  recover.py             G-labels  L13 overlays into INC_DIR/step1_r1/, arm manifests
  adapters/__init__.py   G-data    adapter registry and interface check
  adapters/inc_step1.py  G-data    Step 1 -> census, ledger, frames' raw material, known truth
  domains/weed.json      G-data    the weed domain config
PKG/inc/realloop.py      G-loop    --increment-sources recovered --step1-overlay (new protocol version)
PKG/inc/pilot.py         G-loop    build-baseline: recovery provenance check
weed_llm_benchmark/run_inc_funnel.sh   G-loop   sbatch wrapper
PKG/inc_autopilot/*      G-autopilot  evidence, diagnose, thresholds.json, levers.json, campaign,
                                      brain_plan, validate, remote, outcome, executor, model,
                                      panel.py (new)
PKG/model_router.py      G-autopilot  ROLES["adversary"]
PKG/brain/policy_actions.json  G-autopilot  rows for the new actions
weed_llm_benchmark/run_inc_plan.sh     G-autopilot  PLAN_ROLE (planner | adversary)
```

**Contract map.**

| Contract | Modules |
|---|---|
| [§3] stages, S7b, S8b, veto, H5a | adapters/inc_step1, names, taxonomy, ledger |
| [§4.1] known truth, disjointness, circularity | domain (config), adapters/inc_step1 (items and keys), qualify (enforcement), judges (banks), sheets (boards) |
| [§4.3] judge panel, RL, qualification | judges, embed, rl, sheets, qualify |
| [§5.1–5.3] units, strata, budget, draw | strata, draw |
| [§5.4] estimators | estimate |
| [§6] H0b/H0c, H1-pre, H3a exact | relation |
| [§6] H6 | leak |
| [§6] H0a, H1–H5, H7, H9/H9′, H11, H12 | estimate (H11 also outcome) |
| [§6] H10 | panel, outcome |
| [§7] recovery | recover, realloop, pilot |
| [§8.2–8.9] platform | inc_autopilot/*, model_router, claims, ledger |
| [§9] realloop_v2 | realloop, pilot, recover (arms), panel |
| [§10] F3–F10 | __main__, run_inc_funnel.sh, §6 of this file |

**Cross-group imports.** A group may import another group's module only through the functions pinned in §5. Signatures are given as Python. `Path` means a `pathlib.Path` or a string.

---

## 3. The domain config: `FUN/domains/<domain>.json`, format `funnel-domain/1`

G-stats writes the loader and the schema check (`domain.py`). G-data writes `weed.json`. `domain.validate(raw)` returns every problem found, and `domain.load` raises `DomainError` listing them. Unknown top-level keys are refused.

### 3.1 Schema

```
{
 "format": "funnel-domain/1",
 "domain": "weed",                                  # [a-z][a-z0-9_]*; = prereg "domain"
 "adapter": "inc_step1",                            # module under FUN/adapters/
 "domain_terms": [str, ...],                        # extra grep terms (substring, case-insensitive);
                                                    # must not hold an authority or provider kind
 "classes": {
   "targets": [{"id": int, "name": str, "common": str, "taxon": str,
                "rank": "species" | "genus", "genus": str,
                "not": [taxon, ...],                # named non-members, class policy [§5.1]
                "siblings": [target name, ...]}],   # targets it is confused with (Sp pairs)
   "other": {"id": int, "name": str},
   "genus_answer_unsure_for": [genus, ...],
   "small_class_train_boxes_below": int
 },
 "attractors": [{"id": str, "taxon": str, "common": str, "rank": "species" | "genus",
                 "confused_with": [target name, ...], "option": str}],
 "names": {
   "numeric_regex": str,                            # fullmatch on names.key(name)
   "generic_keys": [str], "generic_regex": str,
   "non_object_words": [str],                       # substring of the key
   "state_words": [str],                            # substring of the key
   "role_names": [str],                             # exact key
   "related_tokens": [str], "related_allowed_keys": [str],
   "frames": {"noinfo": [status], "named": [status], "excluded": [status]}
 },
 "taxonomy": {
   "authority": {"kind": "gbif_backbone", "match_url": str, "search_url": str, "version_url": str},
   "overrides": {"<name key>": {"taxon": str, "why": str}},   # project policy, R4
   "informative_via": ["scientific", "vernacular"],
   "mappable_via": ["scientific", "override"]
 },
 "stages": [{"id": str, "name": str, "role": str, "unit": str, "guard": bool,
             "recoverable": bool | str, "depends_on": [stage id], "filter": str}],
 "exams": {"decision": "dev", "non_decision": [str], "extra_non_decision": [str]},
 "sources": {
   "reference": "train_core",
   "lab_groups": {"<group>": [slug | "train_core" | external dataset name, ...]},
   "authoritative": {"<slug>": {"paper": str, "licence": str, "claimed_by": "H1"}},
   "licences": {"<slug>": str},
   "not_recoverable": {"<slug>": "<reason>"},
   "capture_stem_regex": {"<slug>": str},           # provenance groups by file stem, when confirmed
   "card_image_counts": {"<slug>": int},            # S0 cap evidence
   "card_resolvers": {"<slug>": {"fetch": [fetch spec], "class_table": {"<id>": {"name": str, "taxon": str}} | null,
                                 "table_source": str, "upstream_annotations": fetch spec | null}}
 },
 "known_truth": {
   "<KT id>": {"what": str, "allowed_uses": [str], "independent": bool, "claimed_by": hypothesis | null,
               "never_qualifies": [judge id], "sources": [slug], "split": null | str},
   "qualify_rl_on": [KT id], "forbidden": [split], "kt7": {"provider": fetch spec, "per_taxon_max": int,
               "taxa": [taxon], "licences": [str], "roles": {"exemplar": int, "g0_every": int}}
 },
 "identity_checks": [{"class": str, "n": int, "from": "KT1", "before": [lever], "excluded_taxa": [taxon],
                      "pass_share_min": float}],
 "judges": {
   "features": {"model": "facebook/dinov2-base", "pooling": "cls"},
   "panel": [{"id": str, "kind": "step1_probe" | "zero_shot" | "knn" | "rl", ...kind options}]
 },
 "reference_labeller": {
   "backends": {"RL-A": {"kind": "external", "model": str, "family": str, "may_see": ["pool", "known_truth"]},
                "RL-B": {"kind": "ollama", "model": str, "family": str, "may_see": ["pool", "known_truth", "eval"]}},
   "judged_families": [str],
   "prompt": str, "sheet_prompt": str, "pair_prompt": str, "pair_sheet_prompt": str,
   "options_tail": [str], "pair_options": [str],
   "boards": [{"id": str, "targets_from": KT id, "attractors_from": KT id, "per_option": int}],
   "sheet": {"items": int, "sentinels": int, "min_prevalence": float, "max_low_prior": int}
 },
 "sampling": {"G5_allocation": {"<slug>": int | "all"}, "G5_dup_twins": "all_sha_differs",
              "sentinel_weights": {"<KT id>": int}, "pair_sentinels": {"positive": int, "negative": int}},
 "leak": {"negative_source_pairs": [[source, source]], "families": {"<family>": {params}}},
 "recovery": {"levers": {"<R-x>": {"scope": str, "gate": [str]}}, "domain_dev": {"min_group_images": int}}
}
```

### 3.2 What `weed.json` must hold (G-data)

**Classes.**
- `classes.targets` are the 12 entries in `C.CLASS_NAMES[:12]` order, with ids 0–11.
- `taxon` is `cwd12_species.CWD12_BINOMIAL[name]`, except MorningGlory: `taxon` "Ipomoea", rank `genus` (class policy [§5.1], DEC-3).
- `common` is `cwd12_species.CWD12_COMMON[name]`.
- `not` is copied from `prereg_v1.json class_policy`.
- `siblings`: PalmerAmaranth ↔ Waterhemp.
- `other` is `{"id": 12, "name": "OtherPlant"}`.
- `genus_answer_unsure_for` is copied from the prereg.
- `small_class_train_boxes_below` is 200.

**Attractors.** At least *A. retroflexus*, *A. hybridus*, *Bassia scoparia*, *Erigeron canadensis*, *Euphorbia hirta*, *E. hypericifolia*, *Chamaecrista* (genus), *Digitaria* (genus), *Cyperus* (genus) and *Ambrosia trifida* [§4.1 KT5, KT7; §7 Ragweed identity]. Each has `confused_with` and an option text, for example `"Redroot pigweed (Amaranthus retroflexus)"`.

**Names.** The word lists are copies of verify's lists, split so that v2 can tell the kinds apart. `verify.py` itself is unchanged.
- `generic_keys` = `verify.GENERIC_NAME_KEYS` minus the object and disease words.
- The object words go to `non_object_words`, together with `verify.NON_PLANT_WORDS` minus the disease words, "unknown" and "novel", and the objects the contract names (greenhouse, hut, solar, shed, pave, musor).
- The disease and state words go to `state_words`, with cercospora, xanthomonas, mosaic, leafcurl, healthy, dryleaf and drygrass.
- `role_names` = ["crop", "crops"].
- `related_tokens` = `verify.CWD12_RELATED_TOKENS`; `related_allowed_keys` = `verify.OTHER_ALLOWED_KEYS`.
- `generic_regex` = `verify._GENERIC_RE.pattern`; `numeric_regex` = `[0-9]+`.
- `frames`: `noinfo` = [no_name, numeric, generic, unresolvable]; `named` = [taxon_resolved, target_related, target_synonym, role]; `excluded` = [non_object, state].

**Stages.** The table in §4.3, with `guard: true` and `recoverable: false` on S4 and S5.

**Exams.** `decision` "dev"; `non_decision` ["test", "ood22", "ood23", "imageweeds"] (`C.EVAL_SPLITS` minus dev); `extra_non_decision` ["domain_dev"] (H10d).

**Sources.**
- `lab_groups`: NDSU and LuLab as in `prereg_v1.json known_truth.lab_groups` (with `train_core` added to LuLab).
- `authoritative`: weed_crop and greenhouse, each with `claimed_by` "H1".
- `licences`: CC BY 4.0 for the Mendeley and AgML copies (DEC-7), and one entry per recoverable source, read from its card.
- `not_recoverable`: csgo, uav-wqshy, tomato-leaf and crop-health-advisor, each with the reason "non-plant or leaf-disease source" [§3.4, §7].
- `card_image_counts`: MH-Weed16 6,656.
- `card_resolvers`: MH-Weed16 (the article's Table 2 transcribed with `table_source` "PMC12179629 Table 2", and the Mendeley d3n3mgjjbv v2 annotation archive), weed_crop (10.17632/mthv4ppwyw.2) and greenhouse (hs7d7kpd3z/2), and the H3b sources' cards or project class lists.

**Known truth.**
- KT1–KT7 as in [§4.1] and `prereg_v1.json known_truth`.
- `claimed_by`: KT4 "H1", KT5 "H2b", KT6 "H3a".
- `never_qualifies`: KT7 ["J1", "J-zs"].
- `independent`: true only for KT7.
- `qualify_rl_on` = ["KT1", "KT2", "KT3", "KT7"].
- `kt7`: `per_taxon_max` 30; `roles` {"exemplar": 6, "g0_every": 3} (§4.6); `taxa` = the 12 targets plus the attractors.

**Identity check.** `identity_checks` = `[{"class": "Ragweed", "n": 30, "from": "KT1", "before": ["R-A"], "excluded_taxa": ["Ambrosia trifida"], "pass_share_min": 0.8}]`.

**Judges.**
```
[{"id": "J1",     "kind": "step1_probe", "role": "stratum_key"},
 {"id": "J-zs",   "kind": "zero_shot", "text_encoder": "adapter", "prompt_template":
      "a photo of {lineage} with common name {common}.",
  "non_object_prompts": [...], "collapse": {"attractors": "other", "named_non_targets": "other"}},
 {"id": "J-knn1", "kind": "knn", "bank": ["KT1"], "holdout": "session", "k": 10, "temperature": 0.07},
 {"id": "J-knn2", "kind": "knn", "bank": ["KT1", "KT4", "KT5", "KT6:calibration", "KT7:exemplar"],
  "holdout": "disjoint", "k": 10, "temperature": 0.07, "kt5_per_cell_max": 300},
 {"id": "J-vlm",  "kind": "rl", "backend": "RL-B"}]
```

**Reference labeller.**
- `backends`: RL-A {"model": "claude-opus-5-5", "family": "claude"} and RL-B {"model": "qwen3.8:27b", "family": "qwen"} (DEC-1, DEC-2).
- `judged_families` = ["bioclip", "dinov2", "qwen", "glm"].
- The prompts of §4.10.
- `options_tail` = ["Another plant (none of the numbered plants)", "Not a plant", "The box does not hold one plant (box invalid)", "Unsure"].
- `pair_options` = ["The same photograph (possibly cropped, resized, flipped, recoloured or re-compressed)", "Consecutive frames of the same scene", "Different photographs", "Unsure"].
- `boards`: B1 {"targets_from": "KT1", "attractors_from": "KT7"} and B2 {"targets_from": "KT7", "attractors_from": "KT7"}, each with `per_option` 6.
- `sheet` {"items": 15, "sentinels": 3, "min_prevalence": 0.2, "max_low_prior": 4}.

**Sampling.**
- `G5_allocation` = {rf_tuf: "all", csgo: "all", peradeniya: 20, 5nvic: 20, test-8qezo: 20, vitif: 20} (full slugs) [§5.2].
- `sentinel_weights` = {KT1: 100, KT2: 60, KT3: 40, KT4: 60, KT5: 80, KT6: 60, KT7: 150} (the contract's "about 400 + about 150").
- `pair_sentinels` = {"positive": 20, "negative": 20}.

**Leak.**
- `negative_source_pairs` = [["train_core", "project_agml__mh_weed16_weed_detection"]].
- `families`: flip, rot90, crop (0–20 %), brightness (±25 %), blur (Gaussian radius 1–2 px), shear (±10°), letterbox640, jpeg (quality 50–90) [§6 H6].

**Recovery.** `recovery.domain_dev.min_group_images` = 20.

### 3.3 Prereg and amendments

- `prereg_v1.json` is loaded by `domain.load_prereg`.
- Its `contract.sha256` must equal the sha256 of the contract file at `C.REPO / prereg["contract"]["path"]`, or at `$FUNNEL_CONTRACT` when that is set. Otherwise `PreregError`.
- **`core_sha256`** is the sha256 of the canonical JSON of the prereg with the key `amendments` removed. Canonical JSON means `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`, UTF-8.
- Every artifact records the prereg's raw sha256 and its `core_sha256`. A consumer compares `core_sha256`, so the sample-lock amendment (§4.9) does not make earlier artifacts stale.
- `domain.append_amendment(path, amendment)` is the only writer of the file. It refuses when anything but the `amendments` list would change, and writes atomically.

---

## 4. Shared data formats

All files are under `INC_DIR/funnel/` unless stated. "Cluster-only" means the file is never shipped to the lab (§6.3).

### 4.1 `census_v1.json` (F3; G-data, `format: funnel-census/1`)

```
{header...,
 "rows": [{"source": str, "src_id": str, "src_name": str, "label": class name, "n": int,
           "verdict": {verdict: n}, "pred_all": {class: n}, "pred_confident": {class: n}, "mean_p": {class: float},
           "name_status_v1": str, "name_status_v2": str, "lab": str, "kt": [KT id],
           "fail_modes": {fail code: n} | null,               # target-labelled rows only (below)
           "other_argmax_conflicts": {"n": int, "mean_p_other": float, "tau_other": float,
                                      "second": {class: n}, "mean_p_second": float} | null}],
 "totals": {"embedded_crops": int, "verdicts": {verdict: n}, "admitted_target_boxes": int,
            "small_boxes": int, "pool_images": int, "pool_boxes": int, "images": {verdict: n}},
 "reconciliation": {"ok": bool, "checks": [{"name": str, "got": num, "want": num, "want_from": str, "ok": bool}]},
 "veto": {"lost_boxes": int, "images": int, "by_blocker": {blocker: n}, "combinations": {"a+b": n},
          "per_class": {class: {"verified": n, "admitted": n}}, "contract_check": {...}},
 "s1_skipped": {"<slug>": {"reason": str, "registry_entry": {...verbatim, minus class_names...}}},
 "s8b_reject_class": {"eligible": {"<source>|<name>": n}, "drawn": {"<source>|<name>": n},
                      "drawn_by_source": {slug: n}, "eligible_top_cell": cell, "drawn_top_cell": cell,
                      "rejected_positive_sources_in_drawn": [slug]},
 "h5a": {"twins": int, "dropped_twins_with_target_box_kept_lacks": {"images": n, "boxes": n},
         "dropped_twins_more_informative_names": {"images": n}, "examples": [first 20]},
 "train_core_boxes_per_class": {class: n},
 "h12": {"known_items": {"path", "sha256"}, "items": [{"name", "present": bool, "slugs": [slug]}],
         "present": int, "total": int} | null,
 "name_status_v2": {"path": str, "sha256": str},
 "known_truth_counts": {KT id: int}}
```

Row keys `source` to `mean_p` are census_v0's, so v0 and v1 rows compare field by field. One row per (source, src_id, label).

**Fail codes** (target-labelled, not verified; the first rule that applies, consistent with `verify.verdicts`):
- `argmax_other_confident`: argmax j is the other class and p_j ≥ τ_other.
- `argmax_target_confident`: j is another target and confident.
- `argmax_wrong`: j ≠ label, not confident.
- `p_below_tau`: j = label, p_j < τ_label.
- `cos_below_sigma`: j = label, p_j ≥ τ_label, cos_j < σ_label.
- `failed`: no features.

**Blockers** of a verified target box in a non-admitted image, in priority order: `other_called_target`, `species_conflict`, `species_unknown`, `small_species_box`, `failed`.

**Reconciliation checks** (F3 acceptance [§10]), each asserted:
- embedded crops, the four verdict totals, admitted target boxes, small boxes and pool images, each against `admit_summary.json` or `pool_summary.json`;
- `veto.lost_boxes` = verified target boxes − admitted target boxes, both from `admit_summary.json`;
- every lost veto box has at least one blocker.

`veto.images` and `veto.by_blocker` are compared with the contract's [post hoc] 457 images and 564/276/145/4 in `veto.contract_check`, where they are recorded, not asserted. A failed asserted check raises `AdapterError` after the file is written with `reconciliation.ok: false`.

### 4.2 `ledger.jsonl` (F3; G-data, validated by `ledger.validate_unit_row`)

One JSON object per line, sorted by `id`. Box rows cover every box of every pool image, small boxes included. Pre-pool image rows (`d:` ids) cover every image dropped at S2–S5.

```
{"id": "b:<key>#<box>", "unit": "box", "source": slug, "key": pool key, "box": int, "crop_id": int | null,
 "label": class id, "src_id": str, "src_name": str, "wh_px": [int, int],
 "name_status_v1": str, "name_status_v2": str, "lab": str, "near_dup3": str, "provenance": str,
 "pred": class id | null, "p": float | null, "cos": float | null,          # J1 argmax, as verify.verdicts
 "p_target_max": float | null, "p_other": float | null,                    # from the full P vector
 "second": class id | null, "p_second": float | null,
 "fail": fail code | null, "blockers": [blocker] | null,
 "path": {"S3": "kept", "S4": "pass", "S5": "pass", "S6": "embedded" | "small",
          "S7": "target" | "other", "S8": "verified" | "conflict" | "unknown" | "n/a",
          "S9": "other_ok" | "conflict" | "n/a", "S10": "admitted" | "conflict" | "unknown",
          "S11": "base" | "increment_pool" | "<select status>", "S12": "evidenced" | "not_evidenced" | "n/a"},
 "failed_stages": [stage id, ...],          # in stage order
 "first_cause": stage id | null, "sole_cause": stage id | null,
 "kt": [KT id]}

{"id": "d:<slug>|<rel>", "unit": "image", "source": slug, "rel": str, "dhash": int | null,
 "path": {"S2": "pass" | reason, "S3": "kept" | "exact_dup", "S4": "pass" | "near_eval", "S5": "pass" | "cwd12_copy"},
 "near": {"split": str, "eval": str, "bits": int} | null, "twin_of": pool key | null,
 "failed_stages": [...], "first_cause": ..., "sole_cause": ...}
```

- `failed_stages` holds S3, S6, S8, S9 or S10 when that stage discarded or blocked the unit, and S12 when the unit's image sits in the increment pool of a source with no evidence. S11 is routing, never a failure.
- `sole_cause` is `failed_stages[0]` when there is exactly one failed stage, else null [§5.1].
- `J1` values come from the Step 1 probe. For the OtherPlant training crops they are the out-of-fold values (`verifier.npz other_oof_P`/`other_oof_cos`). The adapter re-derives `verify.verdicts` from them and must reproduce `pool_verdicts.npz` `verdict` and `pred` for every pool crop, or raise `AdapterError`.

### 4.3 `funnel_ledger.json`, format `funnel-ledger/1` (G-data writes through `ledger.py`)

This is the [§8.2] format. Fields marked [ext] are runner extensions, which the generic diagnoses need so that D17 and D19 run on any domain (R13).

```
{"format": "funnel-ledger/1", "domain": str, "inputs": {name: sha256}, header...,
 "derivation": "summaries" | "census",                                   [ext]
 "fingerprint": sha256,                                                   [ext]
 "name_status_version": "v1" | "v2",                                      [ext]
 "target_classes": [class name], "other_class": id,
 "stages": [{"id", "filter", "version", "unit": "box|image|source|class|dataset|increment",
             "role": str,                                                 [ext]
             "depends_on": [stage id], "recoverable": bool | str, "guard": bool,   [ext guard]
             "in": int | null, "kept": int | null, "discarded": {reason: n},
             "kept_by_label": {class: n} | null,                          [ext]
             "discarded_by_label": {class: {reason: n}} | null,           [ext]
             "by_source": {src: {"in", "kept", "discarded": {}}},
             "by_label_pred": {"<label>|<pred>": n},
             "calibration": {"known_truth_sets": [{"id", "sha256", "sources", "domain_score": [lo, hi],
                                                   "domain_basis": str}],
                             "precision", "recall", "domains_covered"} | null,
             "audit": null | {"sha256", "fn_rate": {"estimate", "interval", "n"}}}],
 "label_spaces": {src: {"classes": int, "kinds": {"named": n, "numeric": n, "none": n, "generic": n}, "boxes": int}},
 "domain_scores": {src: median, "reference": {"q05": float, "q50": float}},
 "reject_class": {"stage": id, "eligible_top_cell": cell | null, "drawn_top_cell": cell | null,
                  "sample_sources": {src: n} | null} | null}                [ext]
```

- **`fingerprint`** is the sha256 of the canonical JSON of the adapter's identity inputs (`{"admit_summary", "pool_summary", "select_summary", "calibration"}` → sha256). It is the same in both derivations. An audit records it; D17's condition (A) compares it.
- **`derivation: "summaries"`** is built from the Step 1 summaries and `census_v0.json` alone (`adapters.inc_step1.ledger_from_summaries`). It runs on the lab or the cluster, before F3, so that D17 and D19 can fire at F2. Its label spaces use name status v1: kinds `none` = no_name, `numeric`, `generic`, and `named` = everything else. Its `reject_class` comes from `step1/verifier_fit_info.json` when that file is present, else null.
- **`derivation: "census"`** is written by F3, with v2 kinds (`none` = no_name, `numeric`, `generic` = generic + unresolvable, `named` = the rest). Census replaces a summaries-derived ledger without `--force`. It never replaces a census-derived one with other inputs.

**Stages of the Step 1 adapter.** `role` is the key the generic diagnoses read (§5.5).

| id | role | unit | depends_on | guard | recoverable |
|---|---|---|---|---|---|
| S-1 | discovery | dataset | – | no | "harvest (outside Step 1)" |
| S0 | cap | image | – | no | true (R-F) |
| S1 | registry | source | – | no | "unknown_layout only" |
| S2 | read | image | S1 | no | false |
| S3 | dedup | image | S2 | no | true |
| S4 | guard | image | S2 | yes | false |
| S5 | guard | image | S2 | yes | false |
| S6 | size | box | S5 | no | false |
| S7 | join | box | S5 | no | true |
| S7b | name_status | class | S7 | no | true |
| S8b | reject_class_sample | box | S7b | no | "not a discard" |
| S8 | target_check | box | S7, S8b | no | true |
| S9 | other_check | box | S7, S8b | no | true |
| S10 | image_rule | image | S8, S9 | no | true |
| S11 | selection | image | S8, S10 | no | "routing" |
| S12 | evidence | source | S8 | no | "derived" |
| S13 | relevance | source | – | no | "retired" |
| S14 | gate | increment | S12 | no | "not a data filter" |

- The join stage's `kept_by_label` is the pool's boxes per joined class, small boxes included (`pool_summary.json boxes_per_class`).
- `target_check` has `kept_by_label` (verified per class) and `discarded_by_label` (conflict and unknown per class).
- `other_check` has `discarded` {"conflict": n} and `by_label_pred`.

### 4.4 `name_status_v2.json` (F3; G-data, `format: funnel-name-status/2`)

```
{header..., "frozen": true, "taxonomy_cache": {"path", "sha256"},
 "rule_order": ["target", "no_name", "numeric", "state", "non_object", "role", "generic",
                "target_synonym", "target_related", "taxon_resolved", "unresolvable"],
 "names": [{"source", "src_id", "name", "key", "status_v1", "status_v2", "via": "join|pattern|scientific|vernacular|override|none",
            "taxon": str | null, "rank": str | null, "boxes": int, "conflicts": int}],
 "by_status": {status: {"names": n, "boxes": n, "conflicts": n}},
 "frames": {"noinfo": [status], "named": [status], "excluded": [status]},
 "contract_check": {"broweed_narweed_unresolvable_boxes": n, "crop_role_boxes": n,
                    "non_object_boxes": n, "state_boxes": n, "contract": {...}}}
```

The rules, applied in `rule_order` (`names.status_v2`, §5.2.3):

| Status | Rule |
|---|---|
| target | the adapter's join gave the class a target label |
| no_name | empty key |
| numeric | `numeric_regex` fullmatch |
| state | a `state_words` substring |
| non_object | a `non_object_words` substring |
| role | exact `role_names` key |
| generic | a `generic_keys` key, or `generic_regex` fullmatch |
| target_synonym | a scientific (or override) resolution to a target taxon at the target's rank, or to a descendant of it |
| target_related | a resolution to a taxon in a target's genus or in any target's `not` list, or the key holds a `related_tokens` entry and is not in `related_allowed_keys` |
| taxon_resolved | any other scientific or vernacular resolution to a taxon |
| unresolvable | nothing above |

`status_v1` is computed by the adapter from `verify.other_name_status`, with numeric split out (`name_key(name).isdigit()`). `names.py` never names v1's status strings.

### 4.5 Lab fetch outputs (G-data)

**`taxonomy_cache.json`** (`funnel-taxonomy-cache/1`):
```
{header..., "authority": {"kind", "match_url", "search_url", "backbone_version": str, "version_fetched_utc"},
 "overrides": {"sha256": str, "entries": {...}},
 "responses": {"<kind>:<query>": {"url": str, "status": int, "sha256": str, "body": {...verbatim JSON...},
                                  "fetched_utc": str}},
 "resolved": {"<name key>": {"name": str, "taxon_key": int | null, "canonical": str | null, "rank": str | null,
                             "status": str, "accepted": str | null, "accepted_key": int | null,
                             "lineage": {"kingdom", "phylum", "class", "order", "family", "genus", "species"},
                             "match_type": str, "via": "scientific|vernacular|override|none",
                             "vernacular_candidates": [canonical, ...]}}}
```
`resolved` holds every name in `pool_summary.json per_slug[*].join` plus every target taxon, attractor taxon, `not` taxon and KT7 taxon.

**`known_items_v1.json`** (`funnel-known-items/1`, H12): `{header..., "sources_used": [{"ref", "sha256"}], "items": [{"name", "url", "doc_ref", "species_named": [target], "has_boxes": true, "slug_patterns": [regex]}]}`. It is hashed before any registry comparison. Census does the comparison (§4.1 `h12`).

**`cards/`**: one directory per slug, holding the fetched files as fetched. `cards/index.json` (`funnel-cards/1`): `{header..., "cards": {slug: [{"url", "file", "sha256", "bytes", "fetched_utc", "kind", "licence", "members": [{"name", "sha256"}] | null}]}}`. `members` lists archive members.

**`kt7/`**:
- `kt7/photos/<obs>_<photo>.<ext>`.
- `kt7/kt7_items.jsonl`, one row per photo: `{"id": "t7:<obs>/<photo>", "taxon", "target": class name | null, "attractor": attractor id | null, "observation_id", "photo_id", "url", "licence", "sha256", "quality_grade", "observed_on", "place", "query": {...}, "role": "exemplar" | "g0" | "sentinel"}`. `role` is assigned by `fetch.kt7_roles` (§4.6).
- `kt7/crops_kt7.csv` in verify's `CROP_FIELDS` with `set` "kt7", one row per photo, the box covering the whole photo (cx = cy = 0.5, w = h = 1). `crop_id` is 0..n−1 in its own id space; `label` is the target id, or the other id for an attractor; `src_name` is the taxon.

**`refetch/`** (R-F, optional): `refetch/images/<slug>/...` and `refetch/manifest.json`.

**`fetch_manifest.json`**: `{header..., "files": {relpath: sha256}}` over everything `fetch` wrote. It is checked on arrival on the cluster (§6.3).

### 4.6 Known-truth items (adapter output, in memory)

`adapter.known_truth(domain)` returns `{KT id: [item]}` with items
`{"id": unit id, "kt": KT id, "crop_id": int | null, "crop_set": "core|copy|pool|kt7", "truth": class id | null, "truth_taxon": str | null, "truth_kind": "target|attractor|other|non_object", "claimed": bool, "source", "lab", "near_dup3", "provenance", "session": str | null, "role": str | null}`.

| KT | Items | Truth |
|---|---|---|
| KT1 | every train_core crop | its label |
| KT2 | every `cwd12_copies.jsonl` box of a source other than the KT3 source, box-matched to its train_core twin (`verify.MATCH_TOL`) | the twin's label; unmatched boxes are left out |
| KT3 | the same for the three_season copy | as KT2 |
| KT4 | target-labelled boxes of the authoritative sources | the claimed label |
| KT5 | boxes whose v2 status resolves to an attractor or a `not` taxon | the claimed taxon |
| KT6 | mh_weed16 boxes of the card-resolved ids, only after the H3a exact part passes; split into `calibration` and `estimation` halves by `near_dup3` group, seed `funnel/v1/kt6/split` | the claimed class |
| KT7 | the `kt7_items.jsonl` rows | their taxon |

**KT7 roles** (`fetch.kt7_roles`). Per taxon, photos are sorted by id and permuted with seed `funnel/v1/kt7/roles/<taxon>`. The first `roles.exemplar` (6) are `exemplar`. Of the rest, position i (0-based) is `g0` when `i % roles.g0_every == roles.g0_every − 1`, else `sentinel`.

### 4.7 `frames_v1.json` and `frames_v1/<group>.csv` (F6; G-stats)

`frames_v1/<group>.csv` columns: `unit_id, unit, stratum, source, image_key, crop_id, lab, near_dup3, provenance, label, pred, score, allowed_judges, extra`.
- `score` is J1's `p_target_max` in G4 and empty elsewhere.
- `allowed_judges` is a `;`-separated list: the panel judges whose calibration material shares nothing with the stratum (`qualify.allowed_judges`, §5.4.3).
- `extra` is compact JSON.

`frames_v1.json` (`funnel-frames/1`):
```
{header..., "name_status_v2": {"path", "sha256"},
 "groups": {"<group>": {"design": "srs|two_stage|dual|planted|allocation|sentinel|identity",
                        "N": int, "planned": int, "minimum": int | null, "file": {"path", "sha256"},
                        "strata": {"<stratum id>": {"N": int, "n_planned": int, "definition": str,
                                                    "allowed_judges": [judge]}}}}}
```

### 4.8 Groups, strata and designs (G-stats, `strata.py` and `draw.py`)

`planned` and `minimum` per group come from `prereg_v1.json sampling.groups`. When a frame is smaller than planned, the group takes every unit.

**Allocation.** `strata.allocate(sizes, n, floor=5)`: each non-empty stratum gets `min(N_h, floor)`; the rest of n goes in proportion to `N_h` by largest remainder (ties by stratum id), never above `N_h`.

| Group | Frame | Stratum key order | Design |
|---|---|---|---|
| G0 | KT2 hidden targets + KT7 `g0` photos (targets and attractors) | `G0/all` | planted (below) |
| G1 | OtherPlant-labelled boxes with S9 = conflict | `frame, status, pred` | srs: excluded frame min(N, 10); the rest split equally between `noinfo` and `named`; within a frame, `allocate`. `pred` is the frame's top-3 predicted classes by count (ties by name), else `rest`. |
| G2 | target-labelled boxes, S8 ≠ verified, embedded | `source, label, fail` | srs, `allocate` |
| G2v | verified boxes in images with S10 ≠ admitted | `source` | srs, `allocate`; joint with G2a (below) |
| G2a | verified target boxes of sources outside the reference lab group | `source` | srs, `allocate`; joint with G2v |
| G3 | class units (`c:`) whose v2 status is in `noinfo`, is `target_related` or `target_synonym`, or is a numeric class, with ≥ 100 embedded boxes; a no-name class is replaced by its 8 visual clusters (`k:`), and a cluster enters only with ≥ 100 boxes | `unit` | two_stage (below) |
| G4 | OtherPlant-labelled boxes with S9 = other_ok | `frame, argmax_target, band` | dual (below) |
| G5 | `guard_pairs_v1.csv` pairs of sources outside the reference lab group, per `sampling.G5_allocation`, plus exact_dup twins whose sha256 differs | `source, split, bits` (bits bands 0-2, 3-4, 5-6) | allocation from the config; `allocate` inside a source |
| sentinel | KT items with role `sentinel` (KT7) or any item (KT1–KT6), excluding G0 and exemplar items | `kt, truth_kind` | sentinel (below) |
| identity | `identity_checks[*]` crops from their KT set, excluding board exemplars | `class` | srs, n from the config |

Box-level frames count embedded boxes only. Small boxes are reported, not sampled [§3.1 S6].

- **Planted (G0).** Share s = 0.2 + 0.6 × u, with u = `C.stable_int("funnel/v1/G0/share") / (2**31 − 1)`, and n_pos = round(s × n_G0). Positives: half from KT2 (seed `funnel/v1/G0/pos/kt2`), half from KT7 `g0` targets (seed `funnel/v1/G0/pos/kt7`). Negatives: KT7 `g0` attractors (seed `funnel/v1/G0/neg`). s, n_pos and the truth of every item are written only to the key (§4.9). The sample file shows G0 rows with `unit_id` `G0:<item_id>` and an empty `source`.
- **Two-stage (G3).** With M units: if 20 × M ≤ planned, every unit is taken and per-unit n = min(30, floor(planned / M)), capped at the unit size. Otherwise m = floor(planned / 20) units are drawn (seed `funnel/v1/G3/units`) and per-unit n = 20. π_i = (m/M) × (n_u/N_u).
- **Dual (G4).**
  - *Uniform part.* 300 SRS: 225 in the `named` frame and 75 in the `noinfo` frame (the H4 power rule [§6 H4]); the `excluded` frame is reported, not sampled. u_h = n/N.
  - *Prioritised part.* 150 by Poisson sampling over all of G4, with p_i = min(1, 150 × s_i / Σ s_j), where s_i is J1's `p_target_max` (seed `funnel/v1/G4/prio`).
  - *Union.* π_i = 1 − (1 − u_h)(1 − p_i). An item drawn by both parts is listed once.
  - `argmax_target` ∈ {yes, no}. `band` ∈ {1, 2, 3} is the tertile of `p_target_max` within the frame (numpy `quantile`, method "linear"). Both are post-strata for reporting only.
- **Joint (G2v, G2a).** The two draws are independent. A unit in both frames is listed once, with π = 1 − (1 − π_G2v)(1 − π_G2a), and counts for both groups' estimates.
- **Sentinel.**
  - Pool sheets hold `sheet.items` (15) items, of which `sheet.sentinels` (3) are sentinels. So the sentinel count is 3 × ceil(n_pool_items / 12) for pool sheets, and 3 × ceil(n_G5 / 12) pair sentinels for G5 sheets.
  - Pool sentinels are split over KT ids in proportion to `sentinel_weights` (`allocate`, floor 0). Within a KT id they are split over `truth_kind` in proportion to availability.
  - Pair sentinels: `pair_sentinels.positive` KT2 copy ↔ train_core twin pairs, and `pair_sentinels.negative` provenance-disjoint 7–10-bit pairs from `leak_pairs_v1.csv` negatives.

Seeds: `funnel/v1/<group>/<stratum id>` for each within-stratum draw [§5.2]; `funnel/v1/kt4` for any within-source KT4 draw [§4.1 K1].

### 4.9 `sample_v1.csv`, `sample_v1_key.jsonl` and the sample lock (F6; G-stats)

`sample_v1.csv` columns, one row per item, sorted by `group, stratum, draw_rank`:
`item_id, unit_id, unit, group, stratum, source, image_key, crop_id, pi, pi_parts, seed_text, draw_rank, sheet_class, lab, near_dup3, provenance, kt`.
- `pi_parts` is compact JSON, for example `{"uniform": 0.0042, "prio": 0.0110}` or `{"G2v": .., "G2a": ..}`.
- `sheet_class` is `pool` or `eval`. `eval` is for G5 items and pair sentinels: sheets that show evaluation pixels.
- A unit drawn into two groups has one row per group, with the same `pi` (the union).

`sample_v1_key.jsonl` (cluster-only) holds one row per item: `{"item_id", "unit_id", "truth": class id | null, "truth_taxon", "truth_kind", "pair_truth": "same|consecutive|different" | null}`, plus one row `{"planted": {"share", "n_pos", "n_neg", "seed_text"}}`.

**Sample lock.** Before `draw` returns it calls:
```
domain.append_amendment(prereg_path, {"id": "A<n>", "kind": "sample_lock", "date": "YYYY-MM-DD",
  "prereg_core_sha256", "sample_sha256", "key_sha256", "frames_sha256", "name_status_v2_sha256",
  "frame_sizes": {group: N}, "confirmatory_frames": {"H2a": N, "H2b": N, "H4_named": N, "H4_noinfo": N}})
```
The contract says the H2 v2 frame sizes are written at the lock [§6 H2]. From then on, `draw`, `strata` and `census` refuse to rewrite their outputs (`SampleLocked`), and every later step checks `sample_sha256` against the lock.

### 4.10 Sheets (F7; G-labels)

The directories:
- `sheets_v1/` holds pool sheets and may be shipped to the lab.
- `sheets_v1_cluster/` holds G5 pair sheets and is cluster-only.
- `sheets_v1_key/key.jsonl` is cluster-only.

Each sheet directory holds:
- `board_<id>.jpg` and `board_<id>.json`: `{"format": "funnel-board/1", "board_id", "image": {"file", "sha256", "w", "h"}, "options": [{"n", "text"}], "exemplars": {option n: [unit id]}}`. Pair sheets have no board.
- `sheet_<nnnn>.jpg` and `sheet_<nnnn>.json`: `{"format": "funnel-sheet/1", "sheet_id", "kind": "mc|pair", "image": {"file", "sha256", "w", "h"}, "board": {"id", "file", "sha256"} | null, "options": [{"n", "text"}], "question": str, "items": [{"item_id", "position", "panel": [x, y, w, h], "tiles": {"crop": [x, y, w, h], "context": [x, y, w, h]}}]}`. There is no source, stratum, verdict, score or proposal anywhere in a sheet file.
- `instructions.txt`: the sheet prompt with the options filled in.
- `index.json` (`funnel-sheets/1`): `{header..., "contains_eval_pixels": bool, "sheets": [{"sheet_id", "json_sha256", "image_sha256", "board_id", "n_items"}], "boards": [...], "key_sha256", "sample_sha256", "prereg_core_sha256"}`.

**Key rows** (`key.jsonl`): `{"item_id", "sheet_id", "position", "unit_id", "group", "stratum", "is_sentinel", "kt", "truth": id | null, "truth_taxon", "truth_kind", "pair_truth", "board_id"}`.

**Layout.**
- A panel is 448 × 244 px: a 20 px header "#<position>", then two 224 × 224 tiles.
- The left tile is the crop exactly as verify cuts it (`semisup_labeler._cut`, grey padding, 224 px).
- The right tile is the whole image letterboxed to 224 × 224, with the box drawn as a 2 px red rectangle. A KT7 photo shows the full frame.
- A sheet is 3 columns × 5 rows of panels (15 items).
- A pair panel shows image A (the pool image) and image B (the evaluation or twin image), each letterboxed to 224.
- A board row is "n. <option text>" and `per_option` 112 px exemplar tiles.

**Options** (`domain.options()`): the targets in config order, then the attractors in config order, then `options_tail`, numbered from 1. The same list is on every item of every pool sheet (§9, item 3). Pair sheets use `pair_options`.

**Boards.**
- An item goes on a sheet whose board shares nothing (§1.3) with it: B1 (KT1 target exemplars) unless the item shares with KT1 (the reference lab group, and KT1 and KT2 sentinels), else B2 (KT7 only).
- Exemplars are drawn per option with seed `funnel/v1/board/<board>/<option>`. KT1 exemplars come from 2 train_core sessions per class, chosen by seed. KT1 sentinels and identity items exclude those sessions. KT7 exemplars are the photos with role `exemplar`.

**Composition.**
- Items are sorted into per-group queues ordered by item id.
- Sheets are filled round-robin over groups in the order G0, G1, G2, G2v, G2a, G3, identity, G4. Each sheet holds at most `max_low_prior` (4) G4 items and exactly `sheet.sentinels` (3) sentinels.
- A sheet's expected prevalence is (sentinel targets + Σ over its other items of the stratum's J1 predicted-target share) / 15. It must be ≥ `min_prevalence`. The packer swaps a G4 item for the next G1 or G2 item until it is.
- Sentinel truth kinds are spread by a seeded schedule (`funnel/v1/sheets/sentinels`), 1–3 targets per sheet.
- Positions within a sheet are a seeded permutation (`funnel/v1/sheets/<sheet_id>`).

**Prompts** (in `weed.json`; the engine only formats `{options}`):
- `prompt` (one item, RL-B):
  ```
  You are checking plant labels for a scientific audit. The first image is a reference board: each numbered row shows example photographs of one option. The second image is one numbered panel: on the left, one object cut from a field photograph and enlarged; on the right, the whole photograph with the object's box drawn in red.
  Which ONE option names the object inside the red box?
  {options}
  Answer on one line: the option number, a comma, then YES if the box holds exactly one whole plant of the kind you chose, NO if it does not, or NA if the option you chose is not a plant kind. Example: 7, YES
  ```
- `sheet_prompt` (a whole sheet, RL-A): the same text, with the last paragraph replaced by "For every panel, answer on its own line: the panel number, a colon, the option number, a comma, then YES, NO or NA. Example: 3: 7, YES".
- `pair_prompt` and `pair_sheet_prompt`: "Panel images A (left) and B (right). Are they the same photograph (possibly cropped, resized, flipped, recoloured or re-compressed), consecutive frames of the same scene, or different photographs?", followed by `{options}` and "Answer with the option number only."

### 4.11 Answers and gold

**`rl_answers/<backend>/<sheet_id>.json`** (`funnel-rl-answers/1`):
```
{"format", "labeller": "RL-A:<model>" | "RL-B:<model>" | "human:<actor>", "backend": "RL-A|RL-B|human",
 "sheet_id", "sheet_sha256", "board_sha256" | null, "answered_utc", "model_digest": str | null,
 "answers": [{"item_id", "option": int | null, "box_ok": "yes|no|na" | null, "raw": str,
              "attempts": int, "status": "ok|unparsed"}]}
```

**`gold_v1.csv`** (F7 `ingest`; cluster), one row per (item, labeller):
`item_id, unit_id, unit, group, stratum, labeller, backend, option, answer, answer_level, answer_taxon, box_ok, is_sentinel, kt, truth, truth_kind, correct_species, correct_genus, correct_plant, sheet_id, position, answers_sha256`.
- `answer` is a class name, `other`, `non_object`, `invalid`, `unsure` or `unparsed`. For a pair it is `same`, `consecutive`, `different` or `unsure`.
- `answer_level` is `species`, `genus` (an option whose rank is genus) or `none`.
- The `correct_*` columns are filled for sentinels and identity items only.

### 4.12 Features and judge scores (F5; G-vision)

**DINOv2 crop shards.** `emb_dinov2/emb_sXXX_of_NNN.npz` use verify's shard format: arrays `crop_ids` (int64) and `X` (float16); `meta` a JSON string with `crops_sha256`, `embedder` "facebook/dinov2-base:cls", `dim`, `stats`, `shard`, `nshards`, `crops`, `seconds` and `built_utc`. They cover every `crops.csv` row. A row that fails is NaN.

**Other feature files.**
- `emb_dinov2_kt7.npz`: the same format over `kt7/crops_kt7.csv`.
- `emb_bioclip_kt7.npz`: BioCLIP-2 features of KT7, through the adapter's embedder, for J1 on KT7.
- `emb_dinov2_images_<set>.npz`: whole-image descriptors for `leak` (§4.13).

**Judge score files.** `judges/<judge>__<set>.npz`, set ∈ {crops, kt7}. Arrays: `unit_index` (int64: crops.csv crop_id or kt7 crop_id), `P` (float16 [n, L]) and `top` (int16). `meta` is a JSON string: `{"judge", "labels": [...], "set", "features": {"path", "sha256"}, "bank": {"kt": [...], "n": int, "sha256"} | null, "exclusion": "session|disjoint|none", "k", "temperature", "prompts": [...] | null, "crops_sha256"}`.
- Label space L = the targets, then "other", then "non_object" for J-zs. J-knn1 has targets only. J-knn2 has targets and "other".
- J1's scores are not a file: they are the `ledger.jsonl` values, plus `judges/J1__kt7.npz` from the probe on `emb_bioclip_kt7.npz`.

### 4.13 `leak_v1.json` (F4; G-vision, `funnel-leak/1`)

```
{header...,
 "detector": {"descriptor": {"model": "facebook/dinov2-base", "pooling": "cls", "view": "processor default"},
              "dhash_variants": ["id", "hflip", "vflip", "rot90", "rot180", "rot270", "transpose", "transverse"],
              "dhash_bits_max": 6, "cos_threshold": float,
              "rule": "copy iff cos >= cos_threshold or min over variants of dHash bits <= dhash_bits_max"},
 "calibration": {"ok": bool, "positives": {family: {"n", "hits", "recall", "lb", "params_seed"}},
                 "negatives": {"pairs_7_10": {"n", "false_hits", "fpr", "ub"}, "hard": {"n", "false_hits", "fpr", "ub"}},
                 "recall_min": 0.95, "fpr_max": 0.01},
 "scans": {"source:<slug>" | "base_B" | "realloop_v1:<step>": {"images": int, "copies": int, "copy_found": bool,
                                                              "listed": [first 200 copies]}},
 "h6a": {"scope": [slug], "copy_found": {slug: bool}, "quarantine": [slug], "void_ood_arms_with": [slug]},
 "h6b": {"base_copy": bool, "increment_copies": {step: int}, "incident": bool},
 "h6c": {"groups": {group: [source]}, "added_by_h6a": {source: group}},
 "eval_descriptors": {"path", "sha256"}}                                   # the npz itself is cluster-only
```

- A copy entry is `{"key", "image", "eval_split", "eval_key", "cos", "bits", "variant"}`. Evaluation images appear only as keys.
- `leak_pairs_v1.csv` (cluster-only) lists every copy, and the calibration negatives, with columns `set, key, eval_split, eval_key, cos, bits, variant, kind`.

### 4.14 Relation outputs (G-data)

**`relation_geometry_v1.json`** (F4, `map --part geometry`, `funnel-relation-geometry/1`):
```
{header..., "matches": {"<slug>": {"upstream": {"path", "sha256"}, "pool_boxes": n, "matched_boxes": n,
     "matched_share": float, "tolerance_px": 2, "halves": {"seed_text", "a": [image keys sha], "b": ...},
     "alignments": [{"name": "identity|plus1|minus1|drop<d>", "agreement_a": float, "agreement_b": float, "n_a", "n_b"}],
     "chosen": name, "confirmed": bool, "stems": {"distinct": n, "files": n} | null}},
 "h3a_exact": {"pass": bool, "why": str}, "h1_pre": {"<slug>": {"pass": bool, "why": str}}}
```

**`class_maps.json`** (F4, `funnel-class-maps/1`): the proposals only, immutable after F4. Acceptance is decided by `recover` and recorded in `recovery.json` (§9, item 18).
```
{header..., "proposals": [{"source", "src_id", "src_name", "map_to": class name | null,
   "via": "card+geometry|card|taxonomy", "card": {"path", "sha256", "table_source"} | null,
   "geometry": {"alignment", "agreement"} | null, "status": "proposed|to_L14", "reason": str}]}
```
A class whose count on the card disagrees with the source's class count is `to_L14` [§8.5 L11].

**`relation_audit_v1.json`** (F5, `map --part relation`, `funnel-relation-audit/1`):
```
{header..., "judge": judge id,
 "h0b": {"units": [{"unit", "truth", "r": {class: float}, "mapped_to": class | null, "join": class,
                    "join_flagged": bool}], "pass": bool, "checks": [...]},
 "h0c": {"source", "units": [...], "class_accuracy": float, "maps_relative_to_target": [unit], "pass": bool},
 "visual_proposals": [{"unit", "map_to", "r", "status": "proposal_L14"}]}
```

### 4.15 Qualification files (G-labels)

**`judge_qualification.json`** (F5, locked, `funnel-judge-qualification/1`):
```
{header..., "locked": true,
 "judges": {"<judge>": {"calibration_material": {"kt": [...], "sources": [...], "labs": [...],
                                                 "near_dup3": sha256 of the sorted list, "provenance": sha256},
                        "by_type": {"<stratum type>": {"sets": [KT], "se": est, "sp": est,
                                    "precision_at_half": est, "rescue": est,
                                    "p_wrong_given_j1_wrong": float, "p_wrong": float, "qualified": bool, "why": str}},
                        "never_qualifies_on": [KT]}},
 "phi": {"<a>|<b>": float}, "correlated_groups": [[judge]],
 "h7": {"J-zs": {"predicted": "fail", "qualified": bool}, "J-knn1": {...}, "J-knn2": {"predicted": "qualify", ...}}}
```
- An `est` is `{"k", "n", "estimate", "lb", "ub"}` (§5.1.6).
- The stratum types are `shifted_target` (G2, G2v, G2a), `other_named` (G1 named, G4 named: sets KT7 attractors) and `other_noinfo` (G1 noinfo, G4 noinfo, G3: sets KT5 attractors).
- `precision_at_half` = Se / (Se + 1 − Sp); its bound uses Se's lower and Sp's lower bounds.
- `rescue` is the accuracy on items J1 got wrong in KT2 and KT7. KT4 agreement is reported apart.

**`rl_qualification.json`** (F7, after `ingest`, `funnel-rl-qualification/1`):
```
{header..., "gold_sha256",
 "backends": {"<backend>": {"<scope>": {"<level>": {"se": est, "sp": est, "qualified": bool}}}},
 "agreement_only": {"<backend>": {"KT4": est, "KT5": est, "KT6": est}},
 "pairs": {"RL-B": {"se": est, "sp": est, "qualified": bool}},
 "identity": {"<class>": {"n", "as_class": n, "as_excluded": n, "share": float, "pass": bool}},
 "primary": {"<hypothesis>": {"backend", "level", "scope"}},
 "j_vlm": {...as a judge, by type...}, "unsure_policy": "counted as wrong"}
```
- A **scope** is the `+`-joined list of the KT sets that count for a stratum: `qualify_rl_on` minus every set that shares anything with the stratum. The Lu-lab strata get `KT7` only (§9, item 5).
- Levels: `species`, `genus`, `plant`.
- Qualified means Se.lb ≥ 0.85 and Sp.lb ≥ 0.85 [§4.3].
- **Primary** per hypothesis (DEC-1): the qualified backend with the larger Se.lb + Sp.lb at the level the hypothesis needs; a tie goes to RL-B.

### 4.16 `audit_v1.json` (F8; G-stats, `funnel-audit/1`) and `audit_v1.md`

```
{header..., "ledger_fingerprint": sha256, "valid": bool, "calibration_overlap": [{"judge|rl", "stratum", "shared"}],
 "rl": {"primary": {...from rl_qualification}, "levels_used": {...}},
 "strata": [{"group", "stratum", "N", "n", "n_labelled", "event": str, "labeller", "level",
             "estimate": float, "interval": [lo, hi], "method": "wilson|jeffreys|ht|ppi|kg",
             "rogan_gladen": {"applied": bool, "se", "sp", "flag": str | null} ,
             "unsure": {"as_yes": est, "as_no": est}, "predictor": judge | null, "deff": float | null}],
 "stages": {"<stage id>": {"frames": [group], "fn_rate": {"estimate", "interval", "n", "method"},
                           "recoverable": {"estimate", "interval"}, "p_holm": float, "suspect": bool,
                           "r_min": int}},
 "funnel_recall": {"estimate", "interval", "draws": 20000, "seed_text", "small_boxes_outside": int},
 "label_frequency": {"<source>": {"estimate", "interval", "n"}},
 "h9_grid": [{"join", "admission", "thresholds", "evidence", "boxes": n, "expected_true": {"estimate", "lb", "ub"},
              "status": "evaluated|not_evaluated", "uncovered_boxes": n}],
 "hypotheses": {"<H id>": {"verdict": "supported|falsified|inconclusive|bounded|not_evaluated|descriptive|reported",
                           "rule": str, "estimator": str, "n": int | null, "estimate": float | null,
                           "interval": [lo, hi] | null, "parts": {...}, "under_both_unsure": {...} | null,
                           "why": str}},
 "d18_inputs": [{"stage", "stratum", "kind": "uninformative_label_space|target_rejected|other_predicted_target",
                 "fn_lb": float, "recoverable_lb": float, "source_taxa": [taxon], "relative_of_prediction": bool}],
 "stage_ranking": [{"stage", "recoverable_lb"}],
 "h11": {...} | null,
 "stop": null | {"rule": str, "at": "F8"}}
```

- `rule` is the prereg text of the hypothesis, copied.
- `valid` is false when `calibration_overlap` is non-empty (R12). Nothing may cite an invalid audit.
- `audit_v1.md` prints every hypothesis with its verdict, estimator, n and interval, in the same prominence whichever way it came out [§6].

### 4.17 Claims and the DA

**`claims.json`** (lab: `<campaign dir>/claims.json`, where campaign dir = `inc_autopilot.campaign.Paths().campaign_dir`; evidence name `campaign/claims.json`; `funnel-claims/1`):
```
{"format", "claims": [{"id": "C1", "text", "polarity": "scarcity|negative|positive", "scope": str,
                       "made_by": "human-transcribed|card:<id>|tier2:<model>|human:<actor>",
                       "cites": [cite], "status": "open|challenged|tested_survives|refuted|accepted_open",
                       "history": [{"utc", "from", "to", "by", "reason", "cites": [cite]}]}]}
```
A cite is the `inc_autopilot.model.cite` form.

**`prospective_da.json`** (written on the lab by G-autopilot, pushed to the cluster; `funnel-prospective-da/1`):
```
{header..., "claim_ids": [...], "digest_sha256", "blind_check": {"markers": [...], "found": []},
 "model": {"role": "adversary", "resolved": str, "family": str, "planner_resolved": str,
           "planner_family": str, "same_family": bool},
 "reply": {...inc-da-reply/1...}, "validation": {...}, "stage_forecast": {stage id: float},
 "committed_utc": str}
```
It must exist before `estimate` runs; `estimate` refuses without it. Its absence is a stop, not a skip [§10 F2].

### 4.18 Recovery (F9–F10; G-labels, read by G-loop)

`INC_DIR/step1_r1/`:
- `labels_overlay/<slug>/<key>.<sha16>.txt`: content-addressed like verify's labels, written once.
- `images_masked/<slug>/<key>.<sha16>.png`: EXIF-transposed and lossless. Every masked box rectangle (pixel box rounded outward) is filled with the image's mean RGB.
- `recovered_pool.jsonl`, one row per recovered image: the `C.MANIFEST_KEYS` plus
  `"pool": "VETO|AUTH|CLASS|JUDGE|FETCH", "policy": "R-V|R-A|R-C|R-T|R-J|R-F", "strata": [stratum id], "unmasked_image", "unmasked_sha256", "masked_boxes": [[cls, cx, cy, w, h]], "ctl_label", "ctl_label_sha256", "provenance_group", "lab", "licence", "near_dup3", "dhash_unmasked", "dhash_masked"`.
  `key` is the pool key. `image` is the masked PNG when a box was masked, else the pool image. `ctl_label` is the step1 label (the join labels), for the control arms.
- `domain_dev.jsonl` and `domain_dev.json`: H10d hold-out (§9, item 13).
- `recovery.json` (`funnel-recovery/1`):
  ```
  {header..., "status": "complete|refused", "refusals": [...],
   "gates": {"<stratum id>": {"policy", "level", "n_labelled", "precision": {"estimate", "lb", "ub", "rogan_gladen"},
                              "box_gate": bool, "class_gate": bool | null, "hypotheses": {H: verdict}, "passed": bool}},
   "class_maps": [{"source", "src_id", "map_to", "accepted": bool, "why"}],
   "identity_checks": {...}, "quarantined_sources": [slug],
   "guards": {"never_train": {"unmasked_checked", "masked_checked", "hits": 0, "unhashable": 0},
              "h6": {"unmasked_checked", "masked_checked", "copies": 0}},
   "counts": {"<pool>": {"images", "boxes_by_class", "masked_boxes"}},
   "licences": {slug: str}, "provenance_groups": {group: [slug]},
   "source_labels_unchanged": {"checked": n, "changed": 0},
   "recovered_pool": {"path", "sha256"}, "domain_dev": {"path", "sha256"}}
  ```
- `arms/U.jsonl`, `arms/U_ctl.jsonl`, `arms/CLASS_ctl.jsonl`, `arms/JUDGE_ctl.jsonl` and `arms/arms.json` (`funnel-arms/1`): `{header..., "realloop": {"exp", "exp_sha256"}, "arms": {"<arm>": {"path", "sha256", "images", "from_base", "from_recovered", "by_pool": {...}, "seed_text"}}}`.

### 4.19 `panel.json` (F10; G-autopilot, lab)

```
{"format": "funnel-panel/1", "built_utc", "inputs": {exp: {"dev_scores": {run_id: sha256}}},
 "arms": {"B": {"exps": ["base_b_v1", "rv2_B_extra"], "seeds": [0, 1, 2, 3, 4]}, "U": {...}, ...},
 "comparisons": [{"id": "H10a|H10b|H10c-CLASS|H10c-JUDGE", "with": arm, "without": arm,
                  "truth_detail_3v3": {...gate.truth_detail on seeds 0-2...},
                  "perm_5v5": {"p": float, "observed_diff": float, "n_splits": 252} | null}],
 "exams_read": ["dev"]}
```

---

## 5. Modules

Each entry gives the public functions (signatures), the CLI verb, the inputs and outputs (by the formats of §4), the errors, the contract sections, and the tests the module must have. Private helpers are free.

### 5.1 G-stats

#### 5.1.1 `__init__.py`

```
FUNNEL_DIR = C.INC_DIR / "funnel";  STEP1_DIR = C.INC_DIR / "step1";  R1_DIR = C.INC_DIR / "step1_r1"
FORMATS = {"census": "funnel-census/1", "ledger": "funnel-ledger/1", ...}      # every id of §4
VOLATILE_KEYS = ("built_utc", "seconds", "hostname", "slurm_job_id")
class FunnelError(RuntimeError) ... (every class of §1.4)
def utc() -> str
def canonical_json(obj) -> str
def write_json_atomic(path, obj) -> str            # returns sha256
def write_jsonl_atomic(path, rows) -> str
def write_csv_atomic(path, header, rows) -> str
def read_json(path) -> obj                         # FunnelError on a missing or malformed file
def file_record(path) -> {"path", "sha256", "bytes"}
def check_records(records: dict) -> None           # raises StaleInput naming the first changed file
def seed(text) -> int                              # C.stable_int(text)
def rng(text) -> numpy Generator                   # numpy imported inside
def code_record(*modules) -> {relpath: sha256}
def header(kind, domain, prereg, inputs, seeds=None, modules=(), testing=False) -> dict
def strip_volatile(obj) -> obj
```

Tests (`tests/test_funnel_init.py`): atomic writes leave no `.tmp` behind on success and keep the old file on an exception; `header` holds every §1.2 key; `strip_volatile` removes only `VOLATILE_KEYS`; the module imports without numpy (run with numpy blocked in `sys.modules`).

#### 5.1.2 `domain.py`

```
SCHEMA_VERSION = "funnel-domain/1"
def validate(raw: dict) -> list[str]
def load(name_or_path) -> Domain          # "weed" -> FUN/domains/weed.json
class Domain:
    name, path, sha256, raw, adapter
    targets -> list[dict];  other -> dict;  class_names -> list[str];  target_ids -> list[int]
    def target(name_or_id) -> dict
    def options() -> list[{"n", "text", "kind": "target|attractor|tail", "class": id|None, "attractor": id|None,
                           "rank", "taxon"}]
    def pair_options() -> list[...]
    def stage(stage_id) -> dict;  stages -> list[dict];  recoverable_stages() -> list[str]
    def lab_of(source) -> str;  def lab_groups() -> dict
    def kt(kt_id) -> dict;  def qualify_rl_on() -> list[str]
    def exam_splits() -> {"decision": str, "non_decision": tuple, "extra_non_decision": tuple}
    def non_dev_exams() -> tuple              # non_decision + extra_non_decision
    def terms() -> {"substring": [str], "token": [str]}      # for the grep test (§7.5)
def load_prereg(path) -> Prereg           # validates format "funnel-prereg/1", the contract sha, the domain
class Prereg: raw, path, sha256, core_sha256, domain_name, groups, amendments, sample_lock -> dict | None
def prereg_core_sha256(obj) -> str
def append_amendment(path, amendment: dict) -> dict
def contract_path(prereg) -> Path
```

The schema checks (every one refuses):
- a guard stage with `recoverable` true (mutation point M5, §7.6);
- target ids that are not 0..n−1 in order, or an `other.id` that collides with a target id;
- a `known_truth` entry with `claimed_by` set and `independent` true;
- `qualify_rl_on` holding a set whose `claimed_by` is set (the circularity rule, [§4.1]);
- an exam named in both `decision` and `non_decision`;
- a `domain_terms` entry equal to an authority or provider kind;
- a `judges.panel` entry of an unknown kind;
- a board whose `targets_from` or `attractors_from` is not a KT id;
- `prereg.domain` differing from the config's `domain`.

Tests (`tests/test_funnel_domain.py`): `weed.json` validates; each check above refuses a copy edited to break it; `exam_splits` equals `C.EVAL_SPLITS` split on dev, plus `domain_dev`; `load_prereg` refuses an edited contract file; `core_sha256` is unchanged by `append_amendment`, which refuses an edit outside `amendments`; `options()` are numbered 1..n in the order of §4.10; a synthetic vehicles config (the R13 one) loads.

#### 5.1.3 `ledger.py`

```
FORMAT = "funnel-ledger/1";  STAGE_KEYS = (...);  ROLES = ("discovery", "cap", "registry", "read", "dedup", "guard",
    "size", "join", "name_status", "reject_class_sample", "target_check", "other_check", "image_rule", "selection",
    "evidence", "relevance", "gate")
def new(domain, inputs: dict, derivation: str, name_status_version: str, identity_inputs: dict) -> dict
def add_stage(ledger, stage: dict) -> None           # checks keys, role, unit, depends_on known
def validate(ledger) -> list[str]
def load(path) -> dict                               # LedgerError listing validate() problems
def write(path, ledger) -> str
def fingerprint(identity_inputs: dict) -> str
def stage_by_role(ledger, role) -> list[dict]
def unaudited_dependencies(ledger) -> list[tuple[str, str]]     # (stage, dependency) pairs, for S5
def recoverable_stages(ledger) -> list[str]           # recoverable is True
def attach_audit(ledger, audit_path) -> dict          # returns a new ledger with stage.audit filled
UNIT_ROW_KEYS = {...}
def validate_unit_row(row) -> list[str]
def iter_units(path, unit=None) -> iterator[dict]     # streaming reader of ledger.jsonl
```

`validate` also checks:
- `in = kept + Σ discarded` wherever all three are numbers;
- `kept_by_label` sums to `kept`;
- no guard stage is recoverable;
- every `depends_on` names an earlier stage in the list.

Tests (`tests/test_funnel_ledger.py`): a hand-built ledger validates; each rule above fails on a broken copy; `unaudited_dependencies` on the §4.3 stage table returns (S12, S8) among its pairs; `attach_audit` records the audit sha; `iter_units` streams a 100k-row file without loading it whole (check peak `tracemalloc` under 50 MB).

#### 5.1.4 `strata.py`

```
def allocate(sizes: dict, n: int, floor: int = 5) -> dict
def frames(prereg, domain, census_dir, adapter, judge_qual=None, dinov2=None) -> dict   # in memory
def write_frames(frames, out_dir) -> dict                                         # frames_v1.json + csvs
def visual_clusters(X, seed_text, k=8) -> numpy int array                          # sklearn KMeans, n_init=10
def g4_bands(scores) -> (edges, bands)
def stratum_id(group, **keys) -> str                                              # key order of §4.8
def load_frames(path) -> dict                                                     # checks every csv sha256
```

Inputs:
- `census_v1.json`, `ledger.jsonl` and `name_status_v2.json` (whose sha256 must equal census's record);
- `guard_pairs_v1.csv`;
- `relation_geometry_v1.json` (H3a alignment, for KT6 membership);
- `judge_qualification.json` (for `allowed_judges`);
- `emb_dinov2/` (for the visual clusters);
- the adapter's `known_truth`.

G3 visual clusters are fixed here, before any label exists [§5.2]. Seed text `funnel/v1/G3/kmeans/<slug>|<src_id>`.

Tests (`tests/test_funnel_strata.py`): `allocate` sums to n, respects the floor and the caps, and breaks ties by id; every group's definition on a synthetic ledger (one unit per rule edge: a small box excluded, a verified box in a vetoed image in G2v, a named-other conflict in G1 named, a no-name class split into 8 clusters with the under-100 ones listed and not sampled); the frames refuse a `name_status_v2.json` whose sha256 differs from census's; the G4 bands are tertiles; frames are deterministic.

#### 5.1.5 `draw.py` (verb `draw`)

```
def draw(prereg_path, out_dir, adapter, force=False) -> dict      # writes sample_v1.csv, key, frames; locks
def inclusion(group_design, ...) -> float
def recompute(sample_row, frames) -> dict                          # re-derives the draw of one stratum from seed text
def load_sample(path, prereg) -> list[dict]                        # checks the lock's sample_sha256
```

- **Inputs:** the frames (built by `strata.frames` inside `draw`); `prereg_v1.json` (groups, planned n, minimum n).
- **Outputs:** `frames_v1.json`, `frames_v1/*.csv`, `sample_v1.csv`, `sample_v1_key.jsonl`, and the sample-lock amendment.
- **Errors:** `DrawError` when a group's frame is below its minimum (listed, and the draw still refuses: a shortfall is a decision for a person [§10 F7]); `SampleLocked` on a rerun with other inputs.
- **Contract:** [§5.2], [§5.3], F6.

Tests (`tests/test_funnel_draw.py`):
- every row's seed text regenerates its stratum's draw exactly (`recompute`);
- π is n_h/N_h for SRS rows, the union formula for G4 and for G2v ∩ G2a, and the two-stage product for G3;
- the HT sum of 1/π over each stratum's sample has expectation N_h (checked over 200 seeded redraws, within 3 standard errors);
- G0 rows show no source or truth, and the key holds the planted share;
- sentinels are 3 per 15 sheet slots;
- the lock amendment carries the sha256 of `sample_v1.csv`, the key, the frames and `name_status_v2`, and the confirmatory frame sizes;
- a second draw with the same inputs is a no-op, and with a changed ledger it refuses (`SampleLocked`).

#### 5.1.6 `estimate.py` (verb `estimate`)

The only producer of audit numbers [§5.4, §8.7].

```
Z975 = NormalDist().inv_cdf(0.975)
def beta_ppf(q, a, b) -> float                  # regularised incomplete beta (Lentz), bisection to 1e-12
def binom_interval(k, n, conf=0.95) -> (lo, hi) # Wilson if n >= 30 else Jeffreys; lo is the one-sided 97.5 % bound
def clopper_pearson(x, n, conf=0.95) -> (lo, hi)    # non-integer x, n allowed (Korn-Graubard)
def stratified(strata: list[{"N", "n", "k"}], fpc=True) -> {"estimate", "interval", "var", "n_eff"}
def ht(y, pi, N) -> {"estimate", "interval", "var", "n_eff"}
def ppi_pp(y_lab, f_lab, f_all, pi_lab=None) -> {"estimate", "interval", "lambda"}
def rogan_gladen(k, n, se_k, se_n, sp_k, sp_n, seed_text, draws=20000) -> {"estimate", "interval", "flag"}
def cluster_bootstrap(y, clusters, seed_text, B=2000) -> {"deff", "interval"}
def webber_recall(admitted: {"N", "k", "n"}, strata: list[{"N", "n", "k"}], seed_text, draws=20000) -> {...}
def holm(pvalues: dict, alpha=0.025) -> dict
def perm_test_one_sided(a: list, b: list) -> {"p", "observed_diff", "n_splits"}    # exact, all splits
def evaluate(prereg_path, funnel_dir, adapter) -> dict                             # builds audit_v1.json
def hypothesis(hid, ctx) -> dict                                                   # one per H id, below
def render_md(audit) -> str
```

**Estimators** [§5.4], as pinned here:
- **Per stratum:** `binom_interval`.
- **Aggregate (`stratified`):** θ = Σ W_h p_h with W_h = N_h/N. v = Σ W_h² (1 − n_h/N_h) p_h(1 − p_h)/(n_h − 1), where n_h = 1 gives v_h = W_h² × 0.25. n_eff = θ(1 − θ)/v, capped at Σ n_h (Σ n_h when v = 0). Interval = `clopper_pearson(n_eff θ, n_eff)`. Recorded as `korn_graubard_no_df_adjustment`.
- **HT:** θ = Σ y_i/π_i / N. v = Σ (1 − π_i) y_i²/π_i² / N². Interval as in the aggregate.
- **PPI++:** λ = Cov(y, f)/((1 + n/N) Var(f)), clipped to [0, 1]; θ = λ mean_all(f) + mean_lab(y − λ f); normal interval. Used in a stratum only when n_labelled ≥ 30 and its half-width is smaller than the design interval. f is the allowed judge with the largest `precision_at_half.lb + rescue.lb` for the stratum type (ties by judge id).
- **Rogan–Gladen (melded):** 20,000 draws with seed `funnel/v1/rg/<stratum id>`. Lower bound: θ_obs ~ Beta(k, n − k + 1), Se ~ Beta(x_se + 1, n_se − x_se), Sp ~ Beta(x_sp, n_sp − x_sp + 1). Upper bound: the mirror draws. θ = (θ_obs + Sp − 1)/(Se + Sp − 1), clipped to [0, 1]; the 2.5 % and 97.5 % quantiles. When Se + Sp − 1 < 0.7 (point estimates), the uncorrected estimate is reported with `flag: "se_sp_low"`.
- **Cluster bootstrap:** by image; by source for aggregates that span sources; seed `funnel/v1/boot/<stratum id>`.
- **Webber recall:** A ~ admitted N × Beta(k_adm + 1, n_adm − k_adm + 1). R_h = k_h + BetaBinomial(N_h − n_h, k_h + 1, n_h − k_h + 1). Recall = A/(A + Σ R_h); 2.5 % and 97.5 % quantiles; seed `funnel/v1/recall`. The admitted precision comes from the reference-lab in-domain calibration (`calibration.json` counts) for reference-lab sources and from G2a for the rest [§5.4 review]. It is never assumed to be 1.
- **Screening:** per recoverable stage, p = P(Binom(n_eff, 0.10) ≥ n_eff θ), computed as `1 − betainc(...)`; Holm at 0.025 over those stages. A stage is suspect when it is rejected and R̂.lb ≥ R_min. R_min = 500 at stage level; at (stage, class) level it is 100 for a class with fewer than `small_class_train_boxes_below` train_core boxes (`census_v1.train_core_boxes_per_class`).

**Stages and frames.** How each recoverable stage is estimated, and which `d18_inputs.kind` its strata carry:

| Stage | Frames | FN event | `d18_inputs.kind` |
|---|---|---|---|
| S3 | – (exact, `census_v1.h5a`) | dropped twin with a target box the kept twin lacks | – |
| S4, S5 | G5 | not recoverable; H5b only | – |
| S7, S7b | G3 | class or cluster unit truly target-bearing | `uninformative_label_space` |
| S8 | G2 | a rejected target-labelled box is a correct, correctly boxed target | `target_rejected` |
| S9 | G1 (and G4 for S9's kept side, H4) | an OtherPlant box called a target is truly that target | `other_predicted_target` |
| S10 | G2v | a vetoed verified box is a correct target | `target_rejected` |
| S12 | – | derived; recomputed from the recovered counts after F9 [§7] | – |
| S0 | – | exact image count from `card_image_counts` | – |

**Events.** A labelled item's y for a box stratum is "the answer is a target, at the level used, and `box_ok` is yes" (the FN definition [§5.1]). For H1 it is "the answer equals the source label and `box_ok` is yes". Per stratum, `unsure` and `unparsed` answers are computed both ways, as 0 and as 1. A verdict is `supported` (or `falsified`) only when it holds under both, and `inconclusive` otherwise. A genus answer in a `genus_answer_unsure_for` genus is unsure [§5.1].

**Level.** Each hypothesis uses its primary RL's level from `rl_qualification.json`. When the needed level is not qualified in the stratum's scope, the verdict is `bounded` [§10 stop rules].

**Hypotheses** [§6] (the rule texts are copied from the prereg into `rule`):

| H | Inputs | Verdict |
|---|---|---|
| H0 (a) | G0 target share, RG-corrected at species level, 95 % two-sided interval; the planted share from the key | supported if the interval covers the planted share, falsified otherwise; falsified → `stop` = "H0 falsified" |
| H0 (b) | `relation_audit_v1.json h0b.pass` | exact |
| H0 (c) | `h0c` | reported; gates the visual proposals |
| H1-pre | `relation_geometry_v1.json h1_pre` | exact; a fail is a join error for L11 |
| H1 | G2 strata of the authoritative sources; also per label, and separately "label right" and "box valid" | supported: lb ≥ 0.85 overall and for Ragweed; falsified: ub < 0.85 overall |
| H2a / H2b | G1 `noinfo` / `named` frames (v2) | as the prereg; the role sub-stratum reported apart |
| H3a | `h3a_exact` plus G3 purity of the aligned ids 12 and 5 (answer = Sicklepod; answer in *Ipomoea* at genus) | as the prereg |
| H3b | G3 units of anonymous sources other than mh_weed16 with ≥ 100 boxes | supported: some unit lb ≥ 0.5; falsified: every ub < 0.3 |
| H4 | G4 HT per frame | supported: named ub < 0.02 and noinfo ub < 0.10; falsified: named lb ≥ 0.02 or noinfo lb ≥ 0.10 |
| H5a | `census_v1.h5a` | supported (the guard is not a material FN source) if < 100 boxes, falsified otherwise; exact |
| H5b | G5, answer ∈ {same, consecutive}, RL-B only | supported: lb ≥ 0.8; falsified: ub < 0.8; `not_evaluated` if RL-B is not qualified on pair sentinels (DEC-2) |
| H6 | `leak_v1.json` | `reported`, with the routing |
| H7 | `judge_qualification.json h7` | parts per judge, each supported or falsified |
| H8 | – | `not_evaluated` (F11) |
| H9 | the 24-cell grid | `descriptive` |
| H9′ | best cell's lb and ub of expected true targets | supported: lb ≥ 4,098; falsified: ub < 3,073.5; shift-threshold cells `not_evaluated` unless H1 and H3a are supported |
| H10 | – | not in this file (panel.json, outcome.py) |
| H11 | `prospective_da.json stage_forecast`, `stage_ranking` | supported: the top-1 forecast stage is the stage with the largest R̂.lb and the log-loss beats uniform over the recoverable stages |
| H12 | `census_v1.h12` | supported: lb ≥ 0.70; falsified: ub < 0.70 |

**H9 grid** [§6 H9].
- *Axes:* join ∈ {strict, card, card_taxonomy}; admission ∈ {image, box}; thresholds ∈ {in_domain, shift}; evidence ∈ {on, off}.
- *Recomputation.* A cell recomputes admission from `ledger.jsonl` (`pred`, `p` and `cos` are enough for any label map and thresholds, since verification needs argmax = label).
- *Precision.* Each admitted box takes the precision of the audit stratum whose frame holds it. A box no sampled stratum covers counts toward `boxes` with precision 0 in the lower bound (`uncovered_boxes`).
- *Shift thresholds.* Learn-then-Test per class, at level 0.95 with δ = 0.025, on the known truth outside the stratum's lab. They are computed in `estimate` from the gold-labelled G2 items and KT7 [§6 H9], and only when H1 and H3a are supported.

**Errors.** `EstimateError`; also a refusal when:
- `rl_qualification.json` or `prospective_da.json` is missing;
- the sample lock does not match;
- a stratum uses a judge or a sentinel scope that shares material with it. In that case `valid` is false with `calibration_overlap` filled, and the file is still written (R12).

Tests (`tests/test_funnel_estimate.py`):
- `beta_ppf` against closed forms (Beta(1, 1), Beta(a, 1)) and, when scipy is present, `scipy.stats.beta.ppf` to 1e-9;
- `binom_interval` against the [§5.3] table (0 of n upper bounds, Wilson half-widths) to 3 decimals;
- the coverage of `stratified` on a simulated population (95 % nominal, observed ≥ 0.93 over 400 seeded draws);
- HT unbiasedness on the dual design;
- PPI++ never wider than the design interval once it is chosen;
- Rogan–Gladen returns θ_obs when Se = Sp = 1, and flags when Se + Sp − 1 < 0.7;
- `perm_test_one_sided([3 values], [3 values])` has minimum p 1/20, and at 5 v 5 the minimum is 1/252 [§6 H10a];
- `holm` on a textbook example;
- every hypothesis evaluator on a synthetic gold where the truth is set so each verdict (supported, falsified, inconclusive, bounded) occurs once;
- the both-assignments rule for unsure answers;
- H0 falsified sets `stop`;
- a judge sharing a lab with a stratum makes `valid` false (R12 input);
- determinism of the whole audit.

#### 5.1.7 `claims.py`

```
FORMAT = "funnel-claims/1"
POLARITIES = ("scarcity", "negative", "positive")
STATUSES = ("open", "challenged", "tested_survives", "refuted", "accepted_open")
TRANSITIONS = {("open", "challenged"): ("adversary", "person"),
               ("challenged", "tested_survives"): ("autopilot", "person"),
               ("challenged", "refuted"): ("person",), ("challenged", "accepted_open"): ("person",),
               ("open", "accepted_open"): ("person",), ("tested_survives", "challenged"): ("adversary", "person")}
def load(path) -> dict;  def save(path, claims) -> str;  def validate(claims) -> list[str]
def file_claim(claims, text, polarity, scope, made_by, cites) -> str          # returns the new id
def transition(claims, claim_id, to, by, reason, cites, actor_kind) -> dict  # ClaimsError when not allowed
def open_negative(claims) -> list[dict]                                       # open or challenged, polarity scarcity|negative
def main(argv=None)                                                           # list | file | transition
```

- `actor_kind` is `adversary` (a validated DA reply with ≥ 1 surviving counter-argument, from a model not of the planner's family), `autopilot` (D18 silent on a valid audit with a matching fingerprint; the cite must name the audit's sha256) or `person` (card X12).
- A same-family adversary is refused (R14).
- A COMPLETE card may not call a negative claim concluded while it is `challenged` [§8.6]; `campaign.py` enforces this through `claims.open_negative`.

Tests (`tests/test_funnel_claims.py`): every allowed transition works and every other refuses; history is append-only; C1 and C2 as filed in §5.5.10 validate; a same-family adversary cannot move a claim; the module imports with the standard library only.

### 5.2 G-data

#### 5.2.1 `adapters/__init__.py`

```
INTERFACE = ("census", "ledger_from_summaries", "crop_table", "known_truth", "unit_keys", "guard_pairs",
             "pool_rows", "base_rows", "increment_rows", "eval_rows", "never_train_guard", "text_encoder",
             "step1_features", "j1_scores", "bioclip_embedder", "label_rows", "name_status_v1")
def load(name) -> module        # refuses a module lacking any INTERFACE function (AdapterError)
```

Tests: `load("inc_step1")` passes the interface check; a stub module missing one function is refused.

#### 5.2.2 `adapters/inc_step1.py`

```
def census(prereg, domain, out_dir, taxonomy_cache, known_items=None, step1_dir=None, force=False) -> dict
def ledger_from_summaries(domain, step1_dir, census_v0, out_path, fit_info_projection=None) -> dict
def crop_table() -> verify.Crops                     # step1/crops.csv, with check_fresh
def known_truth(domain, funnel_dir) -> {KT id: [item]}    # §4.6
def unit_keys(unit_ids) -> {unit_id: {"source", "lab", "near_dup3", "provenance"}}
def guard_pairs(out_path) -> dict                    # guard_pairs_v1.csv
def pool_rows(sources=None) -> list[manifest row];  def base_rows() -> list;  def increment_rows(exp) -> {step: list}
def eval_rows() -> {split: list}                     # C.manifest_path(split) for C.EVAL_SPLITS
def never_train_guard() -> C.NeverTrainGuard
def text_encoder() -> relevance.BioclipTextEncoder   # the J-zs text tower
def step1_features() -> (X, info)                     # verify.load_embeddings(crop_table())
def j1_scores(X) -> (P, cos)                          # Verifier.load().scores(norm(X))
def bioclip_embedder() -> verify.BioclipEmbedder
def label_rows(keys) -> {key: [(cls, cx, cy, w, h)]}  # the step1 label files, sha-checked against pool.jsonl
def name_status_v1(name) -> str                       # verify.other_name_status with numeric split out
def verifier_fit_info_projection(step1_dir) -> dict   # {"other_sample": {...fit_info other_sample...}}
```

**Census inputs** (cluster): `step1/{pool.jsonl, pool_meta.jsonl, pool_summary.json, cwd12_copies.jsonl, crops.csv, crops_skipped.csv, crops_info.json, emb/, verifier/{probe.joblib, verifier.npz, thresholds.json, fit_info.json}, calibration.json, verified.jsonl, conflicts.csv, admit_summary.json, pool_verdicts.npz, select_summary.json, base_B.jsonl, increment_pool.jsonl, select_clusters.csv, cache/dhash/*.json, cache/train_core_probe.json}`, the registry and flags, the never-train index, `funnel/taxonomy_cache.json`, and `funnel/known_items_v1.json` (optional).

**Census outputs:** `census_v1.json`, `ledger.jsonl`, `funnel_ledger.json` (derivation census), `name_status_v2.json`, `guard_pairs_v1.csv`.

**How it works.**
- It re-derives P [n, 13] and cos for the pool crops with the fitted verifier. For the OtherPlant training crops it uses the out-of-fold arrays. It then checks `verify.verdicts` against `pool_verdicts.npz`.
- The dropped images of S2–S5 come from `cache/dhash/<slug>.json`, the never-train index, `cwd12_copies.jsonl` and pool_meta dHash equality. They are listed by the same directory reads verify makes (`_resolve_dir`, `_layout`, one `os.listdir` per directory, never a walk). An exact_dup twin's labels are read with `verify.read_source_label` and joined with `verify.class_join`, for H5a.
- `s8b_reject_class` comes from `verifier.npz other_ids` and `fit_info.json other_sample` [§3.1 S8b, F3].
- `s1_skipped` copies the registry entry of each skipped slug [F3].

**Errors:** `AdapterError`; `StaleInput` (a Step 1 file changed after `verify admit`, through `verify.check_fresh`); `TaxonomyError` when a census name has no cache entry. That message names lever L12: "no resolution for <n> name(s) in taxonomy_cache.json: run fetch --what taxonomy (lever L12)".

Tests (`tests/test_funnel_adapter_step1.py`, on `tests/funnel_world.py`):
- census reconciles exactly with the world's summaries;
- the J1 re-derivation reproduces `pool_verdicts.npz`, and a flipped verdict in the npz is refused;
- every box and pre-pool image has one ledger row, with correct `failed_stages`, `first_cause` and `sole_cause` on planted cases (a box vetoed in an image of a non-evidenced source has two failed stages and no sole cause);
- the veto blockers on planted images;
- H5a on a planted exact_dup twin that carries a target box the kept twin lacks;
- `ledger_from_summaries` on the real local `step1/*` and `census_v0.json` gives S2 = 220,104 / 545,318 and target_check kept 2,049, discarded conflict 2,476 and unknown 2,756, and other_check conflict 6,256. These values are read in the test from `census_v0.json` through `name_status_v1` and from `admit_summary.json`, then compared with the ledger;
- `known_truth` KT2 excludes unmatched boxes;
- `unit_keys` gives a copy and its twin the same provenance;
- a missing taxonomy entry refuses with the L12 message.

#### 5.2.3 `names.py` and `taxonomy.py`

```
# names.py
def key(name) -> str                                  # re.sub(r"[^a-z0-9]", "", name.lower())
def status_v2(name, joined_target: bool, domain, resolver) -> {"status", "via", "taxon", "rank"}
def build_name_status(rows: [{"source", "src_id", "name", "joined_target", "status_v1", "boxes", "conflicts"}],
                      domain, resolver, contract_check=None) -> dict      # name_status_v2.json content
def frame_of(status, domain) -> "noinfo" | "named" | "excluded" | "target"

# taxonomy.py
class Resolver:
    def __init__(self, cache: dict, domain, transport=None, allow_network=False)
    def resolve(self, name) -> dict | None            # the "resolved" entry; TaxonomyError on a miss offline
    def is_target_synonym(self, res, target) -> bool  # accepted key = target key, or target in lineage below its rank
    def is_relative(self, res, target) -> bool        # same genus, or in the target's "not" list
    def lineage_string(self, taxon) -> str            # "Plantae Tracheophyta ... Genus species" for J-zs
def build_cache(names, domain, transport, out_path) -> dict     # lab only; network through `transport`
def load_cache(path, domain) -> dict                   # checks the authority kind and the overrides sha256
```

- **Scientific resolution:** `match_url` with `name=<name>&strict=true`. It is accepted when the match type is EXACT (or it is an override) and the rank is at or below genus.
- **Vernacular resolution:** `search_url` with `q=<name>&qField=VERNACULAR&limit=20`. It is informative when any result is in the kingdom of the targets. It is never mappable [§4.3 J-taxon].
- **Overrides** win (for example *Eclipta alba* → *Eclipta prostrata*) and are recorded.
- **Authority version:** from `version_url`, stored in the cache. The exact query parameter names are checked by G-data against the authority's current documentation when the fetch is built. This runner has not verified them.

Tests (`tests/test_funnel_names.py`, `tests/test_funnel_taxonomy.py`):
- the rule order on one name per status;
- on the real local `census_v0.json` with a fixture cache in which only the names listed in the fixture resolve, "BroWeed" and "NarWeed" are `unresolvable` with 197,613 boxes, and "crop"/"Crop" are `role` with 30,042 boxes (both read from census_v0 by name in the test). The non_object and state totals are printed next to the contract's 2,149 and 1,403 and written to `contract_check`; they are not asserted;
- the resolver is offline by default and refuses a miss; the fake transport records the query URLs; overrides win; a vernacular-only match is informative but not mappable;
- `is_target_synonym` accepts a synonym and a subspecies and refuses a sibling species.

#### 5.2.4 `relation.py` (verb `map`)

```
def geometry_match(pool_labels: {image key: [(cls, cx, cy, w, h, W, H)]}, upstream: {file: [(id, x0, y0, x1, y1, W, H)]},
                   tol_px=2) -> {"image_pairs": {...}, "box_pairs": [...]}
def alignments(n_upstream_classes) -> {name: {upstream id: agml id | None}}   # identity, plus1, minus1, drop<d>
def choose_alignment(box_pairs, alignments, seed_text) -> dict                # best on half a, confirmed on half b
def read_upstream(archive_path, kind) -> {file: [...]}                        # Pascal VOC XML or YOLO text in a zip
def relation_scores(unit_crop_ids: {unit: [crop id]}, P, labels) -> {unit: {class: float}}
def relation_audit(units, scores, joins, truth=None, map_min=0.5, flag_max=0.2, min_boxes=20) -> dict
def card_proposals(domain, geometry, pool_summary) -> list                    # class_maps.json proposals
def run_geometry(prereg, domain, funnel_dir, adapter) -> dict                 # relation_geometry_v1.json + class_maps.json
def run_relation(prereg, domain, funnel_dir, adapter) -> dict                 # relation_audit_v1.json
```

- **Relation score** r(u, k) = the mean over unit u's crops of the judge's P(k).
- **Map** u → k when k is the argmax, r(u, k) ≥ `map_min` and the unit has ≥ `min_boxes` boxes.
- **Flag a join** when r(u, joined class) < `flag_max` and another class has r ≥ `map_min`.
- **Judges.** H0(b) uses J-knn1 (leave-session-out, the reference domain, a pipeline check [§6 H0 review]). H0(c) and the visual proposals use the allowed judge with the best `precision_at_half.lb` for `other_noinfo` strata.
- **H0(b) units:** the classes of cottonweed_holdout and of the three_season 2021 copy with ≥ 20 boxes, each under the current join and under the old join (`calibration.json` `old_wrong_joins`). It passes when every unit maps to its twin's species, cottonweed_holdout's old join is flagged, and cottonweed_sp8's current join is not.
- **Geometry.** H3a is mh_weed16 against Mendeley d3n3mgjjbv v2. H1-pre is weed_crop and greenhouse against their Mendeley archives. The half split uses seed `funnel/v1/h3a/half`, by image key.

Tests (`tests/test_funnel_relation.py`):
- the geometry match on synthetic upstream XML with a planted "drop class 5" export: the chosen alignment is `drop5`, identity and ±1 fail, and the agreement is confirmed on the other half;
- a 2 px shift still matches and a 5 px shift does not;
- H0(b) on synthetic copies with a planted old join: flagged, and the current join is not;
- `to_L14` when the card's class count disagrees;
- the proposals are immutable (a rerun with other inputs refuses without `--force`).

#### 5.2.5 `fetch.py` (verb `fetch`, lab only)

```
KINDS = ("http", "archive", "inat_observations", "roboflow_classes", "gbif_backbone")
def fetch_cards(domain, out_dir, transport) -> dict                 # cards/ + cards/index.json
def fetch_kt7(domain, out_dir, transport) -> dict                   # kt7/ (DEC-8)
def kt7_roles(items, domain) -> list                                # §4.6
def known_items(sources: list[Path], out_path) -> dict              # known_items_v1.json, hashed
def taxonomy(domain, names_from: Path, out_path, transport) -> dict # taxonomy.build_cache
def refetch(domain, slug, out_dir, transport) -> dict               # R-F: images the cap left out
def write_manifest(out_dir) -> dict                                 # fetch_manifest.json
def check_manifest(out_dir) -> None                                 # on arrival (cluster): StaleInput on a mismatch
```

- **Transport.** `transport(url, params) -> (status, bytes, headers)` is injected. The default uses `urllib` with a 60 s timeout and 3 retries.
- **KT7** (DEC-8): at most 30 observations per taxon, CC licences only. Observation id, licence and photo sha256 are recorded. The query parameters come from `known_truth.kt7.provider` verbatim (seedling or vegetative stage, US and Indian places, research grade).
- **Roboflow class lists** need an API key in the environment and refuse without it.
- Every response body is hashed before it is parsed.

Tests (`tests/test_funnel_fetch.py`): with a fake transport, cards are written with their hashes; an archive's members are listed; KT7 keeps only CC licences, caps at 30 per taxon and assigns roles by the rule; the manifest check detects a changed file; no real network is touched (the test blocks `socket.socket`).

#### 5.2.6 `domains/weed.json`

The content of §3.2. Tests (`tests/test_funnel_weed_config.py`):
- it validates;
- the targets equal `C.CLASS_NAMES[:12]`, and the taxa equal `CWD12_BINOMIAL` except MorningGlory (genus *Ipomoea*);
- the class policy equals `prereg_v1.json class_policy`;
- `lab_groups` equal the prereg's;
- the `names` word lists together cover `verify.GENERIC_NAME_KEYS`, `NON_PLANT_WORDS` and `CWD12_RELATED_TOKENS`, each word in exactly one list;
- `exams.non_decision` equals `C.EVAL_SPLITS` minus dev.

#### 5.2.7 `tests/funnel_world.py` (shared synthetic Step 1 world)

```
def build_world(root: Path, seed=0, with_probe=True, n_images=60) -> World
World: .inc_dir, .step1, .sources (dict), .planted (dict of the planted cases), .never_train, .registry, .images
```

It writes every Step 1 file the census reads, in verify's exact formats, with tiny PNG images (16–64 px), a 13-class `probe.joblib` (when sklearn is present) and `pool_verdicts.npz` consistent with it. It includes the planted cases the tests above name: a vetoed image, an exact_dup twin, a near_eval pair, a copy with a twin, a no-name source, a numeric source, a named-other source and an authoritative source.

### 5.3 G-vision

#### 5.3.1 `embed.py` (part of verb `embed-judges`)

```
MODEL = domain.judges.features.model (default read from the config)
class Dinov2Embedder:                 # semisup_labeler._load_backbone(MODEL) + _embed_batch(("cls",))
    name, dim;  def __call__(pils) -> float32 [n, dim]
def embed_crops(crop_table, out_dir, shard, nshards, embedder=None, procs=5, batch=64, chunk_images=2000) -> dict
def embed_table(csv_path, out_path, embedder) -> dict          # kt7 and refetch tables (CROP_FIELDS)
def embed_images(rows, out_path, embedder, view="processor") -> dict     # whole-image descriptors (leak)
def load(crop_table_sha, emb_dir, nshards=None) -> (X float16, info)     # same checks as verify.load_embeddings
```

- Crops are cut with `verify._cut_task` (verify's square crop, grey padding, 224 px), so a DINOv2 row and a BioCLIP-2 row describe the same pixels.
- Shards resume per chunk, and a failed crop is a NaN row, as in verify.
- In an array job, task i does shard i.

Tests (`tests/test_funnel_embed.py`): with a fake embedder, every crop id is covered exactly once over 3 shards; a crop that cannot be cut is NaN and does not shift a neighbour; a resumed run skips finished chunks; `load` refuses a shard made from another `crops.csv`; with a fake image, the crop fed to the embedder is byte-identical to `verify._cut_task`'s.

#### 5.3.2 `judges.py` (part of verb `embed-judges`)

```
def zero_shot(X, T, scale, groups: list[int], n_labels) -> P          # softmax(scale * cos) summed per label
def prompts(domain, resolver) -> (texts, label_of_text)                # targets by lineage + common name; attractors
                                                                       # and named non-targets -> other; non_object
def knn(Q, q_keys, bank_X, bank_labels, bank_keys, n_labels, k=10, temperature=0.07, exclude="disjoint") -> P
def banks(domain, adapter, features, name_status) -> {judge: {"X", "labels", "keys", "kt", "sha256"}}
def score_all(prereg, domain, funnel_dir, adapter, text_encoder=None) -> dict      # judges/*.npz
```

- **J-zs.** The text encoder comes from `adapter.text_encoder()`, injected. The engine never names the model class (§9, item 24). The features are the Step 1 shards (`adapter.step1_features()`). The prompt template comes from the config.
- **J-knn1.** The bank is KT1. The exclusion for a train_core query is its own session, and nothing for any other query. The label space is the targets only.
- **J-knn2** [§4.3]. The bank is KT1, KT4, KT5 (at most `kt5_per_cell_max` per (source, name), seed `funnel/v1/knn2/kt5/<cell>`), KT6 calibration half and KT7 exemplars. For each query, every bank entry that shares a source, `near_dup3`, provenance or lab with it is excluded (leave-lab-out and more). The label space is the targets plus "other".
- **kNN rule.** Weights exp(cos/τ) over the top k eligible neighbours, normalised; P is the weighted vote.
- **J1 on KT7:** `judges/J1__kt7.npz` from `adapter.j1_scores` on `emb_bioclip_kt7.npz`.

Tests (`tests/test_funnel_judges.py`):
- `zero_shot` sums to 1 per row;
- `knn` never uses an excluded entry: a bank entry from the query's lab placed at cosine 1.0 does not change P (mutation point M7);
- J-knn1 on a train_core query excludes its session;
- the KT5 cap per cell;
- the score file's meta names the bank sha256;
- a J-zs run with a fake text tower is deterministic;
- the engine file contains no model class name (`inspect.getsource` check).

#### 5.3.3 `leak.py` (verb `leak`)

```
FAMILIES = ("flip", "rot90", "crop", "brightness", "blur", "shear", "letterbox640", "jpeg")
def dhash_variants(image) -> {variant: int}           # C.dhash(BytesIO(PNG of the transformed image))
def augment(image, family, rng) -> (image, params)
def calibrate(adapter, domain, embedder, seed_prefix="funnel/v1/leak") -> dict        # positives and negatives
def threshold(positive_cos: {family: array}) -> float  # min over families of the 5th percentile
def detect(images: list[{"key", "path"}], eval_index, calibration) -> list[copy]
def eval_index(adapter, embedder, cache_path) -> object                                # cluster-only npz cache
def run(prereg, domain, funnel_dir, adapter, embedder=None) -> dict                    # leak_v1.json + leak_pairs_v1.csv
```

- **Positives.** Per family, a seeded sample of min(2,000, |train_core|) train_core images, each augmented once with parameters drawn from the family's range (seed `funnel/v1/leak/pos/<family>`). Never an evaluation image.
- **Negatives:**
  - every 7–10-bit dHash pair between the `negative_source_pairs` groups;
  - the hardest pairs: for a seeded sample of 2,000 train_core images (seed `funnel/v1/leak/neg`), the nearest mh_weed16 pool image by cosine.
- **Threshold:** θ = the minimum over families of the 5th percentile of the positive cosines, so every family has recall ≥ 0.95. `ok` requires FPR ≤ 0.01 on both negative sets. Otherwise `LeakCalibrationError`, raised after the file is written with `ok: false` and without any scan [§10 F4].
- **Scan:**
  - every kept pool image (all sources);
  - base B's harvested images;
  - every realloop_v1 increment manifest (`INC_DIR/realloop_v1/manifests/*.jsonl`).
  Each is compared against every evaluation image (`adapter.eval_rows()`): the top 5 by cosine, plus every image within 10 bits under any variant.
- **H6(a) scope:** the sources that kept images and have a near_eval hit in `pool_summary.json per_slug`, computed by that rule, never listed by hand. A source with any copy goes to `quarantine` as a whole [§6 H6 review].
- **H6(c):** the config's lab groups, plus any source with a copy of an evaluation image, joined to the group of that image's split. dev, test, ood22 and ood23 are Lu-lab; imageweeds is NDSU; both come from `lab_groups` membership of the exam datasets.
- **Evaluation pixels never leave the cluster.** Only keys are written to `leak_v1.json`.
- `recover` calls `detect` on masked copies (CPU is allowed there).

Tests (`tests/test_funnel_leak.py`):
- `dhash_variants` of a horizontally flipped image contains the original's dHash under `hflip`;
- each family's augmentation is deterministic under its seed;
- with a fake embedder on synthetic images, a planted augmented copy of an evaluation image is found under every family, and a distinct image at 8 bits is not;
- a calibration whose negatives exceed 1 % FPR refuses and runs no scan;
- the H6(a) scope computed on the real local `pool_summary.json` has 14 sources (read, not typed);
- no evaluation path or pixel array is written to `leak_v1.json`.

### 5.4 G-labels

#### 5.4.1 `sheets.py` (verb `sheets`)

```
def boards(domain, adapter, out_dirs) -> {board_id: dict}
def board_for(item_keys, domain, boards) -> board_id                     # DisjointnessError if none fits
def compose(sample_rows, key_rows, frames, domain) -> list[sheet plan]   # the §4.10 composition rules
def render_sheet(plan, out_dir, adapter) -> dict                          # jpg + json
def render_pair_sheet(plan, out_dir, adapter) -> dict                     # cluster-only directory
def run(prereg, domain, funnel_dir, adapter) -> dict                      # sheets_v1/, sheets_v1_cluster/, key
def person_items(items, proposals, anchoring_share=0.25, blind_share=0.2, seed_text=...) -> list   # L14 verify-only
```

- **Blinding.** No sheet file (image or JSON) holds a source, stratum, verdict, score, proposal or unit id. The test greps the rendered JSON for every source slug and class name of the key.
- **Separation.** G5 items and pair sentinels are rendered only into `sheets_v1_cluster/`. `index.json` of `sheets_v1/` has `contains_eval_pixels: false`, and `run` refuses to write an item with `sheet_class = eval` there.
- **The key** is written to `sheets_v1_key/key.jsonl`, and its sha256 is in both indexes.
- **Anchoring sentinels** (a wrong proposal on a quarter of the sentinels, drawn from J1's attractor pairs) exist only in `person_items`. Machine sheets have no proposals [§4.3].

Tests (`tests/test_funnel_sheets.py`):
- every sheet has 15 items with 3 sentinels, at most 4 G4 items and expected prevalence ≥ 0.2;
- no sheet mixes boards;
- an item that shares a lab with B1's exemplars lands on B2;
- the crop tile equals `verify._cut_task`'s crop;
- blinding (the grep above);
- G5 never appears in `sheets_v1/`;
- the rendering is deterministic (image sha256 equal over two runs);
- `person_items` puts a wrong proposal on a quarter of the sentinels, drawn from the configured attractor pairs.

#### 5.4.2 `rl.py` (verbs `rl-b` and `ingest`)

```
ANSWER_RE = r"^\s*(\d{1,2})\s*[,;]\s*(YES|NO|NA)\b"          # item answers; case-insensitive
SHEET_ANSWER_RE = r"^\s*(\d{1,2})\s*:\s*(\d{1,2})\s*[,;]\s*(YES|NO|NA)\b"
PAIR_RE = r"^\s*(\d)\b"
def parse_item(text, n_options) -> (option, box_ok) | None     # strips <think>...</think> first
def parse_sheet(text, positions, n_options) -> {position: (option, box_ok)}
class OllamaClient:                                             # POST /api/chat, images base64, stream false
    def __init__(self, endpoint, model, transport=None, num_ctx=8192)
    def check_vision(self) -> None                              # /api/show capabilities must hold "vision"; else RLError
    def ask(self, prompt, images: list[bytes], seed) -> str
def run_rl_b(prereg, domain, funnel_dir, client, sheet_dirs, max_attempts=3) -> dict   # rl_answers/RL-B/*.json
def validate_answers(path, sheet_json, backend) -> list[str]
def ingest(prereg, domain, funnel_dir, answer_dirs) -> dict                            # gold_v1.csv
```

- **RL-B requests.** One request per item: images [board, the panel cut from the sheet image at `panel`]. Pair items send the pair panel only. The prompt is `reference_labeller.prompt` with the options. temperature 0; seed `C.stable_int("funnel/v1/rl-b/" + item_id) + attempt`.
- **Retries.** Up to 3 attempts on an HTTP error or a parse failure. After that the status is `unparsed` and `raw` is kept.
- **Model check.** `model_digest` from `/api/tags` is recorded. The model tag must equal `reference_labeller.backends.RL-B.model`.
- **Validation.** An answer file is refused when:
  - its `sheet_sha256` differs from the sheet's JSON;
  - an item is missing or doubled;
  - an option is out of range;
  - the backend is not the directory's;
  - an RL-A file answers a sheet from `sheets_v1_cluster/` (DEC-2);
  - an RL-A labeller's family is in `judged_families`.
- **Ingest** joins the answers with `key.jsonl` (cluster) and writes `gold_v1.csv`. A person's answers (L14, `labeller` "human:<actor>") go through the same validation.

Tests (`tests/test_funnel_rl.py`):
- `parse_item` accepts "7, YES", "7,yes" and "<think>...</think>\n7, NA", and refuses "seven" and "27, YES" when there are 25 options;
- `parse_sheet` maps positions;
- a fake transport answering garbage twice then valid succeeds on attempt 3; three garbage answers are recorded as `unparsed`;
- `check_vision` refuses a model without vision;
- the RL-A validation refusals, each on its own planted file;
- `ingest` gives one gold row per item and labeller, fills the correct-answer columns for sentinels only, and writes nothing for an item missing from the key.

#### 5.4.3 `qualify.py` (verb `qualify`, and `qualify --rl`)

```
TYPES = {"shifted_target": ("G2", "G2v", "G2a"), "other_named": ("G1:named", "G4:named"),
         "other_noinfo": ("G1:noinfo", "G4:noinfo", "G3")}
def shares(a_keys, b_keys) -> {"source": [...], "near_dup3": [...], "provenance": [...], "lab": [...]}   # empty = disjoint
def allowed_judges(stratum_keys, judge_material) -> list[judge]
def scope_for(stratum_keys, domain, kt_keys) -> str          # §4.15 scopes
def check_circularity(hypothesis, sets, domain) -> None      # CircularityError: a claimed set used for its own test
def se_sp(pairs) -> (se est, sp est)                         # positives: truth target, probe = truth;
                                                             # negatives: attractors (probe = confused_with) and
                                                             # targets with siblings (probe = sibling)
def judges(prereg, domain, funnel_dir, adapter) -> dict       # judge_qualification.json (locked)
def rl(prereg, domain, funnel_dir) -> dict                    # rl_qualification.json
def phi(errors_a, errors_b) -> float
```

- **Machine judges** (F5) are scored on the known-truth items directly, not on sheets.
  - For `other_named`, the negatives are KT7 attractors. For the other types they are KT5 attractors [§4.3].
  - Rescue is measured on J1's errors in KT2 and KT7.
  - KT7 never qualifies a judge in its `never_qualifies` list (J1, J-zs); the attempt raises `CircularityError` (mutation point M2 covers "the probe used as its own judge").
  - φ ≥ 0.5 puts two judges in one correlated group.
- **RL qualification** (F7) counts only sentinels of `qualify_rl_on` sets, per scope (mutation point M6). KT4–KT6 sentinels go to `agreement_only`. `unsure` and `unparsed` count as wrong. The identity check follows `identity_checks`: it passes when no item is answered as an excluded taxon and the share answered as the class is ≥ `pass_share_min` [§7]. Pair qualification for RL-B uses the pair sentinels.
- Thresholds are never fitted on the audit sample: `qualify` reads sentinels and known truth only, and refuses a gold row whose group is not `sentinel`, `identity` or a pair sentinel (mutation point M3).

Tests (`tests/test_funnel_qualify.py`):
- `shares` finds each of the four kinds;
- a judge whose bank holds a source of a stratum is not allowed there;
- qualifying the RL for H1 on KT4 raises `CircularityError`;
- a KT7-based qualification of J-zs raises;
- Se and Sp on a hand-computed example, including a sibling negative;
- Lu-lab strata get scope `KT7`;
- unsure counts as wrong;
- the identity check passes at 24 of 30 with no excluded answer and fails with one A. trifida answer;
- φ on a hand-computed table;
- a gold row from G1 passed to `rl` is refused.

#### 5.4.4 `recover.py` (verb `recover`)

```
POLICIES = ("R-A", "R-C", "R-T", "R-V", "R-J", "R-F")
def gates(audit, rl_qual, relation, leak, domain) -> {stratum: gate record}
def plan(policies, ledger_rows, gates, class_maps, name_status, resolver, judge_scores, leak, domain) -> list[image plan]
def mask(image_path, boxes, out_dir) -> (path, sha256)                  # mean-RGB fill, EXIF-transposed, PNG
def write_overlay(plans, out_dir, adapter) -> dict                       # labels_overlay/, images_masked/, recovered_pool.jsonl
def guard(rows, adapter, leak_calibration) -> dict                      # never-train and H6 on unmasked AND masked
def domain_dev(sources, ledger_rows, domain, seed_prefix="funnel/v1/h10d") -> dict
def run(prereg, domain, funnel_dir, out_dir, policies, adapter) -> dict # recovery.json
def arms(realloop_exp_dir, out_dir, base_manifest, adapter) -> dict      # arms/*.jsonl + arms.json
```

**Policies** [§7]. The scope, label policy and gates are as the contract's table. As pinned here:
- **R-A:** images of the `authoritative` sources with S10 ≠ admitted. Target boxes keep the source label. Boxes whose v2 status is taxon_resolved, target_related or role stay OtherPlant (the sibling guard, mutation point M4). The gates are H1-pre pass, H1 supported, the identity check of every class in `identity_checks` with `before` holding R-A, and the box gate per (source, label).
- **R-C:** mh_weed16 images holding an aligned id mapped by an accepted class map. The other ids stay OtherPlant. The gates are H3a supported, the class gate and the box gate. A class map is accepted when its proposal is `card+geometry` and the purity lb is ≥ 0.8.
- **R-T:** (source, src_id) classes with v2 status `target_synonym` (a scientific resolution only; a vernacular match never maps). The gate is the class gate on the G3 unit.
- **R-V:** the images with a vetoed verified box. Verified and other_ok boxes are kept; unknown and conflict boxes are masked. The gate is the box gate on G2v.
- **R-J:** OtherPlant boxes in `noinfo` strata of sources not quarantined by H6. A box is relabelled when every qualified judge allowed for its stratum gives the same target as top-1 and no source taxon of the class is a relative of that target. Any other box in the image that any judge calls a target is masked. The gates are H7 qualification, H2a supported, and the box gate per (source, predicted class).
- **R-F:** refetched images. They first pass the whole Step 1 guard chain (never-train, the train_core copy check at 6 bits, exact_dup against pool_meta, H6 `detect`, and the embedding by `embed-judges --refetch`), then go through R-C. `recover` refuses R-F without the chain record.

**Gates.**
- *Box gate:* the stratum's RG-corrected precision (label right and box valid) has lb ≥ 0.85 and point ≥ 0.90 at the level used (species; genus for MorningGlory only).
- *Class gate:* purity lb ≥ 0.8.
- A genus answer "*Amaranthus*, species unsure" is never recovered as a species; such a box is masked [§7].

**Never recoverable:**
- guard stages; `no_boxes` images; small boxes;
- named relatives under H2b;
- `not_recoverable` sources and sources H6(a) quarantined (whole source, mutation point M9);
- the H10d hold-out groups.

**Guard.** Every recovered row passes:
- the never-train guard on the unmasked image (pool_meta dHash) and on the masked copy (the dHash of the file) — mutation point M8 covers dropping the unmasked check;
- `leak.detect` on both.
A hit raises `NeverTrainHit` and stops F9 [§10 stop rules].

**Source labels.** The sha256 of every step1 label read is checked again after the write (`source_labels_unchanged`).

**H10d hold-out.** Per recovered source, one near-duplicate group (or one capture-stem group when the config declares a stem regex) with ≥ `min_group_images` images is chosen with seed `funnel/v1/h10d/<slug>`. It is written to `domain_dev.jsonl` and excluded from every pool.

**Arms** (`recover --arms --realloop EXP`), per [§9.2]:
- **U** = B ∪ every recovered image. The recovered part is capped at |B| by a seeded draw stratified by pool (seed `funnel/v1/arms/U`).
- **U_ctl** = the same images with `ctl_label` and the unmasked image.
- **CLASS_ctl** and **JUDGE_ctl** = B ∪ the REC-CLASS-1 or REC-JUDGE-1 images of `EXP`'s manifests, with `ctl_label` and unmasked images.
- B's rows are copied from `base_B.jsonl` verbatim. Every arm manifest is written with `C.write_manifest`.

Tests (`tests/test_funnel_recover.py`, on `funnel_world`):
- each policy on a planted case;
- the sibling guard keeps a named relative OtherPlant;
- a gate below 0.85 recovers nothing for that stratum and says why;
- a masked copy of a planted near-eval image is refused on the unmasked check even though the masked dHash passes;
- a quarantined source contributes no image, even its clean images;
- the mask fills exactly the rounded-out pixels with the mean RGB and the PNG is lossless;
- a genus-unsure *Amaranthus* box is masked;
- the source labels are unchanged;
- the H10d group is absent from every pool;
- the arms' composition, the U cap and the control labels.

### 5.5 G-autopilot

#### 5.5.1 Evidence allow-list (`inc_autopilot/evidence.py`)

Add to `STEP1_FILES`: `pool_summary.json`, `calibration.json`, `verifier_fit_info.json`.

Add the pattern `funnel/(funnel_ledger|audit_v1|class_maps|recovery|prospective_da)\.json`, and `funnel` to `RESERVED_DIRS`.

Add the virtual artifact `campaign/claims.json`, passed in by the ticker like `campaign/context.json`.

The scrub rules are unchanged, except for the exam list (§5.5.8).

`step1/verifier_fit_info.json` and `funnel/recovery.json` are provenance `"source": "derived"` and `"shipped_as"` respectively (the label-audit precedent).

#### 5.5.2 Diagnoses (`inc_autopilot/diagnose.py`, `thresholds.json`)

The signals are functions of the funnel ledger, the claims and the loop reports only, so the same code runs on any domain (R13):

| Signal | Computed from | thresholds.json key(s) |
|---|---|---|
| S1 | (Σ discarded at role `target_check` + `other_check` discarded conflict) / kept at `target_check` | `SIGNALS.S1_reject_accept_min` 1.0 |
| S2 | Σ `label_spaces[*].kinds` none + numeric + generic / Σ boxes; the cite names `name_status_version` | `SIGNALS.S2_uninformative_share_min` 0.10 |
| S3 | per class: `target_check.kept_by_label[k]` / `join.kept_by_label[k]`, over classes with join ≥ min; fires below frac × median | `SIGNALS.S3_min_joined_boxes` 100, `SIGNALS.S3_yield_frac_of_median` 0.25 |
| S4 | every known-truth set's `domain_score` within [reference.q05, 1] and at least one source median below reference.q05 | `SIGNALS.S4_min_sources_below_q05` 1 |
| S5 | `ledger.unaudited_dependencies` non-empty | – |
| S6 | `reject_class`: the drawn sample's sources intersect the `target_check` discard sources, or the top-cell share ≥ min (drawn when known, else eligible) | `SIGNALS.S6_top_cell_share_min` 0.5 (§9, item 19) |
| S7 | a loop step of kind in the list is truth `helps` while no clean step is | `SIGNALS.S7_rejected_kinds` ["unverified"], `SIGNALS.S7_helps` "helps" (mirrors `gate.HELPS`) |

- **D17 `scarcity_conclusion_unaudited`** (warn) = (C) ∧ (S) ∧ (A) [§8.4].
  - (C): the latest finished `inc.realloop build` experiment has no ACCEPT and no clean step `helps`; or its build summary records evidence sizing below the default M; or `claims.open_negative` is non-empty.
  - (A): no evidence artifact `funnel/audit_v1.json` has `ledger_fingerprint` equal to the ledger's `fingerprint`.
  - Keys: `D17.conclusion_polarities` ["scarcity", "negative"], `D17.claim_statuses` ["open", "challenged"].
  - It proposes L10 (verb `census`, ranked first), L12 and L11 for the sources S2 names, the DA pass, and X11 when S4 or S6 holds. It never proposes L13 (`only_after: ["D18"]`). The (C) condition's `if` is mutation point M1.
- **D19 `filter_recall_unmeasured`** (warn): any of the `D19.signals` ["S1", "S4", "S6"] holds for a stage in `ledger.recoverable_stages` whose `audit` is null. It proposes L10, ranked before L2 (DEC-4).
- **D18 `filter_false_negatives`:** reads `funnel/audit_v1.json d18_inputs`.
  - Refused as `calibration_overlap` when `valid` is false.
  - Fires per stratum when `fn_lb ≥ D18.fn_lb_min` (0.10) and `recoverable_lb ≥ D18.recover_min_frac_of_verified` (0.25) × the `target_check` kept count.
  - Levers by `kind` as [§8.4]. `relative_of_prediction` true → recorded as a known confusion, never L13.
  - When D18 is silent on a valid audit whose fingerprint matches, it proposes the claim transition to `tested_survives`.
- Rules order: D17 and D19 after D2, and D18 after D17 (`RULES`). `NAMES` gains the three.

#### 5.5.3 Levers and cards (`inc_autopilot/levers.json`, `levers.py`, `brain/policy_actions.json`)

| Lever | kind | policy_action | risk | argv |
|---|---|---|---|---|
| L10 `funnel_audit` | job | `inc_funnel_audit` | R2 | `["sbatch", "{*sbatch_resources}", "run_inc_funnel.sh", "{verb}", "--prereg", "{prereg}", "--out", "{out}", {"if": "part", "tokens": ["--part", "{part}"]}, {"if": "rl", "tokens": ["--rl"]}]`; verb ∈ {census, leak, embed-judges, qualify, draw, sheets, rl-b, ingest, estimate}; `sbatch_resources` derived from `funnel.__main__.SBATCH_RESOURCES[verb]` (empty for CPU verbs) |
| L11 `resolve_label_spaces` | job | `inc_funnel_map` | R2 | `["sbatch", "run_inc_funnel.sh", "map", "--prereg", "{prereg}", "--out", "{out}", "--part", "{part}"]` |
| L11a `fetch_cards` | lab | `inc_funnel_fetch` | R0 | `["python", "-m", "<pkg>.tools.funnel", "fetch", "--prereg", "{prereg}", "--what", "{what}", "--out", "{out}"]`; what ∈ {cards, kt7, known-items, refetch} |
| L12 `taxonomy_resolve` | lab | `inc_funnel_fetch` | R0 | the same, with what = taxonomy and `--names-from {names_from}` |
| L13 `recover_increment` | job | `inc_funnel_recover` | R3 | `["sbatch", "run_inc_funnel.sh", "recover", "--prereg", "{prereg}", "--audit", "{audit}", "--maps", "{maps}", "--policy", "{policy}", "--out", "{out}"]`; `only_after: ["D18"]` |
| L14 `verify_queue` | lab | `inc_verify_queue` | R2 | writes `sheets.person_items` rows to `<brain dir>/<domain>/known_truth/human_verify_queue.jsonl`; answers are read from `human_verify.jsonl` by `rl.ingest` |

`<pkg>` is the package name. It stays literal in levers.json; the grep test covers only `FUN`.

**Preconditions.**
- L10 `census` has the precondition "`funnel/taxonomy_cache.json` is on the cluster", read from `funnel summary`'s file list. Until it holds, L10 is proposed and deferred with `then: L12`, in the way R2 defers L2 behind L3.
- L10 `estimate` requires `funnel/prospective_da.json` on the cluster.
- A `refusals` entry maps the census message "no resolution ... run fetch --what taxonomy (lever L12)" to prerequisite L12 with `retry: true`.

The realloop_v2 build is L2 with the new params `increment_sources: recovered`, `step1_overlay` and `size`. The `protocol.increment_sources_modes` mirror becomes `inc/realloop.py INCREMENT_SOURCE_MODES`. The baselines of §6 F10 are L8 with `seeds`.

Data movement is the action `inc_funnel_sync` (R0), one ssh, with a fixed file list (§6.3).

**Cards** (R4, never queued):
- X10 class-definition policy.
- X11 verifier recall out of domain: raised by D17 when S4 or S6 holds, and on H1 supported.
- X12 claim status sign-off.

The `hypothesis`, `required_change`, `cheapest_test`, `control` and `success_criterion` fields paraphrase [§8.5].

#### 5.5.4 Campaign (`inc_autopilot/campaign.py`)

- `Paths.claims` = `campaign_dir / "claims.json"`.
- The ticker's context gains `claims`, loaded with `claims.load`, and the evidence gains `campaign/claims.json`.
- The DA pass is staged when D17 fires, when a claim of polarity scarcity or negative is written, and before a COMPLETE card that carries such a claim [§8.6].
- A COMPLETE card may not say "concluded" for a claim in `challenged`.
- Every claim transition is a campaign ledger entry with `decided_by`.
- The decisions DEC-1..DEC-10 are logged once with `decided_by: human-delegated` [§13].

#### 5.5.5 Adversary (`model_router.py`, `brain_plan.py`, `run_inc_plan.sh`, `validate.py`)

**Router.** `ROLES["adversary"] = {"judgement": True, "place": "cluster", "model": "vllm:glm-4.7-flash", "is_async": True, "fallbacks": ["ollama:gemma4"], "desc": "Devil's advocate against negative claims (a family other than the planner's)."}`.

`model_family(model_id)` is the leading letters of the name after the provider prefix, lowercased: "vllm:glm-4.7-flash" → "glm"; "ollama:qwen3.8:27b" → "qwen"; "ollama:gemma4" → "gemma". `same_family` compares the models actually resolved at run time for this reply and for the planner's last reply [§8.2 review].

**Digest.** `brain_plan.py digest --role adversary --claims PATH ...` builds `inc-da-digest/1`: the allow-listed evidence (dev only), the claim text, and the ledger's stage ids with their roles.
- It must not contain any of `DA_BLIND_MARKERS`: the contract's path and sha256, the prereg's path and core sha256, the R14 fixture's sha256, and the ids and argv of the D17 proposals.
- `check_staged` refuses a digest holding one of them.

**Job.** `run_inc_plan.sh` reads `PLAN_ROLE` (default `planner`) and resolves the model with `model_router.resolve($PLAN_ROLE)` and its fallbacks. `brain_plan.py run --role adversary` writes an `inc-da-reply/1` JSON.

**Reply schema:** [§8.6] plus `"stage_forecast": {stage id: probability}` over the ledger's recoverable stages, summing to 1 within 1e-6 [§6 H11 review].

**Validation.** `validate.validate_da(reply, evidence, ledger, menu, resolved_models) -> record` applies the existing cite and literature rules plus:
- a counter-argument citing fewer than 2 distinct artifacts is dropped;
- one naming a non-dev split (the domain's list) is dropped;
- a `prediction.stage` not in the ledger is dropped;
- a `cheapest_test` that is neither a menu lever whose preconditions hold nor X10/X11 is dropped;
- one that adds no artifact beyond D17's own cites is not counted (echo);
- `stage_forecast` must be valid, or the whole reply is invalid;
- a same-family reply may be recorded but moves no claim.

Invalid replies are recorded as invalid and not retried until they pass [§10 F2].

**Outcome.** `outcome.py` scores each DA prediction when its test lands, and H11 once `audit_v1.json` exists: top-1 and log-loss against uniform, recorded under the adversary role's track record.

#### 5.5.6 Remote verbs (`inc_autopilot/remote.py`)

- **`funnel summary`** prints one INCAP line with the sha256 and compact JSON of `funnel/funnel_ledger.json`, `funnel/audit_v1.json`, `funnel/class_maps.json`, `step1_r1/recovery.json` (shipped as `funnel/recovery.json`), `funnel/prospective_da.json`, `step1/pool_summary.json`, `step1/calibration.json`, and `step1/verifier_fit_info.json` (built by `adapters.inc_step1.verifier_fit_info_projection`). It also lists the name, sha256 and size of every file under `funnel/` (not their content), so levers can check preconditions. Aggregates only. It never ships `conflicts.csv`, `pool_verdicts.npz`, `ledger.jsonl`, `sample_v1_key.jsonl`, `sheets_v1_key/`, `sheets_v1_cluster/`, `leak_eval_desc.npz` or `leak_pairs_v1.csv`.
- **`funnel dev-scores --exp E [--exp ...]`** ships `runs/<run>/scores/dev.json` of `base`-kind runs only, for `panel.py`.
- **`funnel ledger-summaries`** runs `ledger_from_summaries` on the cluster's Step 1 summaries and `census_v0.json`, and returns the result (F2).
- `submit` gains the builder `funnel` (run_inc_funnel.sh) with the grammar of §5.6.1. Its resource flags come from `SBATCH_RESOURCES`.

#### 5.5.7 `inc_autopilot/panel.py` (new, lab)

```
def build_panel(dev_scores: {exp: {run_id: score dict}}, spec) -> dict      # panel.json content
def spec_v2() -> dict        # H10a: U vs B; H10b: U vs U_ctl; H10c: CLASS_ctl vs REC-CLASS-1 "with" runs,
                             # JUDGE_ctl vs REC-JUDGE-1 "with" runs (realloop_v2 truth arm)
```

- **H10a:** `gate.truth_detail` on seeds 0–2 of each arm (the pinned 3 v 3 rule), plus `estimate.perm_test_one_sided` on seeds 0–4 of U and of B (B's seeds 0–2 from base_b_v1 and 3–4 from rv2_B_extra, the same manifest bytes). Supported only if both pass [§6 H10a].
- It reads dev only. `exams_read` must be `["dev"]`; the test-blindness replay covers `panel.py` [§9.3 acceptance].

#### 5.5.8 Test blindness from the domain config

- `model.non_dev_exams(domain=None)` = `funnel.domain.load(domain or M.DOMAIN).non_dev_exams()`.
- `evidence.NON_DEV_EXAMS`, `remote.NON_DEV_EXAMS`, `brain_plan.FORBIDDEN_EXAMS` and `validate.LEAK_TEXT_RE` are built from it. The `cwd12 test` phrase comes from `domain_terms` [§8.8 review].
- A vehicles config's exam name is refused in R13.

#### 5.5.9 Executor

- `REPLAY_REQUIRED += ("R9", "R9_early", "R9b", "R10", "R11", "R12", "R13", "R14", "funnel_negative_controls", "funnel_mutations", "domain_free")`.
- `REPLAY_SCRIPTS["funnel"] = "tests/test_inc_ap_funnel.py"` and `REPLAY_SCRIPTS["funnel_mutations"] = "tests/test_funnel_mutations.py"`. `run_replay_tests` runs them with the same closing-line rule.
- `GOVERNANCE_FILES` gains both scripts and `FUN/domains/weed.json`.
- The builder grammar gains `funnel`, `realloop --increment-sources recovered --step1-overlay`, and `pilot build-baseline --seeds`.

#### 5.5.10 Tests (G-autopilot)

**Fixtures.** `tests/fixtures/inc_replay/funnel/`, pinned in `MANIFEST.json`:
- byte copies of `census_v0.json`, `step1/pool_summary.json`, `step1/calibration.json` and `realloop_v1/{exp, report, ledger, build_summary}.json`;
- the derived `funnel_ledger.json` (`ledger_from_summaries` on the local files);
- `claims_seed.json` with C1 and C2;
- `vehicles.json` (R13);
- synthetic audits for R10, R11 and R12, marked `"synthetic": true`;
- the R14 replies;
- `verifier_fit_info.json` once it is pulled.

**Seed claims.**
- **C1:** "The harvested increments do not improve the twelve target species. They are mostly other plants plus about 50 target-species boxes each." Polarity `negative`, `made_by: human-transcribed`, cites `RESEARCH_LOG.md` line 154 and `realloop_v1/report.json /agreement/full`.
- **C2:** "web harvest at this scale supplies volume, not usable supervision". Polarity `scarcity`, cites `docs/poster/figures_data.json /s1_gate_verdict_2026_08_25/reading`.

**`tests/test_inc_ap_funnel.py`** has one case per [§8.9] row:
- **R9:** D17 fires with cites S1 = 11,488/2,049, S2 = 220,104/545,318 (v1), S3 Ragweed 27/3,029, S4, S5 (S12 → S8) and S7 (`/steps/2/truth/verdict` "helps"). Every number is read from the fixtures in the test. L10 `census` is first with the exact argv. L11 and L12 are proposed for the S2 sources. There is no L13. The DA is staged and X11 raised. R1–R8 outputs are unchanged apart from the rules version. It also holds under the test-blindness perturbation.
- **R9_early:** the Step 1 files without realloop_v1: D19 fires and L10 comes before L2. The case reports realloop_v1's GPU-h, read from its report.
- **R9b:** C2 with only the artifacts of 2026-08-25: D17 or D19 fires and asks for an out-of-domain known-truth set.
- **R10–R12:** as [§8.9], on the synthetic audits.
- **R13:** `vehicles.json` plus a synthetic funnel ledger and a claim: D17 and D19 fire with the same code and thresholds, and the vehicles exam name is refused by evidence, remote and validate.
- **R14:** the positive reply keeps 6 counter-arguments and 2 concessions. The sycophantic reply keeps 0 and the claim stays `challenged`. A fabricated cite, a test leak and an echo are dropped or not counted. A same-family model cannot move the claim. A digest containing a blind marker is refused.
- **funnel_negative_controls:** pilot_v1–v3, b0_v1 and base_b_v1: D17 is silent. A synthetic high-yield Step 1 with out-of-domain calibration: D19 is silent.

**`tests/test_funnel_mutations.py`:** §7.6.

**`tests/test_funnel_domain_free.py`:** §7.5.

**`tests/test_inc_ap_panel.py`:** panel on synthetic dev scores; the 5 v 5 permutation p; no non-dev value read.

The existing `test_inc_ap_*.py` suite must still pass. The existing lever tests are extended to the new rows.

### 5.6 G-loop

#### 5.6.1 `funnel/__main__.py`

```
VERBS = ("census", "leak", "embed-judges", "qualify", "draw", "sheets", "rl-b", "ingest", "estimate", "map",
         "recover", "fetch", "sbatch-args")
VERB_CLASS = {"census": "cpu", "leak": "gpu", "embed-judges": "gpu", "qualify": "cpu", "draw": "cpu",
              "sheets": "cpu", "rl-b": "gpu_large", "ingest": "cpu", "estimate": "cpu", "map": "cpu",
              "recover": "cpu", "fetch": "lab"}
SBATCH_RESOURCES = {                                    # extra sbatch flags, before the script name
  "cpu": [],                                            # run_inc_funnel.sh's own #SBATCH lines (one V100, idle)
  "gpu": [],                                            # the same lines; the verb uses the V100
  "gpu_large": ["--gres=gpu:h100-80:1", "--cpus-per-task=12", "--mem=80G"]}
def main(argv=None) -> int
```

Common options: `--prereg PATH` (required; the domain comes from it, and there is no `--domain`), `--out DIR` (default `INC_DIR/funnel`), `--force`, `--quiet`.

| Verb | Options | Calls |
|---|---|---|
| census | `--taxonomy PATH` `--known-items PATH` `--summaries-only` `--census-v0 PATH` | `adapter.census` / `adapter.ledger_from_summaries` |
| leak | – | `leak.run` |
| embed-judges | `--stage embed\|judges\|all` `--shard i --nshards n` `--refetch` | `embed.*`, `judges.score_all` |
| qualify | `--rl` | `qualify.judges` / `qualify.rl` |
| draw | – | `draw.draw` |
| sheets | – | `sheets.run` |
| rl-b | `--endpoint URL` (default `$FUNNEL_OLLAMA_ENDPOINT`) `--model TAG` `--sheet-dirs D ...` `--max-attempts 3` | `rl.run_rl_b` |
| ingest | `--answers DIR ...` (default `out/rl_answers/*`) | `rl.ingest` |
| estimate | – | `estimate.evaluate` → `audit_v1.json`, `audit_v1.md` |
| map | `--part geometry\|relation` | `relation.run_geometry` / `relation.run_relation` |
| recover | `--audit PATH` `--maps PATH` `--policy R-A,...` (`--out` default `INC_DIR/step1_r1`); `--arms --realloop EXP --base PATH` | `recover.run` / `recover.arms` |
| fetch | `--what LIST` `--names-from PATH` `--sources PATH ...` | `fetch.*`, then `fetch.write_manifest` |
| sbatch-args | `VERB` | prints `SBATCH_RESOURCES[VERB_CLASS[VERB]]`, one per line |

- Exit codes are those of §1.2. Each verb prints one closing line `[funnel] <verb>: <summary>` to stdout.
- A GPU verb run without a CUDA device refuses (exit 2), unless `--testing`, which the tests use with fake models.

Tests (`tests/test_funnel_cli.py`): argparse for every verb; the exit code for a `FunnelError` (2) and for an unexpected exception (1); `sbatch-args` equals the table; `--domain` is rejected; every verb reaches its function (monkeypatched).

#### 5.6.2 `run_inc_funnel.sh`

It is modelled on `run_inc_verify.sh` (argument handling, array shard) and on `run_inc_plan.sh` (the nested copy, the sha256 log, in-job ollama):

```
#SBATCH --job-name=inc_funnel
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=45G
#SBATCH --time=08:00:00
#SBATCH --output=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/inc/funnel/logs/%x_%j.out
```

**Why GPU-shared for every verb.** The allocation cis240145p is a GPU allocation, and RM-shared submissions fail with "Invalid qos" (`run_inc_audit.sh`, `run_inc_build.sh`, CHANGELOG v3.0.99.19). The CPU verbs therefore hold one V100 that stays idle, as `run_inc_audit.sh` and `run_inc_build.sh` do. 45G is `run_inc_audit.sh`'s size for loading every crop embedding, which census also does. The SU cost is charged to the campaign envelope (DEC-10).

- **Environment.** `REPO` is exported; `INC_DIR` = `$REPO/results/framework/inc`; `CODE="$REPO/weed_llm_benchmark"`; `cd "$CODE"`; `PYTHONPATH="$CODE"` (the git-tracked nested copy); conda env `bench`; `HF_HUB_OFFLINE=1`.
- **Verb check.** The verb is checked against `VERBS` without `fetch` and `sbatch-args` (fetch is lab-only). In an array job, `embed-judges --stage embed` adds `--shard $SLURM_ARRAY_TASK_ID` unless `--shard` was given.
- **GPU check.** For a verb of class gpu, `nvidia-smi` must list a GPU. For rl-b it must list one with at least 80 GB. Otherwise exit 2 with the exact `sbatch $(python -m ... sbatch-args VERB) run_inc_funnel.sh VERB ...` line to use.
- **Inputs present.** `$REPO/docs/FUNNEL_AUDIT.md` must exist; its sha256 is checked by `domain.load_prereg` against the prereg.
- **Code log.** The sha256 of every file under `FUN`, plus `inc/{__init__,common,verify,select,relevance,driver,gate}.py`, `cwd12_species.py`, `near_dup.py` and `semisup_labeler.py`, is logged from the nested copy. The script refuses when `$REPO/run_inc_funnel.sh` differs from `$CODE/run_inc_funnel.sh`.
- **Imports.** It checks `numpy`, `sklearn` and `joblib`, plus `torch`, `transformers` and `open_clip` for GPU verbs, and `PIL` for leak, sheets and recover.
- **rl-b.** Ollama starts on a per-job port, as in `run_inc_plan.sh` (the same binary, `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_LOAD_TIMEOUT=30m`, two bounded warm-ups). `FUNNEL_OLLAMA_ENDPOINT` is exported, and the tag comes from `weed.json reference_labeller.backends.RL-B.model` through the Python loader.
- **Run.** `python -u -m weed_optimizer_framework.tools.funnel "$VERB" "$@"`; exit with its code.

Tests (`tests/test_funnel_job_script.py`, static): the `#SBATCH` lines; the verb `case` list equals `VERBS` minus fetch and sbatch-args; every module under `FUN` is in the hash list; the ollama block mirrors `run_inc_plan.sh`'s variables; `bash -n` passes.

#### 5.6.3 `inc/realloop.py`: the recovered protocol version

```
INCREMENT_SOURCE_MODES = S.SOURCE_MODES + ("recovered",)
RECOVERED_SEQUENCE = ("REC-VETO", "REC-AUTH-1", "REC-CLASS-1", "REC-JUDGE-1", "REC-AUTH-2", "REC-CLASS-2")
RECOVERED_POOLS = {"REC-VETO": "VETO", "REC-AUTH-1": "AUTH", "REC-AUTH-2": "AUTH", "REC-CLASS-1": "CLASS",
                   "REC-CLASS-2": "CLASS", "REC-JUDGE-1": "JUDGE"}
SUBSTITUTION_ORDER = ("AUTH", "CLASS", "VETO")
KIND_RECOVERED = "recovered"
RECOVERED_RULE = "..."          # the draw rule below, as text
def load_overlay(step1_overlay, base_rows, guard, testing) -> dict
def plan_recovered(pools: {pool: [group]}, m, base_n) -> {"m": int, "steps": {step: pool}, "dropped": [...], "substituted": [...]}
def draw_recovered(exp, step, pool_groups, m) -> list[row]
build(..., increment_sources="recovered", step1_overlay=PATH, size=M)
```

- **Flags.** `--increment-sources recovered` requires `--step1-overlay` and `--size`. It refuses `--relevance`, `--min-evidence` and `--n-verified`. `--step1-overlay` without the mode refuses. argparse `choices` becomes `INCREMENT_SOURCE_MODES`.
- **`load_overlay` refuses** (before anything is written):
  - `recovery.json` not `complete`;
  - a `recovered_pool.jsonl` whose sha256 differs from the one recorded;
  - a row whose image or label does not hash as recorded;
  - a row that is in the base (key, path or sha256);
  - a row whose unmasked image fails the never-train guard (through `pool_meta.jsonl` dHash);
  - a row whose image fails `check_training_manifest`'s guard.
- **Plan.**
  1. `pool_groups`: the pool's rows grouped by 3-bit near-duplicate group.
  2. `floor` = 0.05 × |B| (real).
  3. An arm whose capacity per step (images / steps drawn from it) is below the floor is dropped and reported.
  4. M_eff = min(M, min capacity per step over the kept arms).
  5. If M_eff < floor, the build refuses with the numbers.
  6. A dropped arm's steps are drawn from the arm with the most remaining capacity in `SUBSTITUTION_ORDER`. A tie goes to the earlier arm. If no arm has M_eff left, the build refuses.
- **Draw.** Whole near-dup groups are taken in a seeded order: `numpy.random.default_rng(C.stable_int(exp + "/" + step))`, a permutation of the pool's groups sorted by first key. Groups are added while they fit M exactly (skipping a group that overshoots); the draw refuses if it cannot reach M.
- **Steps.** Every step has `clean: false`, `kind: "recovered"`, `pool`, `policy_counts` and `substituted_from`. There is no UNVERIFIED and no OTHER_HEAVY. The truth arm is on.
- **Records.** `exp.json` `step1.increment_sources` = `{"mode": "recovered", "overlay": {"dir", "recovery_sha256", "recovered_pool_sha256"}, "rule": RECOVERED_RULE, "m_requested": M, "m": M_eff, "dropped": [...], "substituted": [...]}`. `build_summary.json` gains `recovered` (the plan, capacities, per-step counts by pool and class, and the guard record).
- **Unchanged.** A definition built without `recovered` is byte for byte the one built before this change: `DEFAULT_BUILD_DIGEST` in `tests/test_inc_realloop.py` still passes unchanged. realloop_v1's `exp.json` still re-inits under the same `--exp` in a copy. The driver pins (`PINNED_MODULES`) are untouched, so running experiments keep their code pin.

Tests (`tests/test_inc_realloop_recovered.py`, on `funnel_world` plus a synthetic `step1_r1`):
- the sequence and pools;
- every step holds M images, is disjoint from the base and from the other steps, and keeps near-dup groups whole;
- the plan's M_eff, dropping and substitution on three planted capacity cases (all arms full; JUDGE below the floor; AUTH short by one step);
- the floor at 0.05 × |B| (M 196 < 196.35 refuses, 197 builds);
- every `load_overlay` refusal on its own planted defect;
- driver init on `FakeBackend`, with the truth arm comparing each step with the base runs (all steps unclean);
- `tests/test_inc_realloop.py` passes unchanged.

#### 5.6.4 `inc/pilot.py`: recovery manifests in build-baseline

```
def recovery_provenance(manifest_path, rows, testing=False) -> dict | None
```

- A manifest is a recovery manifest when any row's label lies under `R1_DIR/labels_overlay` or its image under `R1_DIR/images_masked`, or when the manifest file lies under `R1_DIR`.
- For such a manifest:
  - `arms.json` or `recovery.json` must name the manifest's sha256 (an arm) or every recovered row must appear in `recovered_pool.jsonl` with the same image and label sha256;
  - `recovery.json` must be `complete`;
  - the unmasked originals must clear the never-train guard.
  A production build refuses a failure; a testing build warns [§9.2].
- `build_baseline` records `recovery_build` in `build_summary.json` only for a recovery manifest. The key is absent otherwise, so a non-recovery build's summary is unchanged.
- `--seeds 0,1,2,3,4` and `--seeds 3,4` already work (`check_seeds`).

Tests (`tests/test_inc_pilot_recovery.py`): an arm manifest builds and records `recovery_build`; a manifest with one overlay label edited refuses; a production build of a manifest whose masked original is near an evaluation image refuses; `base_b_v1`'s build summary on the same manifest has no new key; the existing `test_inc_*` pilot tests pass unchanged.

#### 5.6.5 `tests/test_funnel_e2e.py`

On `funnel_world`, with fake embedders, a fake text tower and a fake RL (answers derived from the world's truth with planted errors), it runs census → leak → embed-judges → qualify → map (both parts) → draw → sheets → rl-b (fake transport) → ingest → qualify --rl → estimate → recover → realloop build (testing) → recover --arms → build-baseline (testing), through the CLI.

It asserts every file of §4 exists and validates, every recorded input sha256 re-hashes, and the audit's H0(a) covers the world's planted share. It skips with a reason when numpy, PIL or sklearn is absent.

---

## 6. Order of execution, F3 to F10

`$INC` is `INC_DIR` on the cluster. `$LINC` is `LAB_INC` on the lab. `$PKG_MOD` is `weed_optimizer_framework.tools`. Cluster commands run from `$REPO`, with `mkdir -p $INC/funnel/logs` done once. Every step is taken by the platform (the autopilot proposes the lever, and the executor runs it within its risk tier). The commands are what a person would type to do the same by hand.

### 6.1 Commands

| Step | Where | Command | Produces |
|---|---|---|---|
| F2a | cluster login (R0) | `python -m $PKG_MOD.inc_autopilot.remote funnel ledger-summaries` | summaries-mode `funnel_ledger.json` in the snapshot; D17 and D19 fire; DA staged |
| F2b | cluster job | `sbatch --export=ALL,PLAN_ROLE=adversary,PLAN_INPUT=$INC/_campaign/plans/<c>/<n>.input.json weed_llm_benchmark/run_inc_plan.sh` | DA reply; the lab validates it → `$LINC/funnel/prospective_da.json`, pushed |
| F3a | lab (L12, R0) | `python -m $PKG_MOD.funnel fetch --prereg $LINC/funnel/prereg_v1.json --what taxonomy,known-items --names-from $LINC/step1/pool_summary.json --sources docs/literature --out $LINC/funnel/` | `taxonomy_cache.json`, `known_items_v1.json`, `fetch_manifest.json`; pushed |
| F3 | cluster (L10, R2, CPU) | `sbatch run_inc_funnel.sh census --prereg $INC/funnel/prereg_v1.json --out $INC/funnel/` | `census_v1.json`, `ledger.jsonl`, `funnel_ledger.json`, `name_status_v2.json`, `guard_pairs_v1.csv` |
| F4a | cluster (L10, GPU) | `sbatch run_inc_funnel.sh leak --prereg ... --out $INC/funnel/` | `leak_v1.json`, `leak_pairs_v1.csv`, `leak_eval_desc.npz` |
| F4b | lab (L11a, R0) | `python -m $PKG_MOD.funnel fetch --prereg ... --what cards,kt7 --out $LINC/funnel/` | `cards/`, `kt7/`; pushed |
| F4c | cluster (L11, R2) | `sbatch run_inc_funnel.sh map --part geometry --prereg ... --out $INC/funnel/` | `relation_geometry_v1.json`, `class_maps.json` |
| F5a | cluster (L10, GPU, array) | `sbatch --array=0-3 run_inc_funnel.sh embed-judges --stage embed --nshards 4 --prereg ...` then `sbatch run_inc_funnel.sh embed-judges --stage judges --prereg ...` | `emb_dinov2/`, `emb_*_kt7.npz`, `judges/*.npz` |
| F5b | cluster (L10) | `sbatch run_inc_funnel.sh qualify --prereg ...` | `judge_qualification.json` (locked) |
| F5c | cluster (L11) | `sbatch run_inc_funnel.sh map --part relation --prereg ...` | `relation_audit_v1.json` |
| F6 | cluster (L10) | `sbatch run_inc_funnel.sh draw --prereg ...` | `frames_v1*`, `sample_v1.csv`, `sample_v1_key.jsonl`, the sample-lock amendment |
| F7a | cluster (L10) | `sbatch run_inc_funnel.sh sheets --prereg ...` | `sheets_v1/`, `sheets_v1_cluster/`, `sheets_v1_key/`; `sheets_v1/` pulled to the lab |
| F7b | lab | RL-A: the external labeller answers `$LINC/funnel/sheets_v1/` into `$LINC/funnel/rl_answers/RL-A/` | answer files; pushed |
| F7c | cluster (L10, GPU large) | `sbatch $(... sbatch-args rl-b) run_inc_funnel.sh rl-b --prereg ... --sheet-dirs $INC/funnel/sheets_v1 $INC/funnel/sheets_v1_cluster` | `rl_answers/RL-B/` |
| F7d | cluster (L10) | `sbatch run_inc_funnel.sh ingest --prereg ...` then `sbatch run_inc_funnel.sh qualify --rl --prereg ...` | `gold_v1.csv`, `rl_qualification.json` |
| F8 | cluster (L10) | `sbatch run_inc_funnel.sh estimate --prereg ...` | `audit_v1.json`, `audit_v1.md`; `remote funnel summary` ships them; D18 reads |
| F9 | cluster (L13, R3) | `sbatch run_inc_funnel.sh recover --prereg ... --audit $INC/funnel/audit_v1.json --maps $INC/funnel/class_maps.json --policy R-A,R-C,R-T,R-V,R-J --out $INC/step1_r1/` | `step1_r1/` |
| F10a | cluster (L2, R3) | `sbatch run_inc_build.sh realloop build --exp realloop_v2 --base $INC/step1/base_B.jsonl --replay-mode full --recipes full --gate-flips-mode net --increment-sources recovered --step1-overlay $INC/step1_r1 --size 287` | `$INC/realloop_v2/` |
| F10b | cluster (L13) | `sbatch run_inc_funnel.sh recover --arms --realloop $INC/realloop_v2 --base $INC/step1/base_B.jsonl --prereg ... --out $INC/step1_r1/` | `step1_r1/arms/` |
| F10c | cluster (L8, R3) | `sbatch run_inc_build.sh pilot build-baseline --exp rv2_U --manifest $INC/step1_r1/arms/U.jsonl --seeds 0,1,2,3,4`; the same for `rv2_Uctl`, `rv2_CLASSctl` and `rv2_JUDGEctl` (seeds 0,1,2); `rv2_B_extra` on `$INC/step1/base_B.jsonl --seeds 3,4` | the baseline experiments |
| F10d | lab | `remote funnel dev-scores --exp ...`, then `panel.build_panel` → `$LINC/realloop_v2/panel.json`; `outcome.py` evaluates H10 | `panel.json` |

**Notes on the table.**
- `--recipes full` and `--gate-flips-mode net` are realloop_v1's, read from its `exp.json` by the lever; the X1 recipe replaces them only if X1 has landed [§9.1]. `--size 287` is realloop_v1's `increment_images`.
- `run_inc_build.sh` imports the outer package copy and stops on drift. The G-loop changes to `realloop.py` and `pilot.py` therefore reach the outer copy (lever X5, `remote.py sync-outer`, R3) before F10a. `run_inc_funnel.sh` imports the nested copy and needs no sync.
- **Stops** [§10]: H0 falsified → stop after F8 (`audit_v1.json stop`). An H6(b) copy in B → R4 incident before F9: C1 void, B rebuilt, B's baseline re-run before F10. A never-train hit in `recover` → stop F9. Campaign stop-losses and the 120 GPU-h cap (DEC-10) → no new submission.

### 6.2 Where each step runs

- **Cluster, compute node:** census, leak, embed-judges, qualify, map, draw, sheets, rl-b, ingest, estimate, recover, the realloop build, the baselines, and the DA job. Everything that reads pool pixels, crops, embeddings or evaluation images.
- **Cluster, login node:** the remote verbs (aggregates only).
- **Lab:** fetch (network), RL-A answering, claims, diagnoses, levers, the DA validation, `prospective_da.json`, `panel.py` and `outcome.py`.

### 6.3 Data movement (G-autopilot `inc_funnel_sync`, R0)

The file lists are fixed in code. Each transfer is verified on arrival against a `MANIFEST.json` of sha256 values (`fetch.check_manifest`).

- **Lab → cluster:** `funnel/{prereg_v1.json, census_v0.json, taxonomy_cache.json, known_items_v1.json, fetch_manifest.json, cards/, kt7/, refetch/, prospective_da.json, rl_answers/RL-A/}`.
  - `prereg_v1.json` and `census_v0.json` come from the git copy.
  - The push of `prereg_v1.json` refuses when the cluster copy holds more amendments, or another `core_sha256`.
- **Cluster → lab:** `funnel/prereg_v1.json` (only when its amendments grew; a person commits it), `funnel/{census_v1.json, name_status_v2.json, funnel_ledger.json, audit_v1.json, audit_v1.md, class_maps.json, relation_geometry_v1.json, relation_audit_v1.json, judge_qualification.json, rl_qualification.json, leak_v1.json, frames_v1.json, sample_v1.csv, sheets_v1/}` and `step1_r1/recovery.json`.
- **Never moved:** `sample_v1_key.jsonl`, `sheets_v1_key/`, `sheets_v1_cluster/`, `leak_eval_desc.npz`, `leak_pairs_v1.csv`, `ledger.jsonl`, `gold_v1.csv`, embeddings, judge score files, anything under `step1/` other than the three summaries, and any evaluation image.

The sync refuses a path outside the list.

---

## 7. Cross-cutting rules the tests enforce

### 7.1 Blinding

A labeller never sees a source name, verdict, score, stratum, proposal or unit id (machine RL), and the RL-A labeller never sees an evaluation pixel. Enforced by `sheets` (§5.4.1), `rl.validate_answers` and the sync list (§6.3).

### 7.2 Disjointness and circularity [§4.1]

- `qualify.shares` is the only implementation of "shares".
- `judges.knn` excludes by it, `sheets.board_for` assigns boards by it, `strata` records `allowed_judges` by it, and `estimate` re-checks every stratum's predictor and RL scope by it (R12).
- Circularity is `qualify.check_circularity`, driven by `known_truth.<KT>.claimed_by` and `qualify_rl_on`.

### 7.3 Freshness chain

The chain is:
- census ← Step 1 files;
- frames and sample ← census and `name_status_v2`;
- sheets ← sample and key;
- gold ← answers, sheets and key;
- rl_qualification ← gold;
- audit ← everything above plus the judge files and `prospective_da.json`;
- recovery ← audit;
- realloop_v2 ← recovery;
- arms ← realloop_v2.

Each consumer calls `check_records` on its producer's recorded inputs.

### 7.4 Numbers

Every audit number comes from `estimate.py` [§8.7]. Every mAP comes from `inc/scorer.py`. `panel.py` calls `gate.truth_detail` and `estimate.perm_test_one_sided` only.

### 7.5 Domain-free engine (`tests/test_funnel_domain_free.py`)

**Files.** Every `.py` under `FUN` except `adapters/` and `domains/`.

**Normalisation.** The text is lowercased, and every occurrence of `weed_optimizer_framework` and `weed_llm_benchmark` is removed (the code's own location).

**Terms,** from every `FUN/domains/*.json` and the R13 fixture, through `Domain.terms()`:
- *Substring, case-insensitive:* `domain_terms` plus the fixed list ("cwd12", "otherplant", "bioclip", "ndsu", "cottonweed"), and every source slug named anywhere in a config.
- *Whole token* (a maximal `[a-z0-9]+` run of the normalised text):
  - the `domain` name;
  - every class name with non-alphanumerics removed;
  - every word of length ≥ 5 in target taxa, `not` taxa, attractor taxa and KT7 taxa;
  - every exam name except "dev" and "test";
  - every lab group name.

**Planted checks.** The test also plants three temporary files in a copy: one with "Amaranthus", one with "OtherPlant" and one with "indicates". The first two must be flagged. "indicates" must not be flagged by the taxon word "indica".

**Coupling outside `FUN`** (`model.py DOMAIN`, the brain_plan prompt header, `verify.CWD12_RELATED_TOKENS`, the D2 summary strings) is out of the test's scope and listed as coupling to migrate [§8.8].

### 7.6 Mutation points (`tests/test_funnel_mutations.py`)

Each check below carries the comment `# funnel-mutation: <id>` at the end of exactly one line, and that line is an `if <condition>:` statement whose body enforces the check.

The harness:
1. copies the package to a temporary directory;
2. rewrites the marked line's condition to `False`;
3. runs the owner's test script against the copy (`PYTHONPATH` = the copy);
4. requires a non-zero exit.

A missing or duplicated marker fails the harness.

| Id | Check switched off | File (owner) | Killing test |
|---|---|---|---|
| M1 | D17's conclusion condition (C) | inc_autopilot/diagnose.py (G-autopilot) | test_inc_ap_funnel.py negative controls |
| M2 | J1 refused as a judge of its own discards / on KT7 | funnel/qualify.py (G-labels) | test_funnel_qualify.py |
| M3 | thresholds fitted only on sentinels, never on audit-sample rows | funnel/qualify.py (G-labels) | test_funnel_qualify.py |
| M4 | the sibling guard | funnel/recover.py (G-labels) | test_funnel_recover.py |
| M5 | a guard stage marked recoverable refused | funnel/domain.py (G-stats) | test_funnel_domain.py |
| M6 | the RL qualified on the claimed labels of the tested population | funnel/qualify.py (G-labels) | test_funnel_qualify.py |
| M7 | a kNN bank entry sharing a lab with the query excluded | funnel/judges.py (G-vision) | test_funnel_judges.py |
| M8 | the never-train check on the unmasked image | funnel/recover.py (G-labels) | test_funnel_recover.py |
| M9 | a source with a detected H6 copy excluded as a whole | funnel/recover.py (G-labels) | test_funnel_recover.py |
| M10 | an exam name in the test-blindness list | inc_autopilot/model.py (G-autopilot) | test_inc_ap_funnel.py R13 |

---

## 8. Acceptance of F1 (the build of this runner)

F1 is accepted when all of the following hold [§10 F1]:
- every test named in §5 passes, or skips only for an absent optional dependency with the reason printed;
- the full existing `tests/test_inc_*.py` and `tests/test_inc_ap_*.py` suites pass unchanged, including `DEFAULT_BUILD_DIGEST`;
- R9 reproduces S1–S5 and S7 exactly, from the fixtures;
- the negative controls are silent;
- R13 and the domain-free test pass;
- the mutation harness kills M1–M10;
- `executor.run_replay_tests` records a pass.

Each new module is then one R4 acceptance, recorded as DEC-5 says.

---

## 9. Choices this runner made where the contract left room

Each item names the contract text and the choice.

1. **Sample-lock amendment.** [§5.2] appends the lock to `prereg_v1.json`. Only `draw`, through `domain.append_amendment`, may change the file, and only its `amendments` list. Every artifact records `core_sha256`, so the lock does not make F3–F5 outputs stale (§3.3).
2. **`recovery.json` location.** [§7] puts it under `INC_DIR/step1_r1/`; [§8.2] lists `funnel/recovery.json` in the allow-list. The file lives in `step1_r1/`, and `funnel/recovery.json` is its shipped name (the label-audit precedent).
3. **Machine-RL options.** [§4.3] says "the stratum's named attractors". Every pool item gets the same superset (all targets, all attractors, the tail), so the option list cannot reveal the stratum.
4. **Exemplars for the machine RL.** [§4.3] shows per-item strips of the proposed class and its KT5 attractor. With no proposal (blind multiple choice) and with exemplar strips counting as calibration under the disjointness rule, each sheet instead shows one board: B1 (KT1 targets, KT7 attractors) or B2 (KT7 only) for items that share anything with KT1. KT5 is never an exemplar source, because its labels are claimed.
5. **RL qualification scope.** Qualification counts as calibration [§4.1 review]. So a stratum's Se and Sp use only the `qualify_rl_on` sets that share nothing with it: Lu-lab strata are qualified on KT7 alone.
6. **Two qualification files.** Machine judges are qualified at F5 (`judge_qualification.json`, locked before the draw). The RL and J-vlm need answered sentinels, so they are qualified at F7 (`rl_qualification.json`).
7. **Anchoring sentinels.** The deliberately wrong proposals of [§4.3] exist only on person verify-only items (L14). Machine sheets carry no proposal.
8. **Sheet size.** 15 items, 3 of them sentinels (the 20 % of [§4.3]). The sentinel count follows from the number of sheets; the contract's "about 400 + about 150" becomes the per-KT weights.
9. **G4 design.** The uniform 300 are split 225 named / 75 no-information (the power argument of [§6 H4]). The prioritised part is 150 Poisson draws with π ∝ J1's target score. The union inclusion probability is used.
10. **Allocation within groups.** Proportional with a floor of 5, largest remainder. G1 is split equally between the H2a and H2b frames.
11. **Name status v2.** The statuses and rule order of §4.4. Non-object and state names form an `excluded` frame, outside both H2 frames. "numeric" is split from v1's "generic" by `name_key(name).isdigit()`, which reproduces census_v0's 55,657.
12. **Floor of M.** [§9.1] says "5 % of B (196 images)". It is applied as the real value 0.05 × |B| (M ≥ 197 for |B| = 3,927), as `levers.json min_increment_frac` already does for R8.
13. **H10d.** `recover` holds out the domain-dev groups. Scoring them needs `inc/scorer.py` to accept a new exam, which it does only through `LOCK.json`: an R4 lock amendment, not built here.
14. **H8 and F11** are outside F3–F10. The audit reports H8 as `not_evaluated`.
15. **R-F** runs only after the whole Step 1 guard chain; `recover` refuses it without the chain record. It stays optional.
16. **Taxonomy offline.** The cluster has no network in jobs, so the cache is built on the lab (L12) and census refuses a miss, naming L12.
17. **H12** is compared in census, where the registry is. The list is fetched and hashed on the lab first.
18. **Class maps.** `class_maps.json` holds proposals only. Applying a map (R3) is decided in `recover` and recorded in `recovery.json`.
19. **S6 cut point.** The contract gives none. 0.5 (a majority of the reject-class sample from one cell) is set in `thresholds.json` with its reason. Since the 0.620 share was seen first, R9-type evidence for S6 is a reproduction.
20. **Verdict vocabulary.** supported, falsified, inconclusive and bounded, plus `not_evaluated` (H5b without a qualified RL-B, H8, shift cells), `descriptive` (H9) and `reported` (H6, H0c).
21. **H12 "supported"** is lb ≥ 0.70 (the prereg gives the target share and the falsifier).
22. **H5b qualification.** RL-B is qualified for pair judgments on pair sentinels: KT2 copy ↔ twin pairs are positives, and provenance-disjoint 7–10-bit pairs are negatives.
23. **`panel.py` on the lab.** The evidence never opens run directories. `remote funnel dev-scores` ships only dev score files of base runs, so the pinned `gate.truth_detail` can run on the lab.
24. **J-zs text tower.** The encoder class name contains a domain term, so it is obtained through the adapter and injected.
25. **Unsure answers** count as wrong in qualification (conservative) and are taken both ways in estimates [§4.3].
26. **Melded Rogan–Gladen interval.** A Monte Carlo over beta confidence distributions, in the direction of each component's effect.
27. **H9 cells.** A box no sampled stratum covers counts toward the cell's boxes with precision 0 in the lower bound.
28. **Ragweed identity check.** It passes when no crop is answered as *A. trifida* and ≥ 80 % are answered Ragweed.
29. **Funnel ledger extensions.** `derivation`, `fingerprint`, `name_status_version`, `role`, `guard`, `kept_by_label`, `discarded_by_label` and `reject_class` are added to funnel-ledger/1, so that S1–S7 are computed from the ledger on any domain (R13). A summaries-derived ledger exists before F3 for D17 at F2.
30. **L10 argv.** For the GPU verbs it carries the sbatch resource flags before the script name, from one table (`SBATCH_RESOURCES`).
31. **DA forecast.** `stage_forecast` is added to inc-da-reply/1, as H11's scoring requires.
32. **No scipy.** The beta quantiles are implemented in `estimate.py`.
33. **Partition.** [§8.2] says "RM-shared for the rest", but the allocation refuses RM-shared ("Invalid qos", `run_inc_audit.sh`). Every verb runs on GPU-shared with one V100 (§5.6.2). rl-b takes an H100-80, as `run_inc_plan.sh` does for the same model.

## Runner notes: G-loop

Choices made while building `funnel/__main__.py`, `run_inc_funnel.sh`, `inc/realloop.py` (recovered mode) and `inc/pilot.py` (recovery provenance), where §5.6 left room. Each is covered by the tests named at the end.

**CLI (`funnel/__main__.py`).**
1. **Argument passing.** A parameter named `prereg` receives the loaded `domain.Prereg` (from `domain.load_pair`), `prereg_path` its path, `domain` the loaded `Domain`, and `adapter` the module `adapters.load(domain.adapter)` returns. `--force`, `--quiet` and `--testing` are passed only to a function with a parameter of that name. `--force` given to a function without a `force` parameter is refused, never dropped. A function with a `prereg` or `domain` keyword that the call does not bind (for example `fetch.write_manifest`, `recover.arms`, `draw.draw`) receives the CLI's loaded objects, so every header records the `--prereg` the CLI was given.
2. **estimate.** `estimate.evaluate` writes `audit_v1.json` and `audit_v1.md` itself, each time with a fresh `built_utc`, and raises `EstimateError` after writing an invalid audit. The CLI keeps the §1.2 rerun rule: when the rewritten audit equals the one on disk with the volatile keys removed, both files get their previous bytes back, so `recovery.json`, the realloop build and anything else that recorded the audit's sha256 do not go stale. An audit the evaluator returns without writing is written by the CLI. A changed audit, valid or not, replaces the file. Tested in `test_funnel_cli.py` with an evaluator that writes its own files, as the real one does.
3. **recover.** `recover.run` reads the other audit files from a funnel directory. `--audit` and `--maps` must therefore be `audit_v1.json` and `class_maps.json` in one directory. That directory is passed as `funnel_dir`, and the two paths as `audit_path` and `maps_path`, since `recover.run` takes them. `--arms --realloop` takes an experiment name (under `INC_DIR`) or a path (it holds a separator), and either must hold `exp.json`.
4. **embed-judges.** `--stage embed` without `--nshards` builds shard 0 of 1. With `--nshards n` and no `--shard`, it builds every shard in turn. The KT7 feature files are made by `judges.score_all`, which embeds them when they are missing, not by the embed stage. `--refetch` embeds `<out>/refetch/crops_refetch.csv` (verify's `CROP_FIELDS`) into `emb_dinov2_refetch.npz` and `emb_<tag>_refetch.npz`. `<tag>_embedder` is the one `adapters.INTERFACE` function ending in `_embedder`, the same rule `judges.py` uses, so no engine file names the model.
5. **fetch.** The transport is passed as `None`, meaning fetch's own urllib default. The kinds run in the order taxonomy, known-items, cards, kt7, refetch, then `write_manifest`. `--what refetch` without `--slug` refetches every source in `sources.card_image_counts`. fetch refuses inside a Slurm job, because compute nodes have no network.
6. **census.** A missing taxonomy cache is refused in the L12 wording ("no resolution ... run fetch --what taxonomy (lever L12)"), so the autopilot's refusal map routes it to L12. `--summaries-only` passes no fit-info projection: the adapter reads `step1/verifier_fit_info.json` itself when it is present.
7. **rl-b.** The endpoint defaults to `$FUNNEL_OLLAMA_ENDPOINT`. The model defaults to, and must equal, the config's RL-B model. The sheet directories default to those of `sheets_v1` and `sheets_v1_cluster` that exist. `check_vision` runs before any request.
8. **GPU check.** A CUDA device is visible when `torch.cuda.is_available()` (or, without torch, `nvidia-smi -L` lists a GPU). rl-b (class `gpu_large`) counts as a GPU verb.
8a. **Real signatures.** Besides the recording fakes, `test_funnel_cli.py` runs every verb against the real engine modules and the real adapter, each function replaced by a recorder that keeps its real signature and binds each call to it. A keyword or positional argument the real function does not take fails the test.

**Job script (`run_inc_funnel.sh`).**
9. **Array jobs.** Only `embed-judges --stage embed` runs as an array (task i gets `--shard i` unless `--shard` is given). Any other verb in an array job is refused.
10. **Script identity.** The script that runs (`$0`) and an outer `$REPO/run_inc_funnel.sh`, when present, must equal the nested copy.
11. **Exit codes.** Refusals exit 2 (verb, array, GPU, contract, script drift). Environment failures exit 1 (conda, a missing module file, an import).
12. **Imports.** PIL is also checked for rl-b, which cuts panels from sheet images. recover (class cpu) also checks torch and transformers, because its H6 detector computes DINOv2 descriptors of every recovered image.
12a. **Code log.** `tools/mega_trainer.py` is logged with the other modules, because it defines the dHash every never-train and near-duplicate check uses.
13. **rl-b warm-up.** The warm-up loads the model at `num_ctx` 8192, `rl.OllamaClient`'s default, so the labelling requests do not reload it.
14. **Test hooks.** `INC_FUNNEL_REPO`, `INC_FUNNEL_CONDA_SH` and `INC_FUNNEL_DRY_RUN=1` (every check, then the command, and no server) exist for tests only, as `run_inc_build.sh`'s hooks do.

**realloop recovered mode (`inc/realloop.py`).**
15. **Freshness.** `load_overlay` re-hashes every input `recovery.json` records. It also refuses:
    - a guard record with a hit, an unhashable image or an H6 copy;
    - a changed source label;
    - a missing domain-dev record;
    - a quarantined source or a domain-dev key among the rows;
    - a pool row whose unmasked original is not verify's pool image (path, sha256, source), or does not hash as recorded.

    A production build refuses a `recovery.json` written by a testing recover. FETCH rows are checked, with the dHash of their file, but never drawn.
16. **Near-duplicate groups.** Groups are the rows' recorded `near_dup3`, checked against `pool_meta.jsonl` dHashes at 3 bits: rows within 3 bits must share a group, and a row without a group is refused. The other modes never let an increment replay a photograph that is in the base or in another increment (select's near-duplicate groups; UNVERIFIED's exclusion). The recovered mode keeps that rule and extends it to the H10d hold-out: a group is left out of every draw when one of its rows lies within 3 dHash bits of a base image of verify's pool or of a domain-dev image, or when its rows lie in more than one pool (drawing each pool's part would put one photograph in two increments). The base's train_core images need no check, since verify's cwd12_copy stage keeps every pool image more than 6 bits from them. The capacities, M and the draw use the rows that remain; `build_summary.json` "recovered" "excluded" records each left-out group with its reasons, pools and nearest match.
16a. **Gates and guard coverage.** A row of a drawn pool must name at least one stratum, and `recovery.json` must record every one of its strata's gates as passed. The guard record must cover exactly the rows of `recovered_pool.jsonl`: never-train and H6 each checked every unmasked original and every masked copy. The H6 detector does not run in the build, so its record is the only evidence, and a record that covers fewer images is refused.
17. **Substitution.** Only kept arms among AUTH, CLASS and VETO substitute; a dropped arm never does. A substituted step keeps its own seed text, `exp + "/" + step`.
18. **Flags.** `--no-truth` is refused in recovered mode, because the truth arm is the loop's measurement. `--n-verified` now defaults to `None` (meaning 6) in argparse and in `build()`, so an explicit value can be refused. The default build is unchanged: `DEFAULT_BUILD_DIGEST` in `tests/test_inc_realloop.py` passes.
19. **Manifest names.** Step manifests are `manifests/<step>.jsonl`, for example `REC-AUTH-1.jsonl`.

**build-baseline recovery provenance (`inc/pilot.py`).**
20. **Naming.** `recovery_provenance` takes an optional guard. A manifest is named when `step1_r1/arms/arms.json` records its sha256, or when it is `recovered_pool.jsonl` itself. Overlay rows must match `recovered_pool.jsonl` rows even in a named arm. Control rows (join labels on unmasked images) are vouched for by the arm's sha256, and so only while `arms.json` is current: it must be `funnel-arms/1`, every input its header records (`recovery.json`, `recovered_pool.jsonl`, the base, the realloop `exp.json`) must still hash as recorded, and a production build refuses one written by a testing run. Every input `recovery.json` records is re-hashed as well (the freshness chain of §7.3).
21. **Guards.** The unmasked-original guard covers every manifest row whose key is a `recovered_pool.jsonl` key. A manifest holding a domain-dev image is refused in every build, testing included, because the domain dev is never trained on. A production build also refuses a testing recovery.
22. **Domain-dev record.** `recovery.json`'s `domain_dev` record may name the rows file itself (`domain_dev.jsonl`, the §4.18 form) or the `domain_dev.json` document whose `rows` record names it (`recover.py`'s form, with `rows_sha256` beside the record). `pilot.domain_dev_record` reads both, checks every file's sha256 and `rows_sha256`, and realloop uses the same reader.

**Tests.**
23. §5.6.3 and §5.6.4 name `tests/test_inc_realloop_recovered.py` and `tests/test_inc_pilot_recovery.py`. Both are one file, `tests/test_inc_realloop_v2.py`. It builds on the Step 1 world of `tests/test_inc_realloop.py` (imported) plus a synthetic `step1_r1/` in the §4.18 format, because `tests/funnel_world.py` does not cover recovery.
23a. **§5.6.5 is `tests/test_funnel_pipeline.py`** (see Verification at the end of this file). It plants what the paragraph below says the plain world lacks, through `funnel_world.build_world(plant=...)`. As first written: **§5.6.5, `tests/test_funnel_e2e.py`, is not written.** `tests/funnel_world.py` paints every picture more than 12 dHash bits from every other, so it holds no provenance-disjoint pair at 7-10 bits. The H6 calibration then has an empty negative set and `leak` refuses (`LeakCalibrationError`), and every later step needs `leak_v1.json`. The world also has no KT7 photos, cards or upstream annotations for the fetch and map steps. The end-to-end test needs those planted in the world first.
24. The CLI tests are `tests/test_funnel_cli.py`, and the job-script tests are `tests/test_funnel_job_script.py`.

## Runner notes: G-vision

Choices made while building `funnel/embed.py`, `funnel/judges.py` and `funnel/leak.py`, where §5.3 left room. Each is covered by `tests/test_funnel_embed.py`, `tests/test_funnel_judges.py` or `tests/test_funnel_leak.py`.

**Features (`embed.py`).**
1. **Model and loader.** The model is `judges.features` of the domain config (weed.json: `facebook/dinov2-base`, CLS), loaded through `semisup_labeler._load_backbone` and embedded through `_embed_batch`, the transformers path the verifier and the curator tools already use. `timm` and `torch.hub` are not used. `LazyEmbedder` loads the model on first use: a no-op rerun loads nothing, and a job that forks its image workers first gets the model loaded after the fork, as `verify embed` does.
2. **Crop shards.** They use the verifier's own helpers (`_cut_task`, `_embed_images`, `_embed_safely`, `_shard_name`, `_save_npz`), so the images per shard, the crops and the NaN rows follow `verify embed`. The shard meta adds `"format": "funnel-crop-features/1"`. `embed.load` returns, besides X, every shard file's sha256 and their digest, which the judge score files record as their `features`.
3. **Other crop tables.** `verify.Crops` accepts only the sets core, copy and pool, so `embed.CropTable` reads any `CROP_FIELDS` table (KT7, refetch) and offers the same `row`/`images` interface. `embed_table` writes one npz (crop_ids, X, meta with the table's sha256 as `crops_sha256`).
4. **Whole-image view.** The descriptor view is: EXIF orientation applied, the shorter edge resized to 256 px (bicubic) in the worker, then the model's processor (resize, centre crop). The pre-resize keeps 30-megapixel field photographs from crossing process boundaries at full size. It is applied to evaluation, pool and positive images alike, and recorded as `detector.descriptor.view`.
5. **Descriptor files** (`funnel-image-descriptors/1`, all cluster-only): `leak_eval_desc.npz` and `emb_dinov2_images_<set>.npz` for the sets reference, pool, extra (base or increment rows that are not pool images) and `pos_<family>`. They hold `keys`, `X`, `H` (the eight dHash variants, "id" first), `hash_ok` and `extra` (augmentation parameters). They resume per chunk of 2,000 rows, and a current file is reused.

**Judges (`judges.py`).**
6. **Keys.** Bank and KT7 keys are read through `qualify.item_keys`, so an item of an independent set is compared per observation. The exclusion applies the rule of `qualify.shares` vectorised (any of the four kinds equal; a missing value never equal). The test checks it against `qualify.shares` item by item. Crop queries take their keys from `adapter.unit_keys`, which gets `domain` and `funnel_dir` when the adapter accepts them.
7. **J-knn1 exclusion.** The runner's rule is "own session for a reference crop, nothing for any other query". The query's provenance group is excluded too, so a copy of a reference photograph never sees its twin (its provenance key equals the twin's). Sessions come from the known-truth items. A bank entry without a session is refused.
8. **Label spaces.** A kNN judge's labels are the targets, plus "other" when any bank item is an attractor or other. A panel entry may pin this with `"labels": "targets" | "targets+other"`. Bank items of kind `non_object`, or without truth, are left out and counted.
9. **Per-cell cap.** It is read from any `<kt>_per_cell_max` key of the entry (weed.json: `kt5_per_cell_max` 300). A cell is (item source, the crop table's `src_name`). The seed text is `funnel/v1/<judge id without "j-", lower case>/<kt>/<cell>`, which for J-knn2 and KT5 is the pinned `funnel/v1/knn2/kt5/<cell>`.
10. **Empty bank sets.** A bank spec with no item (KT6 before H3a's exact part passes, contract §4.1) is recorded under `empty`, not refused. A bank with no usable entry refuses.
11. **J-zs prompts.** They cover every target, then every attractor (one per taxon), then every `not` taxon not yet prompted, then the config's non-object prompts. A `not` taxon has no common name in the config, so its taxon fills `{common}`. The scale is the text tower's `logit_scale`; its name and provenance are recorded.
12. **KT7 features.** `score_all` writes `emb_dinov2_kt7.npz`, and `emb_<tag>_kt7.npz` from the adapter's Step 1 embedder, when they are missing or stale. `<tag>` comes from the one `adapters.INTERFACE` function ending in `_embedder` (the weed file is `emb_bioclip_kt7.npz`), the same rule the CLI uses, so no engine file names the model. That embedder's name must equal the Step 1 shards' embedder. A missing KT7 table refuses and names `fetch --what kt7`.
13. **Score files.** `J1__kt7.npz` also holds `cos` (the probe's prototype cosines), so verdicts can be re-derived. `judges/index.json` (`funnel-judge-index/1`) lists every score file with its sha256, and each bank's sha256, size, drop counts, empty specs and seed texts. A score file's meta without its volatile keys is its identity: the same identity is a no-op, and any other refuses unless `--force`. The identity compares the prereg by its `core_sha256` and the contract and config by their sha256, so the sample-lock amendment leaves the score files current (§3.3).
    - **Feature records.** Every `inputs` record either names a file that `check_records` can re-hash or has `path: null`. The DINOv2 crop shards are a directory, so their record has `path: null`, `dir`, every shard file's sha256 under `files`, and their digest as `sha256`. The adapter's Step 1 features have no single file, so their record has `path: null`, `what: "adapter.step1_features"`, and a `sha256` over the crop table's sha256, the embedder name and the feature values themselves (`judges.array_sha256`). Re-embedded Step 1 features are therefore a changed input even under the same embedder name.
    - **Query keys.** A kNN score file's meta carries `query_keys_sha256`, a digest of the query-side disjointness keys (`adapter.unit_keys` for crops, `qualify.item_keys` for KT7 photos). The bank's keys were already in the bank's sha256. A change in the 3-bit groups, labs or provenance of the queries therefore refuses a rerun without `--force`, instead of keeping stale exclusions.
    - **KT7 rows.** A KT7 item without a row in `kt7/crops_kt7.csv` refuses with `JudgeError`, naming `fetch --what kt7`.

**Copy detector (`leak.py`).**
14. **dHash variants.** They are computed from two thumbnails, of the image and of its transpose, and flips of those. The result equals `common.dhash` of the PNG of each transform, because Lanczos resampling commutes with flips and the transpose fixes the order of the two resampling passes; the test checks the equality on 60 images. The "id" variant is of the stored pixels, as in the never-train index.
15. **Augmentation parameters.** weed.json writes `max_frac`, `max_delta`, `max_deg`, `radius`, `quality` and `size`. The detector accepts those keys, its canonical keys (`frac`, `factor`, `degrees`, `modes`, `angles`, `fill`), or `range` for a scalar range. An unknown key or a missing family refuses. The families behave as follows:
    - crop removes a U(range) share of each dimension, split between the two sides at a uniform ratio;
    - shear is an affine shear about the centre, on the x or y axis at even odds, with a black fill;
    - letterbox fits the image inside a black 640 × 640 square, Roboflow's "fit (black edges)";
    - jpeg re-encodes at an integer quality in the range and decodes the result.

    Augmentations apply to the EXIF-oriented image, as a re-export would. The sample seed is `funnel/v1/leak/pos/<family>`. Each positive's parameters use the seed `funnel/v1/leak/pos/<family>/<reference key>`, so they do not depend on worker order.
16. **Recall is in-sample by construction.** θ is each family's lower order statistic at position floor(0.05 n), minimised over families. So at least 95 % of every family's positives pass on cosine alone. The binding tests of the calibration are therefore the false-positive rates, and any positive that cannot be described, which counts as a miss. If a family's quantile falls on such misses, θ is not finite and the calibration fails. Bounds come from `estimate.binom_interval`.
17. **Negatives.** In `[a, b]` of `negative_source_pairs`, b is the query side: its eight variants are compared with a's dHash, as a pool image is compared with an evaluation image. The 7–10-bit pairs are taken on the "id" dHash. A pair with an image that cannot be described is counted as `unscored`, not as a negative. The hard-pair seed is `funnel/v1/leak/neg` for the first configured pair and `funnel/v1/leak/neg/<i>` for later ones. An empty negative set fails the calibration, because its rate is unmeasured.
18. **Reference and harvested rows.** Reference images are the rows of `C.manifest_path(sources.reference)`. A base or increment row is harvested when it is not a reference image by key, path or sha256. Base B's reference rows carry the manifest's dataset source, such as `cottonweeddet12/train`, not the split name. Increments are scanned for `INCREMENT_EXPS = ("realloop_v1",)` through `adapter.increment_rows`, under the names `realloop_v1:<step>`.
    - **An experiment's copy of its base.** `adapter.increment_rows` returns every top-level `manifests/*.jsonl` of the experiment, and realloop writes its copy of the base there (`manifests/base_B.jsonl`, the same bytes). Increments are drawn from harvested images only, so a manifest that holds any reference-split image is treated as a base, not an increment.
    - When its rows equal the current base's (key, image, sha256), it is skipped and listed in `h6b.same_as_base`. Otherwise its harvested rows are scanned as a base under `<exp>:<stem>`, and it joins `h6b.base_scans`, because H6(b) covers the base the experiment trained on.
    - Without this, a copy in B was also reported as a copy in an increment (`increment_copies["realloop_v1:base_B"]`).
19. **Scan records.** The runner's scan fields are kept. The extensions are:
    - `copies` counts scanned images with at least one copy, and `copy_pairs` counts the pairs;
    - `unscanned` counts rows whose descriptor or hashes failed; such a row is not cleared;
    - `splits_hit` lists the evaluation splits a set's copies hit;
    - H6(a) adds `unscanned` and `missing_from_pool`;
    - H6(b) adds `cleared` (false when any base or increment row could not be scanned), `splits_hit`, `base_scans` and `same_as_base` (note 18). `base_copy` is true when any base scan holds a copy.

    `quarantine` is every source with a copy in any scanned set, including a source outside the pool. `void_ood_arms_with` equals `quarantine`. Contract §6 H6(a) voids the ood numbers of every arm that holds a source with an augmented copy of any never-train image, whichever split that image belongs to. A copy of a dev image therefore voids them too.
20. **H6(c) exam labs.** The domain config maps no exam to a lab. The rule, in order:
    1. `leak.exam_labs` `{split: group}`, when the config gives it;
    2. a lab group that lists the split name, an exam row's `source`, or that source's dataset prefix;
    3. the reference split's group, when the exam rows come from the reference split's dataset.

    An ambiguous or unknown exam gets null and is listed in `unmapped_exams`. With today's weed.json, dev and test map to LuLab (rule 3) and imageweeds to NDSU (rule 2). ood22 and ood23 are unmapped: their rows' source `3seasonweeddet10/data202x` is not a group member, and the group names the AgML copy instead. `leak.exam_labs` = {"dev": "LuLab", "test": "LuLab", "ood22": "LuLab", "ood23": "LuLab", "imageweeds": "NDSU"} in weed.json (G-data) would record contract §2.5's provenance explicitly.
21. **detect.** `detect` accepts `leak_v1.json` or its calibration record. It refuses under a failed calibration (`LeakCalibrationError`), and it refuses an image it cannot describe (`LeakError`, fail closed). `eval_index` refuses when any evaluation image cannot be described.
    - **Matching the calibration.** Given the whole `leak_v1.json`, `detect` also refuses an evaluation index whose embedder is not `detector.descriptor.embedder`. When both are on disk, it also refuses one whose descriptor file is not the `eval_descriptors.sha256` recorded there. The threshold is a cosine of that model's descriptors, and `recover` builds its own embedder.
    - **Given descriptors.** An image passed with its own `desc` and `hashes` refuses when the descriptor is not finite, is zero, or has the wrong size, or when it does not have eight hashes. A NaN cosine compares false, so such an image would otherwise be cleared without being compared.
    - **Consumers do not overwrite.** Called without a store (as `recover` calls it), `eval_index` refuses an existing `leak_eval_desc.npz` made by another embedder or from other evaluation rows, instead of overwriting the file `leak_v1.json` records. Only `leak.run` rebuilds it.
22. **Pairs file.** A negative row has `set` "calibration:<a>|<b>", `key` the b image's key, `eval_split` a, and `eval_key` the a image's key. The pair sentinels of §4.8 can therefore be drawn from it without any evaluation image.
23. **Rerun identity.** The identity is the inputs, the row-set digests, the parameters and the code, plus the prereg's `core_sha256` and the contract and config sha256. The prereg's raw sha256 is not part of it, so the sample-lock amendment leaves `leak_v1.json` current (§3.3). A rerun after a failed calibration refuses again, without rescanning.

**Tests.**
24. `tests/test_funnel_leak.py` takes interval bounds from `estimate.binom_interval`. When `funnel/estimate.py` is absent from the tree, it installs a test-only Wilson stand-in and prints a note saying so.
25. The three tests add the tree to `sys.path` only when the package is not importable already, so a `PYTHONPATH` copy (the mutation harness) is the code under test. With the M7 condition set to False in such a copy, `tests/test_funnel_judges.py` fails 9 checks.
26. **Mutation checks.** Each of the following was switched off in a package copy, and the owning test failed:
    - the dHash half of the copy rule, and the flip and rotation variants in the scan;
    - the temperature in the kNN weights;
    - the empty-negative-set gate;
    - the count of unscanned rows;
    - eval_index's refusal of an evaluation image it cannot describe;
    - the coverage check in `embed.load`;
    - the embedder in the shard, chunk and descriptor-file identities;
    - the table sha256 in `load_table`;
    - each fix recorded in notes 13 (identity, feature records, query keys, KT7 rows), 18, 19, 21 and 23.

    One mutation is equivalent and needs no test. Removing the recall gate changes nothing, because the threshold makes recall at least 0.95 by construction (note 16).

## Runner notes: G-labels

Choices made in `funnel/sheets.py`, `rl.py`, `qualify.py` and `recover.py` where this runner was silent or ambiguous. Each follows the contract where it speaks.

**Disjointness and qualification (`qualify.py`).**
1. **Independent items are compared per observation.** Every item of an `independent` set carries the same `source` ("kt7") and `lab`, so a literal reading of "shares" would make every KT7 item share with every other. `qualify.item_keys` suffixes `source` and `lab` with the observation (the unit id up to its last "/"). Two photos of one observation share; two observations do not. `shares` itself stays a plain comparison of the four kinds.
2. **A set's declared sources are material.** Wherever a known-truth set is used (a labeller scope, or a judge's qualification items and bank), its config-declared `sources` and their labs count as its material too. A copy set's labels are its reference twins' labels, so it belongs to the reference lab. This is what makes reference-lab strata get scope `KT7` alone, as §4.15 and §9 item 5 state.
3. **Bounds.** A qualification estimate is Wilson for n ≥ 30 and exact Clopper–Pearson below, both from `estimate.py`. Clopper–Pearson is more conservative than Jeffreys, and a qualification is a gate. n = 0 gives [0, 1].
4. **Trials.**
   - One trial per item and level: a positive is correct when the answer equals the truth, and a negative is correct when the answer is none of the item's confusions.
   - Negatives are an attractor's `confused_with` and a target's `siblings`.
   - Unsure, invalid and unparsed answers are wrong.
   - At genus level, genera that hold no target collapse to "other", and a genus answer in `genus_answer_unsure_for` is unsure (§5.1).
   - The plant level needs non-object sentinels; without them its Sp has n = 0 and it does not qualify.
5. **A judge needs attractor negatives of the required set.** A type is qualified only when at least one attractor negative of its set was scored: the independent set for `other_named`, and otherwise a claimed set not claimed by a hypothesis of that type. A judge scored on sibling negatives alone (for example J-zs, which never sees KT7) does not qualify for `other_named`. The rl-kind judge (J-vlm, scored from its backend's sentinel answers in `rl_qualification.json`) follows the same rule, and records `n_attractor_negatives` per type. `TYPE_HYPOTHESES["other_noinfo"]` includes H3a, because G3 holds the card-resolved units H3a tests.
6. **Scope of the negative sets.** `other_noinfo` keeps KT5 negatives as §4.15 pins, although it also serves G4's no-information frame, an H4 stratum, for which contract §4.3 names KT7. G4 is Horvitz–Thompson in `estimate.py`, so no judge predicts there. For R-J on a G4 (other_ok) box, the voters must be qualified for both `other_noinfo` and `other_named`, so they are qualified against the independent set's attractors as §4.3 requires for H4 strata.
7. **Rescue sets.** Contract §4.3 measures rescue "on J1's errors on KT2 and KT7". `qualify.rescue_set_ids` reads this from the config: the `qualify_rl_on` sets that are `independent` or list "rescue" in `allowed_uses`. For weed that is KT2 and KT7; KT3 is left out. `judge_qualification.json` also reports each judge's agreement with the claimed labels (`agreement_only`, KT4–KT6), apart from any qualification (§4.3).
8. **The step-1 probe's calls** come from three places:
   - the ledger's out-of-fold values for pool boxes;
   - the fitted probe on the Step 1 features for reference and copy crops (in-sample for reference crops; used only for rescue, P(wrong | J1 wrong) and φ);
   - `judges/J1__kt7.npz` for independent photos.
9. **H7 predictions** are read from each judge's kind: zero-shot fails; kNN with `holdout` "session" fails; kNN with `holdout` "disjoint" qualifies. This is the prereg's H7 sentence, without naming judges in code.
10. **Lab scopes and the material file.**
    - `judge_qualification.json` records, per judge, `by_type` over all items and `by_lab_scope["not:<lab>"]`, with that lab's items and sets left out.
    - The near_dup3 and provenance of a judge's material are stored as hashes, as §4.15 pins. The full lists are in the cluster-only `judge_material_v1.json`, and every material record names that file in `lists`.
    - `allowed_judges` accepts plain materials or judge entries. For an entry it picks the lab scope that leaves out the lab of a single-lab stratum. It resolves hashes through `lists`, and refuses a hash it cannot resolve.
    - The step-1 probe's material, in both files, carries `judges_nothing: true`. Its reject class was fitted on pool boxes, and it never judges its own discards (§4.3). So `allowed_judges` and `qualified_judges` never list it, whichever form the material is passed in.
    - `material_for` and `qualified_judges` expose the same choice to strata, estimate and R-J.
11. **The labeller's primary.**
    - The primary is chosen per hypothesis over the union of that hypothesis's strata's keys, at the needed level "species". That union's scope need not be any single stratum's scope, so each hypothesis scope is qualified in its own right. Otherwise a hypothesis whose strata exclude different sets would have no labeller.
    - When species does not qualify, the primary is the finest qualified level, with `demoted: true`; when no level qualifies, `backend` is null with the reason.
    - Hypothesis-to-group mapping is the §5.1.6 table. H1 takes the G2 strata of `authoritative` sources. H3a takes the G3 units of sources whose card resolver has a `class_table`; H3b takes the other G3 units.
    - Se and Sp are also given per target genus (`by_genus`: the sentinels whose truth is a target of the genus, or an attractor confused with one). Each primary lists `genera_not_qualified_at_species`, the input to the §10 stop rule that bounds such genera. Using it is up to `estimate.py`.
    - When the sample lock records `frames_sha256`, `rl` refuses frames that do not hash to it.
    - `rl_qualification.json` also carries `levels`, `strata_scopes` and `hypothesis_strata`.

**Sheets (`sheets.py`).**
12. **Board material.** A set that is not independent contributes all of its items' keys to a board's material, and an independent set contributes its drawn exemplars. This reads "B1 unless the item shares with KT1" literally: reference-lab items, reference crops and copies go to B2. An item is refused as a board exemplar (`check_not_exemplars`), and an item that fits no board is refused (`DisjointnessError`). So is a pool item whose four keys cannot all be found: it would share with no board and could sit beside its own exemplars.
13. **Sentinels across boards.**
    - A sentinel that fits one board goes there.
    - A sentinel that fits several goes to the board whose sentinel need, 3 × ceil(items / 12), is largest.
    - Surplus sentinels fill empty slots first, then sentinel-only sheets.
    - A shortfall is recorded, not refused, because the sample is locked. It is recorded in `sheets_v1_key/packing.json` (cluster-only) with each sheet's expected prevalence, so no shipped file reveals the G0 share.
14. **Prevalence.**
    - An item's expected target value is its stratum's share of frame units whose J1 prediction is a target. A sentinel or identity item takes its truth, and a G0 item takes the planted share.
    - A G4 item is swapped for the next G1, else G2, item while the sheet stays below `min_prevalence` even with every sentinel slot a target.
    - Target sentinels then go first to each sheet's need (1 to 3), and the rest are spread in a seeded order in proportion to supply.
15. **Pair units** are read from the right, `p:<source>|<x>|<split>|<y>`, so a calibration pair's source (`calibration:<a>|<b>`, G-vision note 22) keeps its "|".
    - A is a reference, copy or pool key, or a path relative to the source's registry directory.
    - B is in an evaluation or reference split, the kept pool twin ("dup"), or a pool or copy key.
16. **The sheet `question`** is the config's sheet prompt with the options filled, the same on every sheet. Pair tiles are named `a` and `b`.

**Answers and gold (`rl.py`).**
17. **Parsing.** A reply parses when its first line after any `<think>` block is the config's line format ("7, YES"), or when it is one JSON object `{"option": n, "box": "YES|NO|NA"}` (a ``` fence is allowed). Nothing else parses. The prompt text stays in the config.
18. **Answer files.**
    - Every raw reply is kept: the last in `raw`, earlier ones in `raw_history` (an extension of funnel-rl-answers/1).
    - An item that the model answered, but not parsably, within `max_attempts` is `unparsed`. An item whose every request failed (HTTP or transport) stops `rl-b` with `RLError` before its sheet's file is written. A server outage is not a labeller's answer, and a rerun resumes at that sheet.
    - `answers_from_text` writes the same format for a whole-sheet reply (RL-A, a person). An unanswered or doubly answered position is `unparsed`.
    - `rl_answers/RL-B/run.json` holds the run's header.
19. **Validation.** An RL-A or RL-B labeller must name exactly the configured model. An RL-A model is also refused when its configured family is judged, or its name contains a judged family. An RL-B file must record the model digest.
20. **Gold.**
    - `ingest` refuses everything when any answer file is invalid, so nothing is ingested partially.
    - `ingest` needs `sample_v1.csv`, and refuses one that is not the sample the sheet indexes and the sample lock record. The gold rows' disjointness keys come from it, and a labeller scope computed without them would ignore near-duplicate and provenance sharing.
    - `gold_v1.csv` appends `pair_truth`, `truth_taxon`, `source`, `lab`, `near_dup3` and `provenance` to the pinned columns. Pair qualification needs `pair_truth`; scopes and genus labels need the rest.
    - `gold_v1.json` records the header, `gold_sha256`, the key sha256 and the items not in the key.

**Recovery (`recover.py`).**
21. **Gates are per audit stratum, and each is read from the estimate of its own quantity.** An audit stratum is the finest unit the audit estimates. The contract's "per (source, label)" is at least as coarse, and a G2 stratum is (source, label, fail mode).
    - Each gate reads the audit `strata` entry of one event, in `estimate.py`'s vocabulary:
      - the box gate of R-A and R-V reads `label` (the source label is right and the box is valid);
      - the box gate of R-J reads `pred` (J1's predicted class is right and the box is valid);
      - a G3 unit's class gate reads `purity:<class>`;
      - R-C's box gate reads `label:<class>` (answered as <class>, with the box valid).
    - `<class>` is the unit's card map, or its scientific synonym (`unit_classes`). A unit with no class, or with two, fails.
    - A stratum whose audit holds no entry of the needed event fails the gate, and `why` names the missing estimate. The audit's per-stratum `target` share never stands in for a gate.
    - Two entries of one event for one stratum are refused.
    - The precision used is the smaller point estimate and lower bound over the stated estimate and both unsure assignments.
    - Rogan–Gladen must be applied and unflagged.
    - The level must be species, or genus for a genus-rank class.
    - A G3 unit's class gate is its purity lower bound ≥ 0.8. R-T needs the class gate; R-C needs both gates. A map is accepted only on gates read for its own class.
22. **One policy per image**, in the order VETO > AUTH > CLASS (R-C, then R-T) > JUDGE. R-T images join the CLASS pool.
    - An image is recovered only when at least one of its boxes is.
    - A target box that is not recovered is masked: its gate failed, its identity check failed, it is small, or its answer is genus-unsure.
23. **R-A, other-class boxes.** A no-information other-class box that J1 calls a target is masked, since YOLO has no ignore region. A named one stays the other class under the sibling guard.
24. **R-J voters.** A judge without a crop score file (the rl-kind judge) does not vote. A relabel needs every other qualified judge allowed for the box to agree.
    - The agreed target must equal J1's predicted class for the box, because the box gate (`pred`) measured the RL's agreement with that class and covers no other.
    - Masking listens to every judge with a crop score file, qualified or not: the contract masks a box that "any judge calls a target".
    - The identity check is read from the config. A class whose `identity_checks` entry names R-A in `before` is recovered under R-A only when `rl_qualification.json` records the check as passed; a missing record masks the class.
25. **R-F** needs `refetch/chain_v1.json`, a record defined here:
    - `images`: [{`key`, `image`, `sha256`, `source`, `label_boxes` (INC ids, after the class map), `masked_boxes`, `near_dup3`, `dhash`, `strata`, `checks`: {`never_train`, `core_copy`, `exact_dup`, `h6`, `embedded`}}];
    - every check must be true.
    - A refetched image is then recovered as R-C. That needs H3a supported, and every target box's class must be the class of an accepted card map of the source whose unit passed its gates. An image that fails is listed in `refusals`, as are images of a quarantined or not-recoverable source.
    - `_ctl` refuses a row without a join label, so a refetched image cannot enter a control arm until it has one.
26. **Licences.** A source without a licence in the config is refused and listed in `refusals` (DEC-7), never guessed.
27. **H6 copies stop the run.** A copy found by the detector among recovered rows stops F9 like a never-train hit (`NeverTrainHit`, recovery.json "refused"). The default detector is `leak.detect` over the cached evaluation index, with `embed.default_embedder`.
    - `recover` refuses before planning when `leak_v1.json`'s calibration is not `ok`. It also refuses when `h6b` is missing, records an incident, or records a base copy (§10 stop rules).
    - Written files:
      - A rerun with other inputs and no `--force` is refused before `recovered_pool.jsonl`, `domain_dev.*` or `recovery.json` is written. The content-addressed overlay files are the only writes before that point.
      - A guard hit replaces a complete `recovery.json` with the "refused" record only under `--force`.
28. **H10d** holds out, per recovered source, consecutive whole groups sorted by first key from a seeded start, until `min_group_images` is reached. A source with fewer than twice that many non-base images has no hold-out, and the reason is recorded. A group that holds a base-B image is never held out, because B would then train on near-duplicates of the domain dev.
29. **Arms.**
    - When the CLI passes no prereg, `arms` takes it from recovery.json and checks its core.
    - The experiment must be a recovered-mode build of this recovery.json: its `step1.increment_sources.overlay.recovery_sha256` must equal the file's sha256. Each control step's manifest must hash to its recorded `manifest_sha256`.
    - A missing REC-CLASS-1 or REC-JUDGE-1 step is listed under `not_built`.
    - U's cap is drawn by largest remainder over pools (ties by pool name), with a seeded permutation per pool (`funnel/v1/arms/U/<pool>`).

**Tests.**
30. The four tests use their own synthetic worlds (`tests/funnel_labels_fixtures.py`), in the formats of §4 and verify, rather than `tests/funnel_world.py`. They add the tree to `sys.path` after any `PYTHONPATH` entry, so a mutation-harness copy is the code under test. With the marked condition set to False in such a copy, M2, M3 and M6 fail `test_funnel_qualify.py`, and M4, M8 and M9 fail `test_funnel_recover.py`.
31. **Rules covered by a failing mutation.** Beyond M2–M9, each of these rules was switched off in a package copy, and the owner test then failed:
    - `sheets`: the G4 cap per sheet; the prevalence swap (on a config with `min_prevalence` 0.3, since at 0.2 three target sentinels always reach it); the exemplar refusal; the keyless-item refusal.
    - `rl`: the board sha256; the configured-model check; the RL-B digest; the transport-failure stop; the sample check in `ingest`.
    - `qualify`: the hypothesis-scope qualification; `by_genus`; J1 never allowed; the rescue sets; J-vlm's attractor negatives; `agreement_only`; the frames lock.
    - `recover`: every gate event; the minimum over unsure assignments; Rogan–Gladen applied and unflagged; `level_ok`; the not-recoverable source (the test config gives that source a licence); R-J's predicted-class rule, masking by any judge and G4 voters; the identity record; the leak and H6(b) preconditions; R-F as R-C; the rerun and refusal write order; the H10d base exclusion; the arms overlay and join-label checks.

## Runner notes: G-autopilot

**Test files.**
1. **Names.** The autopilot's funnel tests are `tests/test_funnel_ap_replay.py` (the §5.5.10 cases, named `test_inc_ap_funnel.py` above), `tests/test_funnel_ap_mutations.py` (§7.6, named `test_funnel_mutations.py` above), `tests/test_funnel_domain_free.py` (§7.5) and `tests/test_funnel_ap_units.py`. The units file holds the panel checks named `test_inc_ap_panel.py` above, plus the checks listed in its docstring. `executor.REPLAY_SCRIPTS` and `GOVERNANCE_FILES` name these files, and the M1 and M10 killing test is `test_funnel_ap_replay.py`.
2. **Mutation criterion.**
    - The harness copies the package, `tests/` and the job scripts into a temporary tree laid out like the repository. `docs/`, `RESEARCH_LOG.md` and `results/` are symlinked, read only. `PYTHONPATH` points at the copy.
    - A mutation counts as killed only when the killing script exits non-zero *and* reports more failures than it does on the unmutated copy, or crashes before its closing line. Without the second condition, a script that already fails for an unrelated reason would "kill" every mutation.
    - A killing script that does not reach its closing line on the unmutated copy fails the harness.
3. **Existing tests extended, not replaced.**
    - `test_inc_ap_levers.py` now covers the L10–L14 and L11a rows, X10–X12, the lab-hook rule for a null argv, the `funnel_walltime` estimator, and the L2 `--increment-sources` enum (= `realloop.INCREMENT_SOURCE_MODES`).
    - `test_inc_ap_governance.py` gives the replay runner stand-in funnel, mutation and domain-free scripts.
    - `test_inc_ap_fixtures.py` leaves out the files pinned by `funnel/MANIFEST.json`.

**Test blindness (§5.5.8).**
4. **The exam lists.** `model.exam_splits()` reads `funnel.domain`: the decision exam must be dev, or the call refuses. `model.non_dev_exams()` holds the non-decision exams plus the extra ones (weed: test, ood22, ood23, imageweeds, domain_dev), refuses an empty list, and carries M10.
    - `evidence.NON_DEV_EXAMS` and `remote.NON_DEV_EXAMS` keep the four non-decision exams their existing tests pin.
    - The blocked lists (`evidence.BLOCKED_SPLITS`, `remote.BLOCKED_SPLITS`, `brain_plan.FORBIDDEN_EXAMS`, `validate.leak_text_re()`) take the full `non_dev_exams()`.
    - The weed values are unchanged. A vehicles config's exam is refused through the `domain` argument (R13).

**Thresholds and signals.**
5. **`checked_against`.** The new `thresholds.json` values that transcribe the pre-registration or `funnel.claims` say `checked_against`, not `mirrors`, because `test_inc_ap_diagnose.py` requires every `mirrors` value to equal a constant of a pinned INC module. `test_funnel_ap_units.py` checks each `checked_against` value.
6. **Stage roles.** S1, S3, S4, D18 and D19 read ledger stages by their funnel-ledger/1 `role` (`target_check`, `other_check`, `image_rule`), never by stage id. The D18 policy follows from the role: `target_check` → R-A, `image_rule` → R-V, an uninformative space → R-C when `class_maps.json` has a card+geometry proposal for the source and R-J otherwise, the other class → R-J.
7. **S2 source share.** The sources S2 names for L11 and L12 are those whose embedded boxes are more than half uninformative (`S2_source_share_min` 0.5). This transcribes the contract's §3.4 wording; the contract gives no number for it.
8. **D17's "needs".** When S4 holds, D17's detail carries `needs`, the known-truth gap: an out-of-domain known-truth set. This is what R9b checks.

**Levers and data movement.**
9. **L10's command.**
    - L10's argv puts the verb's extra sbatch flags (`{*sbatch_resources}`, from protocol `funnel_sbatch_resources[funnel_verb_class[verb]]`, which mirror `funnel/__main__.py`) before `run_inc_funnel.sh`, as §5.6.1 writes the command.
    - The cluster's `remote.py submit funnel` adds the flags itself from the CLI's table. `executor.params_from_argv` and `argv_check` read only the tokens after the script.
    - L10 takes no `--part`: map runs as L11.
10. **Preconditions.**
    - Two L10 stages wait on a file: census needs `funnel/taxonomy_cache.json` (L12 first), and estimate needs `funnel/prospective_da.json` (OP_DA first).
    - `levers.json` records these as `preconditions`. A proposal whose file is not on the cluster carries `waits_for`, and the campaign logs it as not taken until the file is there.
    - The cluster's files are known from `funnel/files.json`, a listing (path, sha256, bytes, never content) that `remote.py funnel summary` ships as a derived artifact. The same listing tells which L10 stage is next.
11. **Lab hooks.**
    - `inc_funnel_fetch` (L11a and L12, R0) runs the funnel CLI's fetch on the lab.
    - `inc_verify_queue` (L14, R2) appends one task per `class_maps.json` `to_L14` entry to `known_truth/human_verify_queue.jsonl`, once per task.
    - `inc_funnel_sync` (OP_FUNNEL_SYNC, R0) rsyncs the fixed §6.3 list and then runs `fetch.check_manifest` over ssh. It uses the tick's one ssh connection, and the campaign queues it when the lab holds a listed file the cluster does not.
12. **The summaries-derived ledger.**
    - When the cluster has no `funnel_ledger.json`, `remote.py funnel summary --derive-ledger` builds the §F2a ledger in a temporary directory and ships it as `derived.funnel_ledger`. Nothing under INC_DIR is written.
    - `remote.py funnel ledger-summaries --write` writes it once, and never over a census-derived ledger.
    - `evidence.from_snapshot` marks this ledger derived.
13. **D18's realloop_v2.** D18 proposes L13 for each stratum that passes. It proposes the L2 `realloop build --increment-sources recovered --step1-overlay <INC_DIR>/step1_r1/` (priced over the 6 unclean steps) only once `funnel/recovery.json` (shipped from `step1_r1/`) reports status complete. Until then that L2 is listed as D18's next step.

**Claims and the devil's advocate.**
14. **Claims into the evidence.**
    - The claims register is the lab file `<campaign dir>/claims.json`. It reaches the evidence as the artifact `campaign/claims.json`.
    - The campaign passes it as `context["claims"]`, which the evidence builder removes from the context, so the existing evidence call signatures are unchanged.
15. **Actor strings.**
    - `policy._ACTOR_RE` does not allow a second ":", so the adversary's actor is `tier2:adversary/<model>` (`brain_plan.actor_for`). The reply record, the campaign ledger, the claims history and the track record all use it.
    - A family that cannot be read counts as the same family: the reply is recorded and moves no claim.
16. **Blind markers.** The DA digest's blind markers include the sha256 of both R14 replies (`da/da_positive.json` and `da/da_sycophantic.json`).
17. **The DA job.** It is the planner's job with `PLAN_ROLE=adversary`. The executor's plan segment passes the role as an optional eighth argument, and names the job `inc_da_<campaign>_<n>`.

**Fixtures.**
18. **Contents beyond §5.5.10.**
    - The Step 1 `admit_summary.json` and `select_summary.json` are copied too, because `ledger_from_summaries` reads them.
    - The derived ledger is stored as `funnel_ledger_summaries.json`, and the synthetic audits are under `audits/`. No fixture has a name `evidence.load_dir` reads from `INC_DIR/funnel/`, so a fixture directory is never read as an INC tree by accident.
    - `tests/funnel_ap_fixtures.py --check` rebuilds every fixture byte for byte.
19. **Pending.** `step1/verifier_fit_info.json` is the cluster-side projection of `step1/verifier/fit_info.json`, which is not local. MANIFEST.json lists it under `pending`, and S6 stays not evaluated until the funnel summary ships it.


## Runner notes: G-stats

Choices made in `__init__.py`, `domain.py`, `ledger.py`, `strata.py`, `draw.py`, `estimate.py` and `claims.py` where this file or the contract left room. Tests: `tests/test_funnel_{init,domain,ledger,claims,strata,draw,estimate}.py` on the shared helper `tests/funnel_stats_world.py`.

**Domain config and prereg.**
1. **Grep terms.** `Domain.terms()` returns only what the config names. The fixed cross-domain list of §7.5 ("cwd12", "otherplant", ...) lives in the grep test, because an engine file cannot contain it. Two names are left out of the substring terms: a known-truth set's pseudo-source named after the set ("kt7", the file vocabulary `kt7/`, `crops_kt7.csv`, `J1__kt7.npz`), and the reference source, whose name is part of pinned field names (census `train_core_boxes_per_class`, the prereg's screening keys).
2. **Options.** A target without an `option` text is shown as "<common> (<taxon>)", or "<common> (<taxon> spp.)" at genus rank. Every option carries `answer`: the class name, `other` (attractors and the first tail option), `non_object`, `invalid` or `unsure`. `options_tail` must hold four texts in that order, and `pair_options` four in the order same, consecutive, different, unsure; the schema refuses other lengths.
3. **Schema.** Required top-level keys: format, domain, adapter, classes, stages, exams, sources, known_truth. The rest are optional, and unknown keys are refused. A guard stage whose `recoverable` is anything but `false` is refused (M5), and so is a source listed in two lab groups. `$FUNNEL_DOMAINS_DIR` points name lookups at another directory (tests with synthetic domains).
4. **Amendments.** An amendment needs `id`, `kind` and `date` (YYYY-MM-DD). `append_amendment` refuses a repeated id, a second `sample_lock`, and an amendment whose `prereg_core_sha256` is not the file's current core (the file was edited outside its amendments). It writes `json.dumps(indent=1)`, the committed file's own format, so a lock is a one-list diff.

**Ledger.**
5. **Extra keys.** A funnel-ledger/1 also carries `input_records` (the path records behind `inputs`, for the freshness check) and `identity_inputs` (so `validate` recomputes the fingerprint). Label-space kinds must sum to the source's boxes.
6. **`unaudited_dependencies`** returns (stage, dependency) pairs whose dependency is `recoverable: true` and has no audit; on the §4.3 table this includes (S12, S8), and a pair drops out once its dependency carries an audit. **`attach_audit`** refuses an invalid audit and one whose `ledger_fingerprint` is another ledger's.
7. **Unit rows.** `validate_unit_row(row, stage_order=None, vocab=None)`: the engine cannot spell the §4.2 path vocabulary (it contains a domain term), so the adapter passes its own vocabulary and stage order.

**Frames and the draw.**
8. **Allocation.** When the floors alone exceed n, the floor is lowered until they fit. `allocate_weighted` splits by weights under caps; it serves the sentinel split, where a weight is not a cap.
9. **G1.** A frame smaller than its half of the non-excluded budget passes the rest to the other frame.
10. **G3.** A no-name class is clustered only when it has at least 100 boxes. Clusters under 100 boxes, and rows whose features are not finite, are listed in `reported`. Features are L2-normalised before KMeans. When `draw` is called without features it reads the DINOv2 shards through `embed.load` against the adapter's crop table, on first use only, and records the shard digest as the input `emb_dinov2`.
11. **G4.** The uniform part of a frame is seeded `funnel/v1/G4/uniform/frame=<frame>`. The Poisson part takes one uniform per unit, in unit-id order, from the generator seeded `funnel/v1/G4/prio`. The excluded frame is reached by the Poisson part only.
12. **G5.** `guard_pairs_v1.csv` is read as the adapter writes it (`pair_id`, `kind`, `source`, `split`, `bits`, `sha_differs`, ...). Lab comes from the config when the file has no `lab` column, and `near_dup3` and `provenance` are empty when absent. Exact-dup twins whose sha256 differs take what the per-source allocations leave of the planned n, so "all_sha_differs" holds only when that fits.
13. **Inclusion across groups.** G2v with G2a is the one joint design: a unit in both frames is listed once, under the first group that drew it, with the union π, and the estimate reads frame membership from the frames. A box that is also in a G3 unit (or G1/G4 and G3) gets one row per group with that group's own design π, because each group is estimated on its own design; §4.9's "the same pi (the union)" is applied only where estimates pool the two frames.
14. **Sentinels.** 3 × ceil(n/12) for the n pool rows drawn (G0, G1, G2, G2v∪G2a, G3, identity, G4), split over sets by `sentinel_weights` and within a set by truth kind. Pair sentinels are 3 × ceil(n_G5/12), split by the `pair_sentinels` ratio, in group `pair_sentinel`, strata `pair_sentinel/kind=positive|negative`. Positives are copy ↔ twin pairs of KT2, id `p:<slug>|<copy key>|<reference source>|<twin key>` (twin from `provenance` `prov:<key>`); negatives are `leak_pairs_v1.csv` rows whose `kind` starts with "neg". Units already drawn in any group are not sentinels.
15. **Exemplar sessions.** Per board option n: the option's items sorted, their sessions sorted, the first two of a permutation seeded `funnel/v1/board/<board>/<n>` (the rule `sheets.board_plans` applies; the strata test checks they agree). An item is excluded from sentinels and the identity check when its session is one its own class's exemplars come from, not every exemplar session of every class.
16. **G0.** The in-domain positives are the hidden targets of the in-domain set: KT2 items whose crop the adapter's crop table labels with the other class (contract §5.2, runner §4.8), not every copy box. The frames refuse an adapter without a crop table, and the crop table's sha256 is an input of the frames. The sample shows `unit_id` `G0:<item_id>`, empty source, key, lab, π and kt, seed text `funnel/v1/G0`, and a draw rank from a permutation seeded `funnel/v1/G0/order`. The key's G0 rows hold `g0` = {part, seed_text, part_rank, pi}; the planted record holds the realised share (`share`), the drawn s (`share_drawn`) and the part counts. **Key truth** is written for G0, sentinel, identity and pair-sentinel items only; an estimation item's claimed label is what the audit tests.
17. **Reruns.** Before the lock, a draw over frames made from other inputs refuses unless `force`. After the lock, a rerun is a no-op when the lock's prereg core is the prereg's, the locked files hash as locked, and every recorded input and the known-truth digest are unchanged; otherwise `SampleLocked`, `force` included. The minimum of a group is checked twice before anything is drawn: the frame, and what the design can draw from it (`draw.planned_sample`: the planned n of srs strata, the first-stage units' n for G3, the expected union size for G4). A G3 frame of a few large units can hold more boxes than the minimum and still yield fewer items (at most 30 per unit).

**Estimates.**
18. **Events.** Box strata: the answer is a target at the level used and `box_ok` is yes. H1: the answer is the source label and `box_ok` is yes (also "label right" alone and "box valid" alone). Purity (H3a, class gates) ignores `box_ok`. G0: target identification, ignoring `box_ok` (the photo is the box). `unsure`, `unparsed`, and a genus-rank answer in a `genus_answer_unsure_for` genus are unsure. At genus level only genus-rank classes count as targets; at plant level no answer establishes a target.
19. **Rogan–Gladen placement.** Each stratum is corrected with its own scope's Se and Sp (`rl_qualification.json strata_scopes`). An aggregate is Korn–Graubard over the uncorrected strata, then corrected once with the hypothesis's primary scope; a stage aggregate uses the sets every one of its strata keeps, and with none left it stays uncorrected (flag `no_scope`). A flagged correction (Se + Sp − 1 < 0.7) reports the design interval.
20. **Uncovered strata.** A stratum with no labelled unit widens an aggregate over its weight w: [lo(1 − w), hi(1 − w) + w].
21. **Stratum records.** `estimate` is the unsure-as-no value and `interval` the envelope of both assignments; both are in `unsure`.
22. **PPI.** Only in srs strata with at least 30 labelled units. The predictor is the judge in the stratum's frame `allowed_judges` that qualifies for the stratum type in the lab scope `qualify.material_for` picks, best `precision_at_half.lb + rescue.lb`; f is the judge's summed target probability (the label's probability for label events). Its variance is λ²Var(f)/N + Var(y − λf)/n.
23. **Funnel recall.** The admitted side is the reference-lab admitted target boxes (ledger.jsonl) at the verified precision of `step1/calibration.json` (the block that holds `current_join`), plus each other source's admitted boxes at its G2a precision. The discard side is G1, G2 and G2v (beta-binomial per stratum) and G4 (N × p, Horvitz–Thompson). The estimate is the Monte Carlo median, so it lies inside its interval; the plug-in ratio is reported beside it as `plug_in`.
24. **Screening.** The Holm p uses the aggregate's effective n and its corrected FN estimate. `by_class` screens each class with R_min 100 below `small_class_train_core_boxes_below` train_core boxes (census) and 500 otherwise. Stage aggregates also report a source-clustered design effect.
25. **H9 grid.** The card join applies `class_maps.json` card+geometry proposals whose G3 unit passes the class gate, and only when H3a is supported; the taxonomy join adds `name_status_v2` target_synonym names (scientific or override) that pass the class gate. Evidence "on" drops boxes whose ledger evidence outcome is `not_evidenced`. Shift thresholds are Learn-then-Test per lab and class (level 0.95, δ 0.025, Bonferroni over 20 score-quantile candidates) on other labs' G2 gold (label right; unsure counts as wrong) and on the independent set's J1 scores when `judges/J1__<set>.npz` exists; the prototype gate is dropped under shift. A stratum's precision is "label right and box valid", unsure as wrong for the lower bound and as right for the upper, with no Rogan–Gladen. Reference-lab verified boxes outside every frame take the calibration precision. A box whose label a card or taxonomy map changed takes the precision of its G3 class unit, the unit the map's class gate was read on; its G1 or G4 stratum pools it with boxes the map does not touch. H9′'s best cell has the largest lower bound; it is supported when that lower bound is ≥ 4,098 and falsified when that cell's upper bound is below 3,073.5 (prereg "best cell UB<3073.5"); the largest upper bound of any cell is reported beside it. Without the verifier thresholds the grid is not evaluated.
26. **H11** scores log-loss against uniform over the ledger's recoverable stages; the stage ranking leaves out stages without a box count (S0 counts images). Stages whose recoverable lower bounds tie are one outcome, because they are estimated from the same frames (S7 and S7b both read G3): the top-1 is right when it names any of them, and the log-loss is taken on the forecast's mass on the tied set against the uniform mass |set| / K. A forecast that is not a probability over the recoverable stages summing to 1 (tolerance 1e-6, as the DA validator), or a DA record without one (an invalid reply), is falsified with the reason.
27. **Stops.** H0 falsified stops at F8, and so does any H0 verdict other than supported (F8 needs H0 to pass).
28. **Verdicts.** H6 and H7 are `reported` (H7 with each judge's prediction scored as a part); H8 and H10 are `not_evaluated` here.
29. **Freshness.** `estimate` re-hashes the frames' recorded inputs, `rl_qualification.json`'s and `judge_qualification.json`'s, and the gold against `rl_qualification.json gold_sha256`. Every artifact a step reads must record the current prereg core (runner §3.3, `funnel.check_prereg_core`): `strata.frames` checks `census_v1.json` and `funnel_ledger.json`; `draw.load_sample`, `draw.load_key` and a locked rerun check the lock's `prereg_core_sha256`; `estimate` checks the frames, `rl_qualification.json`, `judge_qualification.json` and `census_v1.json`. A prereg edited outside its amendments after an artifact was made refuses (`StaleInput`, or `SampleLocked` for a locked draw), so no verdict is read against thresholds other than the pre-registered ones.
30. **R12.** An overlap is recorded when a stratum's own scope, or the primary scope of a hypothesis it serves, holds a set whose sentinel sample rows share a key with it (the independent set's keys read per observation, `qualify.item_keys`), when a scope holds a claimed set, and when the PPI predictor's material shares a key with the stratum. `evaluate` writes the audit with `valid: false`, then raises `EstimateError`.
31. **`d18_inputs`** also carry `source`, the stratum's single source (null when it spans several).
32. **Contract §5.3.** Its first column is the exact (Clopper–Pearson) one-sided 95 % bound; the test checks it with `clopper_pearson(0, n, 0.90)` and checks `binom_interval` against the Wilson columns.

**Claims.**
33. `transition(..., actor_kind, proof=None)`: the adversary's proof is its validation record (`valid`, `surviving`, `model`, `planner_model`, optional `same_family`); the autopilot's is the audit record (`audit_sha256`, `valid`, `fingerprint_match`, `d18_fired`). A person signs `human:<actor>`. `model_family` refuses a name it cannot read, so such a reply moves no claim. A missing register file loads as an empty register.

**Corrections after review (2026-09-28).** Each is covered by the named test; the mutation that reverts it fails that test.
34. **Genus stop rule** (contract §10 stop rules, §5.1). A hypothesis about a genus is `bounded` when its primary labeller is not qualified at species level for that genus in its scope, read from `rl_qualification.json by_genus` (a genus with no record is not qualified). H1 is about every label of its strata; H3a about the species-level card class (the genus-rank class needs genus level only). `test_funnel_estimate.py` "genus stop rule".
35. **H5b frame.** H5b reads the G5 strata whose split is an exam split (decision or non-decision) only: the near-evaluation pairs. The exact-dup twins and reference-copy pairs G5 also holds are reported apart (`parts.other_guard_pairs`), because pooling them, which are nearly always the same photograph, would lift the share the hypothesis tests. `test_funnel_estimate.py` "H5b reads near-evaluation pairs only".
36. **Labeller per stratum.** G1's named strata are answered by H2b's primary labeller in the stage screening, the D18 inputs, funnel recall and label frequency, as in the strata records (`Context.hyp_of`). A stage aggregate whose strata have different labellers or levels takes no single labeller's Se and Sp: it stays uncorrected with the flag `mixed_labellers`.
37. **Label frequency.** Each sampled item counts once (the joint G2v/G2a design lists a unit once, and a stratum of either frame reads it). The reference lab's verified admitted boxes are in no frame; they enter a source's target-labelled mass at the in-domain verified precision, and without that precision the source gets no estimate, with the reason.
38. **Screening by class.** A class missing from `census_v1.json train_core_boxes_per_class` gets no R_min (`r_min` and `screen` null, with the reason) instead of the 500 default.


## Runner notes: G-data

Choices made while building `adapters/__init__.py`, `adapters/inc_step1.py`, `taxonomy.py`, `names.py`, `relation.py`, `fetch.py`, `domains/weed.json` and `tests/funnel_world.py`, where §3–§5.2 left room. The tests named at the end cover each.

**Config (`weed.json`).**
1. **Nested keys beyond §3.1.** `domain.validate` refuses unknown top-level keys only, so these nested keys are used:
   - `names.non_object_keys` and `names.state_keys` match the whole name key, while `*_words` match substrings. verify matches its short object and disease words (car, pot, tag, rot, rust, mold, ...) as whole keys, and a substring "rot" would catch "rotundus". "hut", "shed" and "pave" are whole keys for the same reason. The six lists together hold every word of `verify.GENERIC_NAME_KEYS`, `NON_PLANT_WORDS` and `CWD12_RELATED_TOKENS` exactly once. "novel" is in `generic_keys`.
   - `names.query_rules`: separators and a leading numeric token for authority queries (note 9).
   - `taxonomy.authority.match_params` and `search_params`: the GBIF parameters checked on 2026-09-28 (`strict=true`; `qField=VERNACULAR`, `limit=20`, and the backbone `datasetKey`).
   - `sources.card_resolvers.<slug>`: `names` (name key → taxon, for a card that names species without class ids: greenhouse), `refetch_images` (R-F), and fetch specs with `provider` (url, huggingface, mendeley, zenodo), `dataset`/`version`/`repo`/`record`, `files` or `files_regex` and `folder_regex`, and for annotations `format` (voc, yolo, voc+yolo), `classes_file`, `voc_dir`, `yolo_dir` and `frame_name_regex`.
   - `sources.known_items` (H12: index specs, slug templates, box words, the rule) and `sources.relation_checks` (the H0(b) and H0(c) sources, so the engine never names them).
2. **Attractors.** The runner's ten, plus cotton (*Gossypium hirsutum*) and common bean (*Phaseolus vulgaris*), the two MorningGlory attractors of contract §4.2. `confused_with` is the target J1 called confidently on boxes named for that taxon in census_v0, or the class-policy siblings when the pool has no such boxes. The option list has 28 entries.
3. **Card tables.** MH-Weed16 is PMC12179629 Table 2 (16 classes). weed_crop is PMC11986624 Table 2 (13 ids and names) with the taxa of its data table. greenhouse (PMC11599996) names the species but gives no id table, so it has `names`. All were read on 2026-09-28. A card taxon is matched to a pool class by **name**, never by id, because which id a card row means is what H1-pre and H3a test. MH-Weed16's ids are numbers, so it has no card taxa until H3a passes.
4. **Not added: `leak.exam_labs`.** G-vision proposes `{"dev": "LuLab", "test": "LuLab", "ood22": "LuLab", "ood23": "LuLab", "imageweeds": "NDSU"}` (their note 20). Contract §2.5 supports it. However, `tests/test_funnel_leak.py` asserts that ood22 and ood23 are unmapped under today's config, so the key must land in the same change as that test.
5. **`capture_stem_regex` is empty.** L11a records the stems. Observed while building the parser (archive `intel Real Sense Depth_Annotations.zip`, sha256 fb3a14c3…, 6,656 files): the PASCAL_VOC `<filename>` fields name 240 distinct `.mp4` files (`VID_…mp4_<frame>.png`), and the YOLO ids 0–14 carry the Table 2 names. `run_geometry` records this as `stems` through `frame_name_regex`. Whether it becomes a provenance rule (§7 R-C) is left to the platform.

**Adapter (`inc_step1.py`).**
6. **Discard paths.**
   - S10 counts as failed for a box only when another box of its image blocks the image. A box that blocks its own image fails its own stage only (S6, S8 or S9).
   - S12 is a source-level criterion, so it is recorded for every unit outside base B whose source has no verified box, whether or not the image was admitted. This is what gives the runner's planted case two failed stages: a box vetoed in an image of a non-evidenced source.
   - `PATH_VOCAB` extends §4.2:
     - S2 holds verify's own reasons (including unhashable and calibration_only_not_train_core);
     - a pre-pool row has "n/a" for the stages it never reached;
     - S6 has "no_size", S8 and S9 have "failed";
     - S11 is "base", "increment_pool", select's status (no_evidence, below_gate, no_feature, dropped_other_budget, dropped_cap) or "n/a";
     - S12 is "n/a" for base B.
7. **Blockers and fail codes** follow §4.1. A lost veto box is attributed to its highest-priority blocker (`by_blocker`), and all its blockers are listed (`combinations`). The contract's 457 images and 564/276/145/4 blockers are parsed from the contract text and recorded, not asserted.
8. **Contract numbers are read, not typed.**
   - `contract_numbers(path)` parses the Step 1 counts §1–§3 state, by anchored phrases.
   - `stage_counts(step1, census_v0)` re-derives them. On the local files, all 64 re-derivable counts equal the contract's.
   - The census uses the parsed S7b and veto numbers for its contract checks.
   - `aggregate_rows`, `rows_v0_view` and `rows_from_step1_files` are the census's own row aggregation, reused by the census_v0 reproduction test. `pred_confident` counts the argmax over verified and conflict crops, and `mean_p` is the mean argmax probability per argmax class, rounded to 3 decimals, as census_v0 has them.
9. **Name status v2 on the real names.** Through the recorded GBIF answers:
   - BroWeed and NarWeed are unresolvable (197,613 boxes);
   - crop and Crop are role names (30,042);
   - the object names total 2,149 and the state names 1,403, equal to the contract's numbers.
   - The v2 no-information frame holds 422,464 of 545,318 boxes, against the contract's post hoc 417,717. The difference is seven names GBIF does not resolve as written: Blackbean 2,667, parthenium hysterophorous 1,352, Arachius 345, spurredanoda 261, phhyllanthus 113, mushroom 6, and "grass weeds - v2 release" 3.
   - **Blackbean** is a measured MorningGlory attractor (§4.2), and in the no-information frame it enters H2a and R-J candidacy. A project override (`taxonomy.overrides`, R4) such as "blackbean" → *Phaseolus vulgaris* would move it to the named frame. None is added here, because overrides are a person's decision.
   - Query variants: separators become spaces; a leading numeric token is dropped ("0 ridderzuring" → *Rumex obtusifolius*); "Genus sp./spp." also queries the genus.
10. **Known truth.**
    - KT4 and KT5 hold embedded boxes only: a small or size-less box has no crop, and the judges' banks and the sheets need one (KT4 is then the contract's 4,639 embedded boxes).
    - KT5 = OtherPlant boxes whose card taxon (by name) or v2 resolution (scientific, override or vernacular) is an attractor or "not" taxon; a genus attractor takes any species of its genus. Accepted names come through the resolver, so *Conyza canadensis* meets *Erigeron canadensis*. A census KT5 row whose class no longer resolves to such a taxon (the config or the cache changed after census) refuses, and so does a name status without its taxonomy cache.
    - KT6 = boxes of the ids a confirmed alignment maps from card rows whose taxon is a target synonym, once `relation_geometry_v1.json` records the H3a exact part passed. It is split by `near_dup3` group (the first half of a seeded permutation is calibration).
    - KT7 `crop_id` comes from `kt7/crops_kt7.csv` by item id.
11. **Disjointness keys.**
    - `near_dup3` groups are computed once per process over train_core, the copies, the pool and the KT7 photos, from their dHashes, with names `n:<smallest member key>`.
    - A pre-pool image takes the group of a member within 3 bits, else its own.
    - A KT7 photo's provenance is its observation, `prov:kt7|<obs>`.
    - Class, cluster and source units get `near_dup3` and `provenance` null.
12. **Extra adapter functions** used by other modules: `source_labels` (the source's own class ids with image sizes, for geometry), `relation_units` (copy class units with twin truth, current and old join), `pool_class_units`, `kt5_taxa` and `verifier_fit_info_projection`.
13. **`ledger_from_summaries`.**
    - Without an explicit projection, it reads `step1/verifier_fit_info.json`, else computes the projection from `step1/verifier/fit_info.json`, else leaves `reject_class` null.
    - It refuses before reading anything when the target file is census-derived.
    - The image flow of S2–S5 follows verify's real order: read, near_eval, copy, then exact_dup.

**Relation (`relation.py`).**
14. **Geometry.**
    - A pool image pairs with the upstream file that matches all its boxes; the upstream may hold more boxes (a dropped class).
    - The tolerance is 2 px per corner in the upstream frame, plus half the last written decimal of a normalised upstream. NDSU writes 2 decimals and drops trailing zeros, so the step comes from the file's finest decimal.
    - Alignments the matched ids cannot tell apart are tied. A tie goes first to the one keeping the source's class count, then to the one moving the fewest ids, then to the order identity, plus1, minus1, drop0…. For MH-Weed16, drop15 is therefore preferred to identity when the 16th class never appears.
    - On the real archive with a planted drop3 export, drop3 is chosen and confirmed.
15. **H1-pre** passes when the pool's class names agree with the upstream names on ≥ `id_agreement_min` (0.99) of matched boxes and ≥ `geometry_match_min` (0.95) of pool boxes match. These are H3a's thresholds, since the contract gives none for H1-pre. The confusion pairs are recorded.
16. **Class maps.** A card map goes `to_L14` when its alignment is unconfirmed, keeps a class count other than the source's, or when the card table's names differ from the class names the upstream annotations carry under the same ids (`card_names` in the geometry record; an upstream without names cannot be compared and also goes to L14). The last rule is contract §8.5 L11 ("accepted when the card and the geometry match agree"): a card's ids are read as upstream ids, and only the names show that they are. KT6 takes card truth under the same condition. Taxonomy proposals come from `name_status_v2` target_synonym names, so `map --part geometry` refuses before the census. `class_maps.json` is refused without `--force` when its proposals would change; the refusal comes before any write, and a rerun with the same inputs rewrites neither file (so `class_maps.json`'s record of the geometry file stays current).

**Fetch (`fetch.py`).**
17. **prereg.** The CLI passes no prereg to fetch functions, so the header's prereg defaults to `<out>/prereg_v1.json`, and `known_items` takes its domain from it. The CLI now also binds its loaded prereg (G-loop note 1).
18. **Cards.**
    - Card files are recorded relative to `cards/`. A Mendeley file must match the listing's sha256, and a Zenodo file its md5.
    - An HTTP error of a card or paper is recorded, not raised.
    - A Roboflow class list without `$ROBOFLOW_API_KEY` is recorded as refused and listed in `refused`. Keys are redacted from every recorded URL.
19. **KT7.**
    - One photo per observation: the first with an allowed licence, at `photo_size` "large".
    - iNaturalist has no seedling annotation. "Vegetative" is Plant Phenology (term 12) = No Evidence of Flowering (value 21). Places are the United States (1) and India (6681).
    - A live query for *Amaranthus palmeri* on 2026-09-28 with these filters counted 9 observations (1,626 without the phenology filter). Several taxa will have fewer than 30 (contract §11).
    - `kt7/kt7_index.json` records the query, rule and counts under the funnel header.
20. **Machine-local tables.** `kt7/crops_kt7.csv` and `refetch/crops_refetch.csv` hold absolute image paths, because `verify._cut_task` opens the path as written. The manifest leaves them out, and `check_manifest` rebuilds them after the hashes pass.
21. **H12 items.**
    - A candidate must name a target (class name, common name or taxon, as whole words) and show boxes (a box word at a word start, or the index's detection task).
    - Candidates come from:
      - notes of the configured topic that mention a dataset;
      - the entries of the surveys a survey note links on GitHub, named by their dataset link;
      - the configured index (`task_class_index` format).
    - Slug patterns come from the config's URL templates, from the index's slug template and, for notes and index entries, from the name key.

**Tests.**
22. `tests/test_funnel_adapter.py` (the runner's `test_funnel_adapter_step1.py`), `test_funnel_names.py`, `test_funnel_taxonomy.py`, `test_funnel_relation.py`, `test_funnel_fetch.py` and `test_funnel_weed_config.py`. `tests/funnel_world.py` builds its world by running verify (pool through admit) and `select.build` on painted pictures, so the files are verify's own. `tests/fixtures/funnel/gbif_recording.json` holds real GBIF answers recorded on 2026-09-28, trimmed to the fields the resolver reads, with each body's original sha256. `funnel_world.replay_transport` replays them.

**Review of the G-data build (2026-09-28).** Each item was confirmed on the code, fixed, and covered by a test that fails without the fix (checked by switching the fixed line back).
23. **The sample lock covers every census output.** The census refused a rerun after the lock only when `census_v1.json` existed; without it, a run rewrote the locked `name_status_v2.json`. It now refuses with `SampleLocked` whenever the lock is set and the census is not current (`test_funnel_adapter.py`).
24. **H0(b) and H0(c) fail closed.** H0(b)'s "maps every class" and "does not flag the current join" passed when their sources had no class of ≥ 20 boxes to judge. An untested check now fails and names the untested sources. H0(c) passes only when the source has at least one class with a name-given truth. H0(c) now also fails when any taxon-resolved named class (not only a same-genus relative) is mapped to a target; the contract's own example is a crab grass. Its class accuracy counts only classes whose names give a truth (targets, resolved non-targets, role names) and lists the others (`test_funnel_relation.py`).
25. **Name status v2: an override resolves a name.** A project override to a non-target taxon fell through to `unresolvable` (no-information frame), because `override` is mappable but was not in `informative_via`. A mappable resolution is now informative, so an override such as "blackbean" → *Phaseolus vulgaris* puts the name in the named frame, as note 9 intends (`test_funnel_names.py`).
26. **Disjointness keys agree between writers.** `unit_keys` computed a pool unit's provenance without its image file name, so with a capture-stem rule it would differ from the census rows'. It now reads the pool image. A pre-pool reference copy (`d:`/`p:` ids) now takes its twin's provenance (`prov:<train_core key>`, §1.3), and `guard_pairs_v1.csv` carries `lab`, `near_dup3` and `provenance` columns, which the G5 frame reads (`strata._g5`) and which were missing (`test_funnel_adapter.py`).
27. **Guard pairs refuse instead of guessing.** A missing evaluation manifest, or a near_eval hit that no manifest lists, wrote a pair with an empty evaluation side. Both now raise `AdapterError` (`test_funnel_adapter.py`).
28. **Fail code with no threshold.** A class whose τ is NaN (never confident in `verify.verdicts`) was coded `cos_below_sigma`; it is `p_below_tau` (`test_funnel_adapter.py`).
29. **Upstream class files.** When an archive holds several `classes.txt` files that number the classes differently, one id → name table would misname one folder; `read_upstream` now refuses (`test_funnel_relation.py`).
30. **Input records.** `relation_audit_v1.json` records the H0(c) judge's score file and `ledger.jsonl`; `relation_geometry_v1.json` records `pool_meta.jsonl`; `class_maps.json` records `name_status_v2.json` and the taxonomy cache.
31. **Tests added where a behaviour could change unseen:** the blocker names, the census refusal on a failed reconciliation (written with `ok: false`), the state-before-object rule order, the related-token rule, a "not" taxon outside the target's genus (*Chamaecrista pumila*), EXACT-only scientific matches, a vernacular hit on a target species never mapping, and an alignment the second half does not confirm.
32. **Config taxa that are homonyms across kingdoms.** GBIF's strict match answers "Multiple equal matches" (match type NONE) for *Digitaria* and *Phaseolus*, which are also animal genera. The resolver left them unresolved, so the zero-shot prompts (`judges.prompts`, `lineage_string`) would refuse on the cluster at F5, and KT5 compared literal strings. A taxon the configuration names is now re-queried in the targets' kingdom (`kingdom=Plantae`, recorded in `tests/fixtures/funnel/gbif_recording.json`, `added`) and accepted when the answer is that very name at genus rank or below. Source class names never take this path, so no name status changes (the v2 no-information frame is still 422,464 boxes). `build_cache` now refuses when any configured taxon stays unresolved (`test_funnel_taxonomy.py`).
33. **KT7 truth is checked against each observation's own taxon.** iNaturalist's `taxon_name=Digitaria` (research grade) answered on 2026-09-28 with *Sorghastrum nutans*, *Cynodon dactylon* and *Bothriochloa ischaemum* among its first 30 results; species queries (*Amaranthus palmeri*, *Eleusine indica*) and the other genera returned only their own taxon. An observation is now kept only when its taxon is the queried one or a name below it, and each row records `observed_taxon`; the rest are counted as `skipped.taxon` (`test_funnel_fetch.py`).

## Verification

This section records how the six groups' modules were checked together after each group's own review: which tests ran and what they counted, the defects the integration found and how each is covered, what was mutation-checked, and what can only be checked on the cluster or the lab. All counts below are from the runs of 2026-09-28 on a local macOS workstation, not the lab server (Python 3.12, numpy, PIL, sklearn, joblib, torch and transformers present; open_clip absent; no GPU, no network).

### V.1 How the suites were run

Each script ran on its own, `cd weed_llm_benchmark && python3 tests/<file>.py` (pytest for `test_policy_adversarial.py`), eight at a time. A script passes when it exits 0; "checks" counts its `ok` lines (for pytest, its passed tests). The set is every `tests/test_inc_*.py` and `tests/test_funnel_*.py`, plus `test_species_docs.py`, `test_domain_config.py`, `test_model_router_placement.py`, and the four other scripts that read a module the groups changed: `test_policy.py` and `test_policy_adversarial.py` (`brain/policy_actions.json`), `test_poster_data.py` (`model_router.py`) and `test_species_robo.py` (`policy_actions.json`).

| Script | Exit | Checks passed | Failed | Skipped |
|---|---|---|---|---|
| `test_domain_config.py` | 0 | 53 | 0 | 0 |
| `test_funnel_adapter.py` | 0 | 128 | 0 | 0 |
| `test_funnel_ap_mutations.py` | 0 | 46 | 0 | 0 |
| `test_funnel_ap_replay.py` | 0 | 98 | 0 | 0 |
| `test_funnel_ap_units.py` | 0 | 158 | 0 | 0 |
| `test_funnel_claims.py` | 0 | 71 | 0 | 0 |
| `test_funnel_cli.py` | 0 | 67 | 0 | 0 |
| `test_funnel_domain.py` | 0 | 75 | 0 | 0 |
| `test_funnel_domain_free.py` | 0 | 9 | 0 | 0 |
| `test_funnel_draw.py` | 0 | 55 | 0 | 0 |
| `test_funnel_embed.py` | 0 | 43 | 0 | 0 |
| `test_funnel_estimate.py` | 0 | 246 | 0 | 0 |
| `test_funnel_fetch.py` | 0 | 49 | 0 | 0 |
| `test_funnel_init.py` | 0 | 74 | 0 | 0 |
| `test_funnel_job_script.py` | 0 | 31 | 0 | 0 |
| `test_funnel_judges.py` | 0 | 50 | 0 | 0 |
| `test_funnel_leak.py` | 0 | 60 | 0 | 0 |
| `test_funnel_ledger.py` | 0 | 56 | 0 | 0 |
| `test_funnel_names.py` | 0 | 48 | 0 | 0 |
| `test_funnel_pipeline.py` | 0 | 67 | 0 | 0 |
| `test_funnel_qualify.py` | 0 | 65 | 0 | 0 |
| `test_funnel_recover.py` | 0 | 72 | 0 | 0 |
| `test_funnel_relation.py` | 0 | 64 | 0 | 0 |
| `test_funnel_rl.py` | 0 | 57 | 0 | 0 |
| `test_funnel_sheets.py` | 0 | 55 | 0 | 0 |
| `test_funnel_strata.py` | 0 | 77 | 0 | 0 |
| `test_funnel_taxonomy.py` | 0 | 52 | 0 | 0 |
| `test_funnel_weed_config.py` | 0 | 43 | 0 | 0 |
| `test_inc_ap_brain.py` | 0 | 270 | 0 | 1 |
| `test_inc_ap_campaign.py` | 0 | 242 | 0 | 0 |
| `test_inc_ap_dashboard.py` | 0 | 21 | 0 | 0 |
| `test_inc_ap_diagnose.py` | 0 | 218 | 0 | 0 |
| `test_inc_ap_e2e.py` | 0 | 47 | 0 | 0 |
| `test_inc_ap_evidence.py` | 0 | 65 | 0 | 0 |
| `test_inc_ap_fixtures.py` | 0 | 99 | 0 | 0 |
| `test_inc_ap_governance.py` | 0 | 298 | 0 | 0 |
| `test_inc_ap_levers.py` | 0 | 280 | 0 | 0 |
| `test_inc_ap_remote.py` | 0 | 302 | 0 | 0 |
| `test_inc_ap_replay.py` | 0 | 185 | 0 | 2 |
| `test_inc_audit.py` | 0 | 81 | 0 | 0 |
| `test_inc_driver.py` | 0 | 351 | 0 | 0 |
| `test_inc_gate.py` | 0 | 140 | 0 | 0 |
| `test_inc_integration.py` | 0 | 37 | 0 | 0 |
| `test_inc_lora.py` | 0 | 65 | 0 | 0 |
| `test_inc_realloop.py` | 0 | 117 | 0 | 0 |
| `test_inc_realloop_v2.py` | 0 | 79 | 0 | 0 |
| `test_inc_relevance.py` | 0 | 64 | 0 | 0 |
| `test_inc_scorer.py` | 0 | 71 | 0 | 0 |
| `test_inc_select.py` | 0 | 139 | 0 | 0 |
| `test_inc_splits.py` | 0 | 85 | 0 | 0 |
| `test_inc_step0_fixes.py` | 0 | 99 | 0 | 0 |
| `test_inc_train.py` | 0 | 122 | 0 | 0 |
| `test_inc_verify.py` | 0 | 151 | 0 | 0 |
| `test_model_router_placement.py` | 0 | 28 | 0 | 0 |
| `test_policy.py` | 0 | 45 | 0 | 0 |
| `test_policy_adversarial.py` | 0 | 61 | 0 | 0 |
| `test_poster_data.py` | 1 | 32 | 1 | 0 |
| `test_species_docs.py` | 0 | 86 | 0 | 0 |
| `test_species_robo.py` | 0 | 106 | 0 | 0 |
| **59 scripts** | | **5955** | **1** | **3** |

The one failure is `test_poster_data.py` (V.6 item 1). The three skips are the existing ones: `test_inc_ap_brain.py` (a relevance digest waiting for a fixture) and `test_inc_ap_replay.py` (R2 and R4b's committed record). `tests/test_funnel_pipeline.py` was run once more on its own after its last edit: 67 of 67, 31 s.

### V.2 The end-to-end test, `tests/test_funnel_pipeline.py` (§5.6.5)

§5.6.5 names `tests/test_funnel_e2e.py`; the test is `tests/test_funnel_pipeline.py`. It replaces G-loop note 23a.

**Chain.** fetch taxonomy → census → (the realloop_v1 build) → fetch kt7 → the arrival check (`fetch.check_manifest`) → leak → (the fetched cards) → map geometry → embed-judges → qualify → map relation → draw → sheets → rl-b → (RL-A's answers) → ingest → qualify --rl → (the DA record) → estimate → recover → (the realloop_v2 build) → recover --arms. Every funnel verb runs through `funnel/__main__.py main` with `--testing`. The steps in parentheses are not funnel verbs.

**World.** `tests/funnel_world.py` runs Step 1's own code (verify pool, crops, embed, fit, calibrate and admit; select build) on painted pictures, each box painted in the colour of its true class. The test grows it through a new `plant=` hook (with `core_frames=` and `ctx.registry_extra`; defaults unchanged, and the other world users pass unchanged):
- 9 train_core pictures per session, and copies in the two reference-copy sources and `cottonweed_sp8` (slot ids 0-7), so that each holds one class of at least 20 boxes (H0(b));
- 80 vetoed images in weed_crop (R-V), 40 numeric-source images with a hidden Purslane (R-J);
- 30 vetoed images in greenhouse, a licensed source quarantined as a whole by a flipped dev copy that Step 1's plain dHash kept, and 30 hidden Purslane images in a source the config marks not recoverable. Their strata's gates pass, so only the guards keep them out;
- six calibration negatives for the copy detector (7-10 dHash bits from a train_core picture, recoloured).

The world has 393 pool images and 1,166 pool boxes; the sample fills 110 pool sheets and one pair sheet (RL-B gave 1,019 answers).

**Fakes.**
- The DINOv2 stand-in describes a 224-px crop tile by its centre colour and a whole image by a saturation-weighted chromaticity histogram. The histogram survives the eight H6 augmentation families; the leak calibration's cosine threshold was 0.896 in the final run, with no false hit on either negative set.
- The step-1 embedder reads the corner of a KT7 photo, so the probe errs on the independent photos and the judges' rescue can be measured.
- The web is the recorded GBIF answers plus a fake iNaturalist serving painted photos.
- The cards are the fetched layout (`cards/index.json` and the upstream annotations), built from the pool's label files.
- RL-B is a fake ollama server and RL-A a stub. Both answer from the painted colour of the crop tile they are shown, and nothing else.
- `prospective_da.json` is a synthetic fixture.
- The test's copy of `prereg_v1.json` has its sampling minimums set to 0, because draw refuses a frame below its minimum and the world's frames are far smaller than the real pool's.

**What it asserts** (67 checks, 31 s alone):
- The freshness chain of §7.3 over 19 artifacts (349 recorded inputs): each records this prereg's core, the contract and the domain config; every recorded input still hashes as recorded; every code record is the code that ran; 15 producer → consumer links each carry the producer's sha256; and the sample lock names the sample, key, frames and name status.
- The census reconciles, leak calibrates and quarantines the planted source whole, H0(b) passes, pairs stay on cluster sheets, and no pool sheet or board names a source, unit, stratum or verdict.
- RL-A answered pool sheets only, and the machine RL qualifies at species level in H0's scope from sentinels alone.
- The estimates cover the planted truth. H0 is supported, with the planted share inside its interval. Every labelled box stratum's interval (events target, label and pred) covers the share of its frame's units that the painted truth makes true: 32 estimates in the final run.
- The audit holds the events the recovery gates read. The autopilot's H11 scorer (`outcome.h11`) agrees with the estimator's. D18 reads the audit and proposes only policies that recovered rows. `remote.py funnel_summary` ships the aggregates with the audit's d18_inputs, stage ranking and verdicts unredacted, and no row-level file.
- Recovery respects every guard:
  - only strata whose gate passed recover, each gate Rogan–Gladen corrected and above the prereg's bounds;
  - no row from the quarantined source (not even its clean images) or from a not-recoverable source;
  - no domain-dev image, no base image and no non-pool image;
  - every unmasked original and every masked copy is clear of the never-train index, recomputed here;
  - every recovered label is the painted truth (R-V 62 Waterhemp and 62 OtherPlant boxes; R-J 30 Purslane);
  - every mask is one flat fill;
  - the step-1 labels are unchanged.
- realloop_v2 (built on the driver's FakeBackend) records the overlay's hashes and draws 6 unclean steps from the overlay's rows only. Build-baseline's `recovery_provenance` accepts every arm.
- census, draw, estimate and recover rerun as byte-for-byte no-ops.
- Unit regressions of each defect in V.3.

### V.3 Defects the integration found and fixed

Each fix is covered by a check in `tests/test_funnel_pipeline.py` that fails when the fix is reverted (V.4).

1. **sheets could not read the frames.** `strata.py` writes `pred` and `label` in `frames_v1/<group>.csv` as class names; `sheets.stratum_priors` parsed `pred` as an id and crashed (`ValueError: could not convert string to float: 'MorningGlory'`). The labels fixtures wrote ids, so the group's tests could not see it. Fix: `sheets._pred_class_id` takes a class name or a decimal id and refuses anything else (`SheetError`).
2. **Recovery could never pass a gate.** `recover.gates` reads each gate from the audit estimate of its own event ("label" for R-A and R-V, "pred" for R-J, "purity:<class>" and "label:<class>" for a class unit), but `estimate.py` wrote only each stratum's "target" (or "pair") estimate, so every gate failed with "the audit holds no 'label' estimate". This was open item 1 of the G-labels review. Fix: `audit_from_context` also writes, per stratum, the "label" estimate for G2 and G2v, "pred" for G1 and G4, and for a class unit "purity:<class>" and "label:<class>" of the class `recover.unit_classes` names; `Context.event` learns "label:<class>". `outcome._prediction_outcome` now reads a stratum's purity from its "target" record, not from whichever record comes first.
3. **An all-positive stratum got the interval [0, 1].** In `ht` (and `combine`) the Korn–Graubard effective n, θ(1 − θ)/v, is 0 when θ is 1 and v > 0, which happens in any HT stratum whose labelled units are all positive (the HT total can also pass 1 before clipping). The best case therefore failed every gate: in the world, the G2v stratum, 68 of 68 labelled boxes right, was recorded as [0, 1] with Rogan–Gladen flag `no_data`. Fix: `estimate.kg_n_eff` uses n when θ is 0 or 1, as Korn and Graubard (1998) do.
4. **A card-resolved class unit was answered by the wrong labeller.** `estimate.Context.hyp_of` sent every G3 stratum to H3b's primary labeller, while `qualify.hypothesis_strata` qualifies H3a's primary on the card-resolved units. So the audit read those units through another labeller, and through none when H3b has no stratum, in which case R-C could never pass. Fix: a G3 unit of a card-resolved source is H3a's. The test checks that every stratum `rl_qualification.json` assigns to H1, H2a, H2b, H3a, H3b or H4 is answered by that hypothesis.
5. **An input record that `check_records` cannot read.** `relation_geometry_v1.json` recorded a many-file upstream (one label file per image) as `{"path": <directory>, "sha256": <digest>}`. `check_records` re-hashes files only and would refuse it as missing. Fix: that record carries no path (`dir` and `what` say what the digest covers), and `cards/index.json`, which pins every file's sha256, is recorded as a file.

### V.4 Mutation checks

- **§7.6 harness** (`tests/test_funnel_ap_mutations.py`). M1–M10 are each killed by their owner test, with failures against 0 unmutated: M1 by `test_funnel_ap_replay.py` (6 failures), M2 by `test_funnel_qualify.py` (3), M3 by `test_funnel_qualify.py` (1), M4 by `test_funnel_recover.py` (2), M5 by `test_funnel_domain.py` (2), M6 by `test_funnel_qualify.py` (8), M7 by `test_funnel_judges.py` (9), M8 by `test_funnel_recover.py` (a crash), M9 by `test_funnel_recover.py` (1) and M10 by `test_funnel_ap_replay.py` (5).
- **Integration fixes.** Each of the seven code changes of V.3 (item 2 holds three) was reverted in a private copy of the package, and `tests/test_funnel_pipeline.py` was run against the copy: all seven fail it. The sheets fix, the gate events and the Korn–Graubard n fail it at a step (sheets exits 1; realloop_v2 finds no recovered image, or no arm with M images left). The "label:<class>" event, the stratum-to-hypothesis map, the directory input record and the DA purity row each fail their named check.
- **§7.6 markers against the pipeline.** With each marked condition of M2–M9 set to False, the pipeline test fails only for M9: the realloop_v2 build refuses the quarantined source's rows, a second barrier behind recover's. M2–M8 are killed by their owner tests only. The world holds no case that reaches them end to end: for example, no pool image lies within 6 bits of an evaluation image (M8), and no named-frame box is relabelled (M4).
- **The groups' own mutation checks** are in their notes above: G-vision note 26, G-labels notes 30–31, G-autopilot note 2, and the G-data and G-stats corrections.

### V.5 What the world does not exercise

- R-A: H1 is bounded. The world's sentinels qualify the machine RL at species level overall, but not per genus (the stop rule).
- R-C and R-T: no card-mapped or synonym class unit reaches 100 boxes, so H3a's purity is not evaluated. The "label:<class>" and "purity:<class>" events are covered by the unit regression only.
- R-F: optional; no refetch.
- H5b: 3 pair sentinels do not qualify RL-B. H12: no known-item list is fetched.
- `fetch --what cards`, `known-items` and `refetch` against a fake web: the cards are written in the fetched layout. The layout was read against `fetch.fetch_spec` (single archive: `cards/<slug>/<file>`; many files: `cards/<slug>/annotations/...`, both `what: annotations`).
- `census --summaries-only`: covered by `test_funnel_adapter.py` on the real local summaries.
- Training: realloop_v2 is built and initialised on the driver's FakeBackend, and no run is scored.

### V.6 Open items found by the integration, not fixed here

1. **`test_poster_data.py` fails one check:** "model_router declares the eight roles Figure 4 prints". `model_router.ROLES` now has nine roles: `adversary` (contract §8.2, G-autopilot). The poster is frozen, so whether the check should pin the eight roles the poster printed or the router's current count is a decision for its owner. Nothing in the funnel code can fix it without removing a role the contract requires.
2. **R-J masks every box it does not relabel.** `recover` masks a box when "any judge with crop scores calls a target" (contract §7 R-J, taken literally). J-knn1's label space holds targets only (§4.12), so it calls a target on every box, and every other box of an R-J image is masked (in the world, two per image). Counting only judges that have a non-target option would keep the images' OtherPlant boxes. This needs a decision on the contract's wording.
3. **Most recoverable sources have no licence.** `domains/weed.json sources.licences` has entries for weed_crop, greenhouse and mh_weed16 only, and `recover` refuses a source without one (DEC-7). No other source can contribute a recovered image until L11a records its licence.
4. **Name status from the recorded GBIF answers.** The recording in `tests/fixtures/funnel/gbif_recording.json` holds GBIF's answers of 2026-09-28. Under those answers, `names.status_v2` over census_v0 [post hoc] gives:
   - "Blackbean" (2,667 boxes in weed_crop and greenhouse; 333 OtherPlant conflicts, 321 of them confident MorningGlory calls) is `unresolvable`, which puts it in the no-information frame of H2a and G4 and makes it an R-J candidate. The cards name its taxon (*Phaseolus*).
   - Other names are also `unresolvable`: "parthenium hysterophorous" (1,352 boxes), "Arachius" (345), "spurredanoda" (261), "phhyllanthus" (113), "mushroom" (6) and "grass weeds - v2 release" (3).
   - Several crop names resolve vernacularly to another taxon: "corn" to *Agrostemma*, "cotton" to *Eriophorum*, "Kochia" to *Neokochia californica*, "Horseweed" to *Laennecia*, "Field Pea" to *Sphaerophysa* and "Lentil" to *Vicia*. The frame is right for those, since a resolved name is named, but the taxon is not. KT5 reads a card taxon first, so weed_crop's and greenhouse's Kochia stay KT5; imageweeds_aerial's "kochia" (565 boxes), which has no card resolver, does not.

   Name status is frozen at F6. Whether card taxa, or overrides (R4, `taxonomy.overrides`), decide these names before then is a decision for a person.
5. **Frame-file types.** `frames_v1/<group>.csv` holds class names in `label` and `pred` for box units and a class id in `label` for known-truth units (`strata._kt_unit`), and §4.7 does not say which. sheets now reads both, and estimate reads names.
6. **Control arms after a substitution.** When REC-CLASS-1 or REC-JUDGE-1 is drawn from another pool (§9.1 substitution), `CLASS_ctl` and `JUDGE_ctl` are same-image controls of that pool's images. `arms.json by_pool` records it, but the panel still names the comparison H10c-CLASS or H10c-JUDGE.

### V.7 Checks only the cluster (or the lab) can make

- **F3 census on the real Step 1.**
  - The reconciliation: 545,318 crops; verdicts 2,049 / 8,732 / 2,756 / 531,781; 1,060 admitted target boxes; 125,500 small boxes; the 989 veto losses attributed.
  - The J1 re-derivation reproduces `pool_verdicts.npz` for every crop.
  - The drawn S8b composition from `verifier/fit_info.json`, and the S1 quarantine reasons from the cluster registry.
  - The memory and time of streaming the full `ledger.jsonl`.
- **Lab fetches.**
  - The real GBIF query parameters and the frozen `name_status_v2.json` (V.6 item 4).
  - The real Mendeley, Hugging Face, PMC and Roboflow answers behind L11a (the Roboflow class lists need `ROBOFLOW_API_KEY`), and which class AgML dropped from MH-Weed16 (H3a's alignment).
  - The iNaturalist KT7 query (term 12, value 21; CC licences) and how many photos per taxon it yields, which decides whether any genus can qualify at species level.
  - `fetch.check_manifest` on arrival through `inc_funnel_sync`.
- **F4 leak.** DINOv2 (`facebook/dinov2-base`, offline Hugging Face cache on Bridges-2) over about 103 K pool images, the evaluation images and the synthetic positives: recall ≥ 0.95 per augmentation family and a false-positive rate ≤ 1 % on the real train_core × MH-Weed16 pairs; H6(a) over the 14 sources; H6(b) on the real B and realloop_v1's increments.
- **F5 judges.** DINOv2 over 545,318 crops (array job), the BioCLIP-2 text tower (open_clip, absent locally) for J-zs, and judge qualification on the real KT sets.
- **F7 RL-B.** `ollama` with `qwen3.8:27b` on an H100-80: vision capability (`/api/show`), memory, answer parse rate. Whether RL-A or RL-B qualifies per genus.
- **F8 and F9 on real data.** Estimate time with 20,000 Rogan–Gladen draws for each of about twice as many strata records as before (V.3 item 2). recover's DINOv2 descriptors of every masked and unmasked image, and PNG writes on Lustre.
- **F10.** `run_inc_build.sh` imports the outer package copy: the realloop and pilot changes reach it through lever X5 before the realloop_v2 build. Then the real driver on Slurm and the baselines.
- **The job script.** `run_inc_funnel.sh` on Bridges-2: the conda env `bench` imports, the GPU-shared partition for every verb, the per-job ollama port, the array shards of `embed-judges`, and the sbatch flags from `sbatch-args`.

## Decisions after F1 (2026-09-28, before F2)

These settle V.6 items 1, 2 and 4, and one inconsistency the full pipeline test exposed. They were made before any census, sample or label. Items 3, 5 and 6 of V.6 stand as written: recovery refuses a source without a recorded licence until L11a records it, the frame files keep names for box units, and the H10c arm names follow the pool actually drawn (`arms.json by_pool`).

| Id | Decision | Change | Test |
|---|---|---|---|
| F1-R1 | **Name overrides** (contract §4.3 J-taxon: project policy overrides the backbone, R4). Names that GBIF leaves unresolved although they name a plant (Blackbean, the misspellings "parthenium hysterophorous", "phhyllanthus" and "Arachius", "spurredanoda") would sit in the no-information frame, where R-J may relabel them. Crop and weed names whose vernacular answer is another taxon (corn, cotton, Field Pea, Flax, Horseweed, Kochia, Lentil, Canola, chilli, Soybean, swinecress) would carry a wrong taxon into the sibling guard. Both groups get an override with its reason. Crabgrass, nutsedge and cotton are overridden at genus rank, because the sources do not name the species. "mushroom" joins `names.non_object_keys`. This was decided after the GBIF answers were read, so it is [post hoc] with respect to them. The frozen name status (F3) records the overrides' sha256. | `funnel/domains/weed.json` `taxonomy.overrides` (19 entries), `names.non_object_keys`; GBIF answers for the override taxa recorded in `tests/fixtures/funnel/gbif_recording.json` (`FUNNEL_GBIF_RECORD=1` in `tests/funnel_world.py`, never set in a test run) | `test_funnel_taxonomy.py`: Kochia resolves by override to *Bassia scoparia*, and, with the override removed, only vernacularly to another genus. `test_funnel_names.py` and `test_funnel_adapter.py` pass on the recorded answers. |
| F1-R2 | **Which judges mask a box under R-J** (V.6 item 2). "Any judge calls a target" counts a judge whose label space holds a non-target option. A judge that can only answer with a target (J-knn1: a bank of target crops) names a target for every box, so its answer is not a call. It may still vote where it is qualified. J1's own confident target calls (a conflict) still mask. | `funnel/recover.py` `mask_judges_of`, used by `_apply`; `recovery.json` records `mask_judges` | `test_funnel_recover.py`: `mask_judges_of` drops a targets-only label space; an unqualified judge with a non-target option still masks. `test_funnel_pipeline.py`: J-knn1 is not a masking judge, and 62 of 88 recovered images carry a mask (the R-V ones). |
| F1-R3 | **D18 proposes a card map (R-C) only when H3a is supported.** Contract §8.4 applies a card map "when the card and the purity check agree"; otherwise the stratum goes to the judges (R-J). `recover.py` already refused R-C without H3a, so D18 could propose a recovery that recovery must refuse. | `inc_autopilot/diagnose.py` `d18` (`why_not_card_map` in the stratum record) | `test_funnel_ap_replay.py` R11: with a proposed card+geometry map, H3a missing → R-J, inconclusive → R-J, supported → R-C. `test_funnel_pipeline.py`: its world now holds a card-resolved numeric class of at least `G3_MIN_BOXES` boxes (the H3a stratum), and every policy D18 proposes recovers rows. |
| F1-R4 | **The poster's router check** (V.6 item 1). The poster is final and is never rebuilt. The check now pins the eight roles the poster printed by name, and lists roles added after it with their date (`adversary`, 2026-09-28). | `tests/test_poster_data.py` | the check passes, and fails if any printed role is renamed or removed |
