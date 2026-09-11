# What the supervision benchmark can and cannot measure

*Computed 2026-09-10 from `results/framework/supervision_bench/` — 162 `truth.json`
files plus `split.json`. No model was run and no compute was spent; these are properties of
the corpus itself, and they bound every number the benchmark will ever produce.*

---

## 1. Shape

| | count | share |
|---|---|---|
| cases | **162** | — |
| dev split | 149 | 92.0% |
| test split | **13** | 8.0% |
| incidents (`incident: true`) | **127** | **78.4%** |
| controls (`incident: false`) | 35 | 21.6% |

`pre_registered.detect` agrees with `incident` on all 162, so the detection label is the
incident label — there is no second, independent annotation behind it.

**The base rate is 78.4%.** A model that answers "incident" on every case scores detection
recall 1.000 and a false-alarm rate of 1.000. Detection recall must always be reported
beside the false-alarm rate; on its own it is not a result.

**13 cases is not a test split.** It is a held-out sample too small to separate arms that
differ by less than roughly 0.15. Report dev results as dev results.

## 2. What kind of failures these are

| class | n | | escalation expected | n |
|---|---|---|---|---|
| `code` | 72 | | `human` | 95 |
| `none` (control) | 35 | | `none` (control) | 35 |
| `config` | 32 | | `tier1` | 32 |
| `design` | 16 | | | |
| `plan` | 7 | | | |

Provenance: **140 raw** (labelled from the artifacts themselves), **22 record-only**
(labelled from the engineering record, with no raw artifact behind them). The 22 must be
excluded from any claim about reading evidence, because there is no evidence to read.

## 3. The ceiling on the deterministic baseline

Each case declares which of the twelve signals should fire on it.

| | count | share of incidents |
|---|---|---|
| incidents a signal **can** reach | **71** | **55.9%** |
| incidents **no** signal can reach | 56 | 44.1% |
| cases whose correct answer is "no signal fires" | 91 | — (56.2% of all cases) |

Signals expected, by name: `walltime_bound` 20, `pool_growth` 18, `gate_noop` 13,
`stale_artifact` 9, `plateau` 7, `source_degraded` 7, `job_unknown` 6,
`ownership_violation` 4, `budget` 2, `mongo_down` 1 — and `none` on 91.

**A signals-only arm cannot exceed 0.559 detection recall on this corpus.** Forty-four
percent of the incidents are, by the corpus's own pre-registration, outside what any of the
twelve deterministic checks can see. That number is a property of the corpus, fixed before
any model ran.

### What follows for reporting

- A margin of "model over rules" measured across all 127 incidents is **partly measuring
  this ceiling**, not the model. The fair contest is the **71 signal-reachable incidents**,
  where the rules are allowed to win.
- Both numbers should be reported: the margin on the 71, and the margin on all 127 with the
  0.559 cap stated beside it.
- The 91 no-signal cases are where a model can contribute something a rule cannot **by
  construction** — that is the interesting half, and it must be labelled as such rather than
  folded into one average.

## 4. The reachability problem, separately

The retracted first run (job 45344219) measured the signals-only arm at **0.095** detection
recall — **17% of its own 0.559 ceiling**. That shortfall is not the rules being weak. It is
that the archived evidence bundles carry only 12 of the 87 expected signal firings: the
`ledger`, `harvest`, `su`, `resources`, `corrections`, `plan` and `registry_diff` sections
are empty in all 162 cases, so most checks return `unknown` rather than running.

**Two denominators must therefore be reported for any signals-only number**: out of what the
corpus expects (87), and out of what the bundles actually make reachable (12).

## 5. Standing caveats

- No number from job 45344219 is quotable. It was retracted twice: its scripted baseline was
  undecidable on all 149 cases, 51 of 149 prompts exceeded the context window and were scored
  as misses, and the corpus was re-frozen afterwards so the run is no longer comparable.
- The corpus's ground truth was written from this project's own engineering record, by the
  same system that wrote the reviewer prompt. That is a real limitation and belongs on the
  slide, not in a footnote.
