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

---

## 6. First cluster review, and the first head-to-head — n = 1, and it does not favour the big model

Job `45775879` is the first review this project has ever run on a cluster model.
`run_llm_review.sh` started ollama inside its own H100 allocation, spent **10 min 26 s**
loading `glm-4.7-flash` off Lustre, and reviewed round 15's train bundle in 147.8 s.

The lab's 7 B model had already reviewed **the same bundle**. Same prompt, same evidence, same
holdout:

| | `qwen2.5-coder:7b` · lab RTX 3060 | `glm-4.7-flash` · cluster H100 |
|---|---|---|
| verdict | issue, confidence 1.0 | issue, confidence 0.9 |
| findings | 2 | 2 |
| which signals | `epochs_truncated`, `plateau` | `epochs_truncated`, `plateau` — **the same two** |
| citations accepted | 0 of 2 | 0 of 2 |
| tokens read | 23,333 | 20,700 |
| time to answer | **23.5 s** | 147.8 s (+ 10 min 26 s to load) |

On diagnosis quality the smaller model was **better**, not worse:

> **7 B** — "The recipe asked for 60 epochs and 25 ran (0.42 of it, under the 1.00 floor)… It
> stopped at epoch 25, which is best epoch 5 plus patience 20, so early stopping ended it: the
> metric last improved 20 epochs before the end." Quote: `epochs=60 | epochs_completed=25 |
> patience=20`.
>
> **GLM-4.7-flash** — "The training stopped at epoch 25, far short of the requested 60, due to
> early stopping triggered by the patience parameter." Quote: `epochs=60`.

The 7 B reproduced the arithmetic and quoted three fields; the 30 B restated the signal and
quoted one. Neither grounded a single finding to a resolvable line, and neither found anything
the twelve deterministic checks had not already said on this bundle.

**This is one bundle.** It is not a benchmark and must not be reported as one — a single case
cannot separate two models, and this section exists precisely to say what a single case cannot
do. The five queued jobs (`45746472` qwen3.8:27b, `45746473` glm-4.7-flash, `45746474`
qwen3:14b, `45746475` qwen2.5:7b, `45746483` deepseek-v3:671b) run the full 149-case dev split
under identical conditions and are what will settle it.

What this one case does establish is narrower and still worth having: **the cluster path works
end to end**, a 19 GB model loads in about ten minutes on an H100 off Lustre, and the cost of
using it is roughly 6× the latency for no visible gain on a bundle this size. If the benchmark
agrees, the honest recommendation will be a small model for every step and a large one only
where the corpus shows the small one failing — which is a result about *where* model capacity
matters, not an excuse for having used the small one.

---

## 7. The first scored result: what each kind of supervision actually catches

Scored 2026-09-11 by walking every file in `results/framework/supervision_bench/verdicts/*/`
against each case's `truth.json`. The scorer was written from scratch rather than reused, so the
definitions are visible: **recall** is over the cases whose truth says `incident`, the
**false-alarm rate** is over the cases whose truth says control, and a case an arm could not
decide is counted as a miss and reported separately rather than dropped. Dev split, 149 cases —
**116 incidents, 33 controls**.

| arm | what it reads | TP | FN | FP | recall | false-alarm rate | cases |
|---|---|---|---|---|---|---|---|
| **A0** scripted watchdog | status fields | 0 | 116 | 0 | **0.000** | 0.000 | 149 |
| **A0p** signals-only | the 12 deterministic checks | 11 | 105 | 2 | **0.095** | 0.061 | 149 |
| **L2** · deepseek-v4-flash | raw artifact excerpts | 45 | 71 | 3 | **0.388** | 0.091 | 149 |
| **L3** · deepseek-v4-flash | excerpts + signal list | 38 | 78 | 3 | 0.328 | 0.091 | 149 |
| **L2** · qwen3.8:27b | raw artifact excerpts | 37 | 7 | 7 | **0.841** ⏳ | 0.212 ⏳ | **77 of 149, still running** |

**A0 flags nothing, on any case.** Its verdict files carry
`escalate.reason: "no signal fired"` with `signals_n: 11` — the scripted arm reads the status
fields, the status fields say the step succeeded, and it reports success 149 times out of 149.
That is the baseline this project's dashboard was for six months.

**A0p is deterministic and its three directories are byte-identical** (`model: ""`,
`deterministic: true`), which is the correct behaviour and a useful internal check: the
signals-only arm does not vary with which model is loaded beside it. It catches **11 of 116**
incidents. §3 of this document fixes its ceiling at **0.559** — 56 of the 127 incidents declare
no signal that could reach them — so 0.095 is **17 % of what the rules could achieve even in
principle**, and the shortfall is the reachability problem: the archived bundles carry 12 of 87
expected signal firings.

**The qwen3.8:27b row is not final.** Job `45746472` is still running; 77 of 149 cases are
scored (44 incidents, 33 controls) and the remaining 72 may not resemble them. Its 0.841 must
carry that mark until the job lands. The completed deepseek-v4-flash row at 0.388 is from the
run retracted on 2026-09-06 for context overflow — a third of its prompts were refused and
scored as misses — so it is a **floor** for what a model reading artifacts can do, not an
estimate.

### What can be said now, and what cannot

Sayable: **a scripted watchdog over status fields catches none of 116 real incidents; the
deterministic signal set catches 11; a model reading the raw artifacts catches several times
more, at a higher false-alarm rate.** The direction is large and consistent across two
independent model families.

Not yet sayable: any single number for "the model arm", because the strongest one is 52 %
complete and the completed one comes from a retracted run. Not sayable at all: that bigger
models are better at this — `deepseek-v4-flash` (284 B) scored 0.388 where `qwen3.8:27b` is
tracking 0.841, and the difference is confounded with the context-overflow bug the September fix
removed.

---

## 7a. §7 superseded: the same verdicts under the committed scorer

§7 was scored by a counter written for that table rather than by `bench`. On
2026-09-12 the same committed verdicts were re-scored with the project's own
instrument — `bench reproduce --split dev`, which makes no model call and reads
the verdict files already on disk — and the numbers move, in one direction, for
every model arm.

| arm | reads | recall | grounded | false alarms | cases |
|---|---|---|---|---|---|
| **A0** scripted watchdog | status fields | — | — | — | 149 |
| **A0p** deterministic signals | the 12 checks | 0.095 | 0.095 | 0.061 | 149 |
| **L2** · qwen3:14b | raw artifact excerpts | 0.664 | 0.602 | 0.606 | 149 |
| **L3** · qwen3:14b | excerpts + retrieval | 0.675 | 0.614 | 0.636 | 149 |
| **L2** · qwen3.8:27b | raw artifact excerpts | 0.553 | 0.518 | 0.212 | 149 |
| **L3** · qwen3.8:27b | excerpts + retrieval | 0.702 | 0.702 | 0.242 | 92 |
| **L2** · glm-4.7-flash | raw artifact excerpts | 0.821 | 0.769 | 0.200 | 78 |

Source: `results/framework/supervision_rescore/results/run_reproduce-20260912T022035.json`,
committed as `docs/poster/supervision_table.json`.

**Why they differ.** Two definitions, not one:

* `bench` counts a detection only when the verdict is `issue` **and** carries at
  least one finding at or above a severity bar. The §7 counter counted any flag.
* A case that produced no answer — a model error, a context overflow, an
  undecidable export — leaves the denominator in `bench` rather than counting as
  a miss. That is why the incident denominators read 113, 114 and 57 instead of
  116.

`detection_grounded` is the same rule with one more requirement: the finding must
quote a line that resolves in the artifact. It is 0.03–0.06 below recall for every
model arm, which is the fraction of detections that fire without evidence a reader
could check.

**What §7 got wrong about A0.** It reported the scripted watchdog at recall 0.000.
Under the committed scorer A0 has no rate at all: all 149 of its verdicts come
back `undecidable`, because "no signal fired" is not a judgement about the case.
The distinction matters — 0.000 reads as an arm that looked and found nothing,
where the truth is an arm that never produced a finding to score.

**The deepseek-v4-flash rows are gone, and not by hand.** Their verdicts predate
the 2026-09-07 split re-cut, so the scorer will not mix them with the current
corpus. The retraction §7 described in prose is now enforced by the instrument.

**Still partial.** `L3 · qwen3.8:27b` stands on 92 of 149 cases and
`L2 · glm-4.7-flash` on 78; jobs `45824752` and `45824751` complete them with
`bench run --resume`, which re-reads the committed verdicts and calls the model
only for the remainder.
