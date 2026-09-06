# Scoring rubric — supervision benchmark

**Frozen before any model is run. The split's sha256 and this file's sha256 are printed in every
results file; a number produced under a different rubric is a different experiment.**

This benchmark asks one question: *given the evidence a step actually left behind, what does a
supervisor catch, what does it invent, and what does it cost?* It does not ask whether a model is
clever. Every rule below exists because the alternative would let a wrong answer look right.

## 1. What a case is

A case is one step of one round, exported by `corpus.py` from artifacts that existed before anyone
diagnosed the incident. `truth.json` carries the labels; `bundle.json` carries the evidence; the
`artifacts/` directory carries the files the bundle quotes, hashed as bytes.

* `provenance: raw` — at least one artifact file survived and is stored. **140 of 162.**
* `provenance: record-only` — the incident is real and dated but its artifacts are gone. Exported with
  an empty artifacts directory and a reason. **22 of 162.** Record-only cases are reported in a
  separate column and are never mixed into a headline number, because a supervisor cannot read what
  does not exist.

Controls are cases whose correct verdict is *no incident*:
* **31 healthy controls** — completed steps of real rounds and the sealed measurement runs.
* **4 designed controls** — conditions that look like incidents and are not. The chronic
  `SKIPPING github` line is one of them, and it sits inside the worked example's own bundle, so an
  arm that fires on it is charged a false alarm on the very case it is supposed to get right.

## 2. What counts as a detection

An arm **fires** on a case when it returns `verdict: "issue"` with at least one finding at severity
`warn` or above. Severity is pre-registered here so it cannot be reinterpreted after the fact: a
finding at `info` is never a detection and never a false alarm.

* **detected** — fired on a case whose truth is `incident: true`.
* **false alarm** — fired on a control.
* **detected_grounded** — detected *and* at least one quote resolved into the bundle. This is the
  number that goes on a slide. A detection whose evidence does not exist is not a detection.

The scripted baseline **A0** has no findings, so its detection is defined in code rather than
asserted: A0 detects a case when the step is recorded `failed` and the domain pauses within one tick
of the artifacts the bundle carries. By that definition A0 detects the 2026-08-29 walltime pair and
diagnoses nothing — which is exactly what happened, and is why the baseline row reads
"detected and paused, 0 diagnosed, 0 notified" rather than 0.

## 3. Citations

A finding must quote an artifact line verbatim. The validator normalises whitespace and requires at
least 20 characters, then resolves the quote to an absolute `(artifact_id, line)` address — absolute,
because a tail-relative index resolves to the wrong line as soon as a file grows.

Three failures are counted **separately**, because collapsing them hides the interesting one:

| counted as | meaning |
|---|---|
| `findings_without_quote` | the model asserted something and cited nothing |
| `rejected_unverifiable` | the quote does not resolve anywhere in the evidence it was given |
| resolved but not load-bearing | the quote is a real line that does not support the claim |

The third is the one worth watching. A model that cites a real line which does not support its
finding is harder to catch than one that fabricates, and it is the failure this campaign already
suffered from a plausible narrative that survived several human passes.

`evidence_hit_rate` is the fraction of a case's `load_bearing_lines` that the arm actually quoted.

## 4. Corrections

Correction correctness is scored **only on the R1/R2-correctable subset** — the cases whose truth
lists an acceptable correction an agent is permitted to apply. Roughly two thirds of the corpus is
code-class or design-class, where the correct behaviour is to escalate, not to act; scoring those as
correction failures would punish the right answer. Detection is reported over all classes;
correction is reported over the correctable subset, and the two are never averaged together.

Escalation is scored against `escalation_expected`: `none`, `tier1`, `human`. Escalating a
code-class or design-class finding to a human is correct; proposing a parameter change for it is not.

## 5. Reporting

* Every model arm is reported as a **margin over the signals-only arm (A0p)**, per incident class,
  never as an absolute. The signals were written from these incidents; an absolute recall would be
  measuring the author, not the model.
* Every proportion carries a **Wilson score interval**.
* Every comparison prints its **minimum detectable difference** beside it. A difference smaller than
  that is reported as "not separable at this n", never as a result.
* Three passes, and what each may claim:
  1. **Signal-blind, all 116 in-scope incidents.** Each case is scored with the signal that covers it
     removed from its bundle. This is the headline pass, because it is the only one that asks what
     the model adds beyond the rule that was written from that same incident.
  2. **Leave-one-class-out.** Generalisation across incident classes.
  3. **The pre-registered date split (dev 149 / test 13).** Reported as a **count and an interval, not
     a comparison.** At this size a paired test needs a recall margin of about 0.67 to reach
     significance. Its job is to answer when the detectors were conceived, and that is all it is
     allowed to answer.
* The 2026-08-29 walltime pair is the **worked example** and sits in dev. It is used to develop and
  to explain, and it never appears in a reported number.

## 6. The ceiling, stated before scoring

**56 of the 127 incidents have no expected signal at all** — library-version mismatches, path
assumptions, ownership holes, experimental-design errors. No artifact invariant reaches them. The
deterministic arm therefore cannot exceed roughly half the corpus by construction, and that is the
room a supervisor model has to earn. Saying it afterwards would look like an excuse; saying it here
makes the signals arm's ceiling a prediction rather than a finding.

## 7. Cost

Service units come from `sacct` (`AllocTRES` times `Elapsed`), never from an estimate. `tokens_in` is
recorded per call and asserted against the model's `num_ctx` before the call, because a bundle
silently truncated at the context boundary is the same defect class this benchmark exists to measure.

## 8. What would falsify the thesis

If the artifact-grounded arm (L3) shows no margin over signals-only (A0p) on the signal-blind pass,
outside its interval, then artifact grounding adds nothing over the rules on this corpus, and that is
the result that gets published. The same applies if metrics-only (L1) matches L3: it would mean the
silent-wrongness class is reachable from status fields after all.
