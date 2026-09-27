# INC literature corpus

The line-addressed literature the INC research brain may quote (docs/INC_AUTOPILOT.md, (c)). A brain plan's literature quote is accepted only when it is a verbatim substring of the passage line it names here and carries at least 20 characters of the passage's own text; the `KIND [topic]: ` label and the truncation mark do not count (`inc_autopilot/validate.py`, `corpus.Corpus.check_quote`). A quote of a US line is recorded as a project note, not as the paper's finding.

## What the lines are

**The passages are not verbatim text from the papers.** They are curated research notes: for each paper, one RESULT line (what the paper found), one US line (how the notes read it for this project) and one CORRECTION line (what a verifier checked against the paper, and what it corrected). The contract asked for verbatim passages from the PDFs. Nothing in this repository holds those, and a model paraphrase of a PDF would be worse than a checked note. A "quote" in a plan is therefore a quote of these notes, never of the authors. Every file repeats this on its `provenance:` line.

Two more limits:
- Some note fields were cut at about 400 characters when the notes were collected. Such a line ends with `[note truncated at collection]`.
- US lines were written on 2026-09-26 and can describe repo code that has changed since (for example the DINOv2 reference pool, fixed in Step 0.3).

## Files

- `<id>.md`, one per paper. `<id>` is the arXiv id, or a slug of the title for a paper without one. Lines 1-8 are the header (title, generated-file mark, id, bib, topics, status, informs, provenance); the passages follow `## Passages`. Only passage lines (`RESULT|US|CORRECTION [<topic>]: ...`) are citable, by file line number.
- `index.json`: every paper's id, file, title, url, venues, topics, levers, verifier status, line count and passage lines; the topic-to-lever map; the aliases; the path and sha256 of the source notes.
- `_source/notes.md`: the source the corpus is built from. It is the extract of the curated notes that the build reads: every TOPIC header, paper entry and RESULT/US/CORRECTION line, verbatim. The sweeps' SYNTHESIS, GAPS and VERIFIER blocks are not paper notes and are left out. The notes were collected on 2026-09-26; the full collection had sha256 `4cd5417d4db1aaddd919812656ba6a3a65759483d6b2457c982205cff4d33b15`, and building from it or from the extract gives the same files.

A paper that appears in several topic sweeps (for example OWL-ST in curation, label noise and self-training) has one file holding every sweep's lines, each tagged with its topic.

## Levers each paper informs

Assigned by topic, not per paper (`corpus.TOPIC_LEVERS`):

| Topic | Informs |
|---|---|
| continual (replay, forgetting, recipes) | L1, L2, X1 |
| curation (relevance, filtering) | L3, X2 |
| labelnoise | L4, X3 |
| attribution (statistics, data attribution) | L5, D8, X3, X4 |
| selftrain | L4 |
| weeds | L3 |
| agents | D13, D14 |
| webresearch | none |

## Seed papers

The papers the protocol cites are present, reachable by the aliases levers.json uses: `bouthillier2021` (2103.03098), `ibrahim2024` (2403.08763), `time2024` (2412.06712), `sorscher2022` (2206.14486), `owlst2023` (2306.09683), `dinov2_curation` (2304.07193), `dinov3` (2508.10104), `northcutt2021` (1911.00068, candidate). `lwf2016` (Li and Hoiem's LwF) is not in the notes; the corpus holds its YOLO descendant, 2503.04688.

## Rebuilding

The files are generated; do not edit them by hand.

```
python -m weed_optimizer_framework.tools.inc_autopilot.corpus build --notes ../docs/literature/_source/notes.md --out ../docs/literature
python -m weed_optimizer_framework.tools.inc_autopilot.corpus check
python -m weed_optimizer_framework.tools.inc_autopilot.corpus search "replay forgetting" --lever L1
```

(run from `weed_llm_benchmark/`). The build is deterministic, merges duplicate arXiv ids, drops the topic SYNTHESIS / GAPS / VERIFIER blocks (they are not paper notes), writes `_source/notes.md`, and fails if an alias target is missing. `check` verifies the source's sha256 against `index.json` and rebuilds the corpus from `_source/notes.md` into a temporary directory; any byte that differs from the committed files is reported. Built from 144 note entries in 8 topic sweeps: 128 papers, 432 passages.

## How it was verified

`tests/test_inc_ap_brain.py`: files and index agree; the committed source rebuilds the corpus byte for byte; every passage, whole and as a 40-character slice, resolves as a quote; header lines, quotes under 20 characters, quotes that are only the `KIND [topic]: ` label or the truncation mark, one-word paraphrases, unknown papers and out-of-range lines are refused; the seed papers carry the levers above; retrieval is deterministic; a rebuild is byte-identical.
