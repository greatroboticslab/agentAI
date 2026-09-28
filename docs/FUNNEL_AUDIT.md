# Funnel audit: what the Step 1 filters discarded, whether it can be recovered, and whether the platform can find this by itself

This is the contract for the FUNNEL campaign. It builds on:
- docs/INCREMENTAL_PROTOCOL.md: Step 1, and Steps 2–3;
- docs/INC_AUTOPILOT.md: diagnoses D1–D16, levers L1–L9, cards X1–X9, and replay cases R1–R8.

**Version: pre-registration v1, 2026-09-28.**
- This file is committed before any audit sample is drawn and before any reference label exists. So is its machine-readable transcription, `INC_DIR/funnel/prereg_v1.json`: the thresholds, strata, seeds, gates and class policy of §5–§7.
- If the file and the JSON differ, that is a defect. The doc wins, and the fix is recorded as a dated amendment.
- Every later audit artifact records the sha256 of both files.

**Markers:**
- **[exact]**: a count read from a Step 1 or realloop artifact. No sampling is involved.
- **[post hoc]**: suggested by data already seen. That data is `census_v0.json`, the image-rule query and the dataset forensics of 2026-09-28. Such a claim is confirmatory only on the new reference-labelled sample.
- **[unverified]**: not checked against its source.
- **[review]**: added or changed by the completeness and rigour review of 2026-09-28, before the first commit of this file. The review read `census_v0.json`, the local copies of `step1/{pool,admit,select}_summary.json`, `realloop_v1/report.json`, `verify.py`, `gate.py`, `model_router.py` and the autopilot modules. A [review] count computed from those files is also [post hoc]. `prereg_v1.json` must be regenerated from this version before F0 (§12).

**Paths:**
- `INC_DIR` = `$REPO/results/framework/inc` on the cluster. Local copies are under `weed_llm_benchmark/results/framework/inc/`.
- Code paths are relative to `weed_llm_benchmark/weed_optimizer_framework/tools/` unless stated.
- The census used below is `results/framework/inc/funnel/census_v0.json`. It has 140 rows (source × source class name × label) covering all 545,318 embedded crops, with verdicts, predictions and confident predictions. Its crop sheets are in `funnel/sheets/`.

---

## 1. Question and why

**The question has four parts:**
1. **Discard.** Did the Step 1 filters discard target-species data that was correctly boxed? Target species means the cwd12 species as defined by `cwd12_species.CWD12_BINOMIAL`. If so, how much, at which stage, and with what uncertainty?
2. **Recovery.** Can that data be recovered at a measured precision?
3. **Value.** Does recovered data raise dev accuracy when it is added to base B?
4. **Platform.** Can the platform reach 1–3 by itself, in any domain? That means noticing that a conclusion rests only on filtered data, auditing its own filters, recovering data, and testing it.

**The result that prompted it (RESEARCH_LOG 2026-09-27):**
- **Harvest.** Step 1 harvested 102,920 images and 670,818 boxes from 52 sources [exact, `step1/pool_summary.json`]. 13 further slugs were skipped.
- **Verifier.** The BioCLIP-2 verifier verified 2,049 target-species boxes. That is 28.1 % of the 7,288 boxes whose source name joins to a target, and 0.31 % of all boxes [exact, `admit_summary.json`].
- **Image rule.** Only 1,060 of those 2,049 boxes survived the image-level admission rule.
- **Base B.** B has 3,927 images: train_core 3,049 plus 878 harvested.
- **realloop_v1 increment pool.** The source-level evidence criterion kept 5 of 38 sources and 1,439 of 96,088 increment-pool images.
- **realloop_v1 results.** The loop decided 6 increments of 287 images each.
  - Truth arm: V1–V3 neutral; OTHER_HEAVY and V4 hurt.
  - Chain: accepted nothing.
  - B ∪ every verified increment scored dev 0.8093 ± 0.0027, against B's 0.8133 ± 0.0070. On test it scored 0.841 against 0.850.
  - Cost: 26.2 GPU-h.
- **The log's reading:** "The harvested increments do not improve the twelve target species. They are mostly other plants plus about 50 target-species boxes each."

**Why that reading is not yet established:**
- **It is observed only through the filters.** Every increment was drawn after the filters ran, so a negative loop result says nothing about what the filters discarded (selective labels, §2).
- **The only increment drawn from rejected data is the only one the truth arm scored "helps".**
  - That increment is UNVERIFIED: 287 images of `rf_tuf__weed-3434e` that the verifier did not admit, labelled all-OtherPlant by the join.
  - Its truth P was 0.889; dev was 0.8184 ± 0.0031 with it and 0.8151 ± 0.0012 without it [exact, `realloop_v1/report.json /steps/2/truth`].
  - **[review] This is weak evidence.**
    - The difference is +0.0033, below the 0.011 minimum detectable effect of §6 H10.
    - The chain arm scored the same step "hurts" (`/steps/2/chains/full/truth_equivalent`).
    - With 3 seeds per arm and no true effect, `gate.truth_detail` returns "helps" (P ≥ 0.75) with probability 4/20 = 0.20. That is the exact permutation null of `p_greater` for continuous scores, before the species guard. So one "helps" among six steps is unsurprising.
    - It is a reason to look, not evidence that rejected data helps.
- **The verifier was calibrated only in its own domain.**
  - Its six known-truth sets are all copies of cwd12 photographs (`calibration.json`): verified precision 0.9996–1.0, recall on correct labels 0.9457.
  - Two NDSU sources publish papers that name their species. On their target-labelled boxes it verified 855 of 4,639, that is 18.4 % (Wilson 95 % 17.3–19.6 %) [exact]. For Ragweed it verified 13 of 1,086 boxes in one source and 14 of 963 in the other.
  - **[review] The earlier "measured label precision" of the NDSU sources is the same kind of reading.** The 2026-08-25 S1-gate audit gave weed_crop 0.737, imageweeds_aerial 0.650 and mh_weed16 0.452 (`docs/poster/figures_data.json /s1_gate_verdict_2026_08_25`).
    - Those numbers are a probe's agreement with the source labels. The mh_weed16 figure predates the v3.60.0 join fix.
    - That audit's reading ("the probe reads 1.000 on human-labelled cwd12, so those are the data, not the instrument") is the in-domain-calibration inference this campaign questions.
    - It is filed as a second claim (C2, §8.3). It is no longer used to justify the recovery gate (§7).
- **Many OtherPlant boxes carry no species information in their source name.** There are 220,104 such embedded boxes [exact, census_v0]:
  - no name: 69,867;
  - numeric name: 55,657;
  - generic name ("weed", "others", "grass weeds"): 94,580. [review] The original text said "crop" here. `verify.other_name_status` classifies "crop" as a named other plant, not as generic.

  The join can only call them OtherPlant, and the evidence criterion can never admit their sources.
- **[review] That count understates the uninformative names**, because the name-status rule has holes [post hoc; `other_name_status` run locally over `census_v0.json`]. Of the 316,417 boxes it calls "named other plant":
  - 197,613 are the abbreviations "BroWeed" (196,057) and "NarWeed" (1,556), both from `rf_weed-tnf9e__weed-bqdok`. They mean broadleaf weed and narrowleaf weed.
  - 30,042 are "crop" or "Crop".
  - 2,149 are non-plant objects: greenhouse, hut, solar, shed, pave, "Musor" (Russian for rubbish), and others.
  - 1,403 are disease or state names: cercospora, xanthomonas, mosaic, leaf curl, healthy, dry leaf, dry grass.
  - About 85,210 carry a real taxon or crop name.

  With the two abbreviations counted as generic, uninformative names cover 417,717 of 545,318 embedded boxes (0.766), not 0.404.
- **[review] The same holes shape the verifier's OtherPlant class.**
  - `_other_sample` draws its 5,000 OtherPlant training crops uniformly from the "named other plant" boxes.
  - So about 62 % of the draw is expected to be "BroWeed" from a single maize-field source. [exact, `step1/verifier/fit_info.json` read 2026-09-28] The eligible pool is 316,417 boxes over 62 names, and BroWeed is 196,057 of them (0.620). The per-name counts of the 5,000 drawn crops are recorded by F3.
  - Broadleaf weeds in maize can include *Amaranthus* and *Portulaca*. Only 4 of the 196,057 BroWeed boxes are conflicts.
  - This is signal S6 of §8.4 with a concrete, checkable mechanism.
- **The one filter with a fail-closed calibration check failed it.** Zero-shot relevance scored 34.5 % of genuine train_core weeds as non-plant.
- **The opposite reading is also open.** Many conflicts are confusions between related plants. For example, redroot pigweed (*Amaranthus retroflexus*) is confidently called Palmer amaranth. So "the verifier is wrong" is not established either.

**What is needed.** An audit on labels that are independent of the filter, reported in both directions, and a controlled training test of whatever is recovered.

---

## 2. What the literature says (verified claims only)

Each row was checked against the paper's abstract [abs] or the named passage of its full text [full]. A row marked [corpus] was taken from the verifier-checked note in `docs/literature/` and not re-checked.

### 2.1 Filters discard good data unevenly; audit the discarded set

| Finding | Source | Checked | Consequence here |
|---|---|---|---|
| Blocklist filtering of C4 "disproportionately removes text from and about minority individuals". AAE and Hispanic-aligned English were removed at 42 % and 32 %, against 6.2 % and 7.2 %. The method: keep the unfiltered corpus, cluster what was removed, and hand-check samples. | Dodge et al., EMNLP 2021, https://arxiv.org/abs/2104.08758 | abs; full §5 | Keep a discard ledger per stage. Report removal rates per class and per source. |
| Recall is estimated by sampling both the retrieved and the unretrieved segments. The normal approximation "provides poor coverage"; beta-binomial posteriors with a Monte Carlo estimate do better. | Webber, ACM TOIS, https://arxiv.org/abs/1202.2880 | abs | Estimator for funnel recall (§5.4). |
| "The observed outcomes are themselves a consequence of the existing choices"; observed instances are not a random sample. | Lakkaraju et al., KDD 2017, https://doi.org/10.1145/3097983.3098066 | abs (Semantic Scholar) | realloop_v1 observed outcomes only for data the filters admitted. |
| A classifier trained on positive and unlabelled data differs from the true one by a constant factor c = p(s=1\|y=1), provided the labelled positives are selected at random. | Elkan & Noto, KDD 2008, https://cseweb.ucsd.edu/~elkan/posonly.pdf | abs; full | A box with no target name is unlabelled, not negative. For a no-name source, c ≈ 0 by construction, so the assumption fails. |
| Merging datasets with partial annotations makes a category foreground in one and background in another. A unified detector without pseudo-labels scored 36.5 (V-on-C); with pseudo-labels, 62.2. | Zhao et al., ECCV 2020, https://cseweb.ucsd.edu/~mkchandraker/pdf/eccv20_unifiedobjectdetection.pdf | full Table 2 | Relabel or mask rather than drop, or treat a missing class as OtherPlant. |
| Label relations are discovered from visual evidence, as the mean prediction in both directions over easy instances. Binary-relation PR-AUC: 0.71–0.83 with visual embeddings, 0.74–0.83 with WordNet plus visual, against 0.51–0.71 for language-only methods. | Uijlings et al., "The Missing Link", ECCV 2022, https://arxiv.org/abs/2206.04453 | full Fig. 4, Table 2 | A class-level relation audit comes before any box is relabelled (L11). |
| Label spaces are merged by visual merge cost. A new dataset's class is merged into a unified class when the cost is below 5 AP, and appended as a new class otherwise. | Zhou et al., UniDet, CVPR 2022, https://arxiv.org/abs/2102.13086 | full App. C | The class-map decision rule. |
| Keeping all pseudo-annotations above a moderate 0.3 works well, and strict thresholds lead to poor results. Boxes above 0.1 were kept, in images with at least one box above 0.3. | Minderer et al., OWL-ST, NeurIPS 2023, https://arxiv.org/abs/2306.09683 | full §4.4 | Our image rule is the opposite: every species box must be verified and no box may conflict. |
| At small scale, intersecting two filters scored 0.144, against 0.173 for CLIP score alone. | Gadre et al., DataComp, NeurIPS 2023, https://arxiv.org/abs/2304.14108 | full Table 3 | We stack join, verifier, image rule, evidence and gate at the small-data end. |
| A filter's quality is distinct from its downstream performance. | Fang et al., DFN, ICLR 2024, https://arxiv.org/abs/2309.17425 | abs | The verifier's in-domain precision says nothing about its recall on the harvest, or about the value of what it keeps. |
| The highest-precision threshold gave the worst downstream model. High auto-label recall was the best single predictor of downstream performance. | arXiv 2506.02359 | corpus | Tune admission on dev mAP, not on label precision alone (§9). |

### 2.2 Judges, calibration and species identification under shift

| Finding | Source | Checked | Consequence here |
|---|---|---|---|
| Post-hoc calibration "falls short" under dataset shift. | Ovadia et al., NeurIPS 2019, https://arxiv.org/abs/1906.02530 | abs | Thresholds τ and σ were set on train_core, and the relevance τ on the train_core 5th percentile. Neither transfers. |
| Target accuracy can be predicted from a source-learned confidence threshold (ATC). Any such method rests on assumptions about the shift. | Garg et al., ICLR 2022, https://arxiv.org/abs/2201.04234 | abs | Label-free screens can prioritise sources only; they are never truth. |
| BioCLIP 2 on PlantVillage: zero-shot 25.1, one-shot 67.6 ± 1.1, five-shot 83.9 ± 0.9 (Table 1). On PlantDoc: 40.4 ± 3.7, against 40.3 ± 1.2 for DINOv3 (Table 2). | BioCLIP 2, NeurIPS 2025, https://arxiv.org/abs/2505.23883 | full, arXiv HTML checked 2026-09-28 | Zero-shot is weakest on close-up agricultural crops. DINOv3/DINOv2 is a strong second judge from a different family. |
| Test text of taxonomic + common names gives 38.0 on Rare Species, against 31.6 for common names and 30.1 for scientific names. | BioCLIP, CVPR 2024, https://arxiv.org/abs/2311.18803 | full Table 5 | Prompt format for J-zs. |
| Open-ended species identification is poor, while multiple choice reached 82.6 % (GPT-4V, birds). | VLM4Bio, https://arxiv.org/abs/2408.16176 | full (HTML) | A VLM judge answers closed multiple choice only. |
| "Palmer Amaranth and Waterhemp that are both pigweed species may look similar and are difficult to distinguish." Morningglory classes are grouped into one class. | CottonWeedID15, https://arxiv.org/abs/2110.04960 | full | The class policy (§5.1); a reference-labeller level for *Amaranthus* (§4.3). |
| With a standard triplet loss, 481 of 538 *Chenopodium album* images were classified as *A. tuberculatus*; a taxonomy-aware loss cut this to 43. | Fontaine et al., Sci. Rep. 2026, https://pmc.ncbi.nlm.nih.gov/articles/PMC12852093/ | full | A closed-set head pulls a lookalike to the nearest target. This is the probe-attractor mechanism (§4.2). |
| Models agree about 60 % of the time when both are wrong. | Kim et al., ICML 2025, https://arxiv.org/abs/2506.07962 | abs | Agreement between judges is not truth. Correlated judges count once. |
| 51 % of algorithmically flagged candidates were confirmed as label errors. | Northcutt et al., NeurIPS 2021 D&B, https://arxiv.org/abs/2103.14749 | abs | A flag (a conflict) is a candidate in both directions. |
| The best automatic methods still miss up to 66 % of label errors. | Rechecked, https://arxiv.org/abs/2508.06556 | abs | Automatic re-judging alone under-counts, so the estimate needs a reference-labelled sample. |
| Yes/no box verification takes 1.6 s, against 26–42 s to draw a box. 48 % of verification errors are on objects smaller than 10 % of the image. | Papadopoulos et al., CVPR 2016, https://arxiv.org/abs/1602.08405 | full | Verify-only reference labelling, with crops shown enlarged. |

### 2.3 Statistics

| Finding | Source | Checked | Use |
|---|---|---|---|
| Prediction-powered inference: valid confidence intervals from a small gold sample plus ML predictions on the full set. | Angelopoulos et al., Science 382:669 (2023), https://arxiv.org/abs/2301.09633 | abs | Judge-assisted estimates. |
| PPI++ always improves on intervals from the labelled data alone. Stratified PPI gives tighter intervals when the judge's quality varies by stratum. | https://arxiv.org/abs/2311.01453; Fisch et al., https://arxiv.org/abs/2406.04291 (venue unverified) | abs | Stratified PPI with power tuning. |
| Active testing gives unbiased, lower-variance evaluation from actively chosen labels. | Kossen et al., ICML 2021, https://arxiv.org/abs/2103.05331 | abs | Horvitz–Thompson weights for prioritised draws. |
| Wald intervals are unreliable; use Wilson or Jeffreys for small n. | Brown, Cai & DasGupta, Stat. Sci. 16(2), 2001 | full | Per-stratum intervals. |
| Prevalence correction for test sensitivity and specificity. | Rogan & Gladen, Am J Epidemiol 107:71 (1978) | abs | Reference-labeller error correction. |
| Reusing the selection data for analysis gives distorted statistics and invalid inference. | Kriegeskorte et al., 2009, https://pubmed.ncbi.nlm.nih.gov/19396166/ | abs | Calibration sets must be disjoint from audited strata (§4.1). |
| Preregistration separates prediction from postdiction. | Nosek et al., PNAS 2018, https://pubmed.ncbi.nlm.nih.gov/29531091/ | abs | This document. |

### 2.4 Agents that check their own conclusions

| Finding | Source | Checked | Consequence here |
|---|---|---|---|
| LLMs "struggle to self-correct their responses without external feedback". GPT-3.5 on CommonSenseQA fell from 75.8 to 38.1 after self-correction. | Huang et al., ICLR 2024, https://arxiv.org/abs/2310.01798 | abs; full Table 3 | "Question the conclusion" must trigger a measurement, never a re-read. |
| The model's own critiques gave −0.03 and +2.33 F1; with tools the gain was +7.7 F1. | Gou et al., CRITIC, ICLR 2024, https://arxiv.org/abs/2305.11738 | full §4.1 | The devil's advocate must name a tool-backed test. |
| Answering verification questions independently of the draft raised precision from 0.29 (joint) to 0.36 (two-step). | Dhuliawala et al., CoVe, https://arxiv.org/abs/2309.11495 | full Table 1 | Judges and the reference labeller are blind to the filter's verdict and the source name. |
| LLM evaluators favour their own generations. Assistants are sycophantic toward the user's view. | Panickssery et al., https://arxiv.org/abs/2404.13076; Sharma et al., https://arxiv.org/abs/2310.13548 | abs | The adversary is a different model family from the planner. Outcomes are reported in both directions. |
| Sequential falsification with Type-I control. A naive Fisher combination had Type-I error 0.311 at α = 0.1. | Huang et al., POPPER, https://arxiv.org/abs/2502.09858 | full Table 3 | Multiple-testing control across strata (§5.4). |
| 85.5 % of data-analysis statements were reproducible, but only 57.9 % of synthesis statements were accurate. | Mitchell et al., Kosmos, https://arxiv.org/abs/2511.02824 | full §2 | "The harvest has few target labels" is a synthesis statement and is routed to an audit (D17). |
| 29 teams analysed one dataset: odds ratios 0.89–2.93; 69 % found a significant effect and 31 % did not. | Silberzahn et al., AMPPS 2018, https://doi.org/10.1177/2515245917747646 | abs (Semantic Scholar) | The target-count conclusion is re-derived over a grid of defensible filter settings (H9). |

### 2.5 [review] Provenance, reference labelling and independent truth

| Finding | Source | Checked | Consequence here |
|---|---|---|---|
| "[L3.2] Nonindependence between train and test samples constitutes leakage, unless the scientific claim is about a distribution that has the same dependence structure." The paper also lists "[L1.4] Duplicates in datasets". | Kapoor & Narayanan, https://arxiv.org/abs/2207.07048 (venue unverified) | full text §taxonomy | A pixel-copy detector cannot see a same-lab, same-site, same-season dependence. Provenance is checked separately (H6c). |
| 3.3 % and 10 % of the CIFAR-10 and CIFAR-100 test images have duplicates in the training set. | Barz & Denzler, J. Imaging 6(6):41 (2020), https://arxiv.org/abs/1902.00423 | abs | Near-duplicates survive standard splits. The copy detector is calibrated on the augmentations re-exports actually apply (H6). |
| "50 % prevalence produced 7 % miss errors … 10 % prevalence produced 16 % errors, while at 1 % prevalence errors soared to 30 %." | Wolfe, Horowitz & Kenner, Nature 435:439 (2005); author copy https://search.bwh.harvard.edu/new/pubs/Nature_suppl.pdf | full (author copy) | A person, or a model with a sheet-level prior, under-reports targets in low-prevalence strata such as G4. Sheets interleave strata, and sentinel prevalence matches the sheet's (§4.3). |
| iNaturalist Research Grade requires that "more than 2/3 of identifiers agree on a taxon at species-level". | https://help.inaturalist.org/en/support/solutions/articles/151000169936-what-is-the-data-quality-assessment-and-how-do-observations-qualify-to-become-research-grade- | page | A community-verified species truth that is independent of every audited source (KT7). |
| TreeOfLife-200M is curated from GBIF, EOL, BIOSCAN-5M and FathomNet, including 151 M GBIF citizen-science images. | BioCLIP 2, https://arxiv.org/html/2505.23883 | full (HTML) | KT7 images may be in BioCLIP-2's training data, so KT7 never qualifies J1 or J-zs. It qualifies the RL and the DINOv2 judges only. |
| ImageWeeds is from NDSU. Its field images are from Casselton (late May–late June 2021), Carrington (mid-July–late August 2021) and Grand Farm (mid-August–late September 2022). Its individual-weed images are from the NDSU greenhouse, taken with a Canon 90D. Its augmented images are part of the distributed dataset. | PMC10618417 (Data in Brief), https://pmc.ncbi.nlm.nih.gov/articles/PMC10618417/ | full | The ImageWeeds exam shares lab, sites, seasons and camera model with the recovery sources (next two rows). |
| Weed-crop dataset (`weed_crop_detection`): NDSU; Casselton, Grand Farm and Carrington; summers 2021 and 2022; Canon EOS 90D on ground robots. | PMC11986624, https://pmc.ncbi.nlm.nih.gov/articles/PMC11986624/ | full | Same lab, sites and seasons as ImageWeeds. |
| Greenhouse dataset (`greenhouse_crop_weed_detection`): NDSU Waldron Greenhouse; Canon EOS T7 and 90D; 6 weeds including *A. palmeri* and *A. artemisiifolia*. | PMC11599996, https://pmc.ncbi.nlm.nih.gov/articles/PMC11599996/ | full | Same lab as ImageWeeds, and possibly the same greenhouse [unverified]. |
| 3SeasonWeedDet10 (creator Y. Lu, Michigan State University): the 2021 and 2022 images are from Mississippi State University farms, and the 2023 images are from Holt, MI. The 2021 subset is "derived from our previous CottonWeedDet12 dataset". CottonWeedID15's authors are D. Chen, Y. Lu, Z. Li and S. Young. | https://zenodo.org/records/14861516; https://arxiv.org/abs/2110.04960 | record page; abs page | cwd12, CottonWeedID15 and the ood22 and ood23 exams come from one lab. ood22 is the next season on the same farms, which is the designed temporal shift. |
| MH-Weed16: 16 classes (ids 0–15), 6,656 images with boxes, still images from soybean fields in Maharashtra (July–November 2023 and 2024). The article mentions no videos. The AgML card lists 6,656 images and 15 classes under CC BY 4.0; Mendeley v2 is CC BY 4.0. | PMC12179629; https://huggingface.co/datasets/Project-AgML/MH_Weed16_weed_detection; https://data.mendeley.com/datasets/d3n3mgjjbv/2 | full; card; page | AgML dropped one of the 16 classes, and which one decides the id alignment (H3a). The "240 videos" of §7 and §11 is not supported by these sources. |

**Not used, or unverified:**
- the MSeg "1.34 annotator-years" figure (not found in the PDF);
- the Co-Scientist venue;
- Goyal et al.'s epoch-4 crossover [corpus only];
- the "2SeasonWeedDet8" description (search-engine summary only).

**Corpus defect found while checking.** `docs/literature/2505.23883.md` (lines 11 and 13) attributes 67.6 % and 83.9 % to PlantDoc. The arXiv HTML gives them for PlantVillage (Table 1); PlantDoc is 40.4 (Table 2). The note is corrected through the corpus verifier, not by hand.

---

## 3. The funnel stages and their discard counts

All counts are [exact] unless marked. Sources:
- `step1/{pool_summary, admit_summary, select_summary, calibration}.json`;
- `funnel/census_v0.json`;
- `realloop_v1/{report, build_summary}.json`.

### 3.1 Stage table

| # | Stage (code) | Unit | Rule | Discarded or diverted | Recoverable |
|---|---|---|---|---|---|
| S−1 [review] | Discovery: the harvest brain's search and download choices (`dataset_discovery`) | dataset | whatever the harvest agent searched and accepted | not counted anywhere. The claim "the harvest has few target-species labels" depends on this stage too. | yes (harvest), outside Step 1. Measured by a known-item recall test (H12). |
| S0 | Harvest cap (location of the limit not found in the local repo [unverified]) | image | download limit | `project_agml__mh_weed16_weed_detection`, `project_agml__three_season_weed_detection` and `fvossel__csgo_player_detection` each list exactly 5,000 images. MH-Weed16's card gives 6,656, so 1,656 of its images were never fetched. [review] `rf_weed-tnf9e__weed-bqdok` lists exactly 10,000 images, which may be a second cap [post hoc]. three_season contributes 0 pool images: 2,576 are cwd12 copies and 2,424 are near_eval. | yes (re-fetch, through every Step 1 guard; §7 R-F); three_season is exam material and never recovered |
| S1 | Registry skip (`pool_summary.skipped`) | source | never_train, quarantine, layout, user flag | 13 of 65 slugs: never_train 2, quarantined 6, unknown_layout 2, user_flag_garbage 3. [review] The quarantined slugs are `dronefreak__{brackish,exdark,gwhd,seadronessee}`, `project_agml__cotton_weed_detection` and `project_agml__synthetic_cowpea_pod_detection`. Their reasons are not in `pool_summary.json`, and the local registry copy is stale. AgML's cotton_weed_detection card lists 262 UAV images with the classes weed and cotton, so no target names (https://huggingface.co/datasets/Project-AgML/cotton_weed_detection). | unknown_layout only (a parser gap). [exact, cluster registry read 2026-09-28] The six quarantine reasons: brackish (underwater fish), exdark (low-light street scenes), gwhd (wheat heads) and seadronessee (maritime search and rescue) are off-goal; `project_agml__cotton_weed_detection` has degenerate labels (all 5,670 sampled boxes sub-pixel) and only the classes weed and cotton; `synthetic_cowpea_pod_detection` has degenerate boxes and is off-goal. No target-species data is lost at S1 through quarantine. |
| S2 | Read | image | a label file with no valid box | no_boxes 3,506 (rf_tuf 1,843; csgo 983). [review, exact] no_label 0; unreadable 0; unmapped_class 0; unhashable 0. Box level: 4 malformed lines, 2,662 boxes clipped to the frame. | no (unlabelled). [review] rf_tuf's 1,843 (21 % of its images) is recorded, not explained. Whether the label files are empty upstream is checked with its card in L11a. |
| S3 | exact_dup | image | dHash-equal; the alphabetically first slug is kept | 44,845 | per twin, when the dropped twin carries a target box the kept one lacks (H5a) |
| S4 | near_eval | image | within 6 dHash bits of dev, test or an exam | 8,465 (test 5,662; dev 1,744; ood23 784; ood22 271; imageweeds 4) | **guard**: never relaxed globally (H5b) |
| S5 | cwd12_copy | image | within 6 bits of a train_core image | 8,769 | **guard** |
| S6 | Small boxes | box | under `MIN_BOX_PX`, never embedded | 125,500 | not auditable by a crop judge; reported as such |
| S7 | Species join (`cwd12_species.species_of` via `verify.class_join`) | (source, class) | strict alias match; a miss becomes OtherPlant | Of 538,037 embedded OtherPlant boxes, by name status (`verify.other_name_status`): named other plant 316,417; generic 94,580; no name 69,867; numeric 55,657; target-related 1,516 | yes: class maps (L11) and taxonomy (L12) |
| S7b [review] | Name-status rule (`verify.other_name_status`: hand lists `GENERIC_NAME_KEYS`, `NON_PLANT_WORDS`, `CWD12_RELATED_TOKENS`) | (source, class) name | decides which names feed the OtherPlant training sample, and (in this document) the strata of G1, G4, H2 and H4 | Misclassified as "named other plant": BroWeed and NarWeed 197,613 boxes; "crop" 30,042; non-plant objects 2,149; disease or state names 1,403 (§1) [post hoc] | yes. **Audited exhaustively, not sampled.** census_v0 has 140 (source, name) rows, about 100 distinct names. Each name is resolved through J-taxon (§4.3), and a name that does not resolve to a taxon is uninformative. The result is `funnel/name_status_v2.json`, fixed before the draw (F6). `verify.py` is unchanged. |
| S8 | Verifier on target-labelled boxes (`verify.verdicts`) | box | verified if argmax = label, p ≥ τ[k] and cos ≥ σ[k]; thresholds set at 95 % recall on train_core session CV | Of 7,281 boxes: **verified 2,049**; unknown 2,756 (Ragweed 1,833; Waterhemp 598; Palmer 144; Sicklepod 89; …); conflict 2,476. Of the conflicts, 2,001 are confidently OtherPlant (Ragweed 1,142; Waterhemp 489; Palmer 265; MorningGlory 77; …) and 475 another target (461 → PalmerAmaranth). | yes, where the source label is authoritative (H1) |
| S8b [review] | OtherPlant training sample (`verify._other_sample`, `N_OTHER` = 5,000, uniform over S7b-eligible boxes) | box | the probe's reject class | BroWeed is 0.620 of the eligible pool (196,057 of 316,417) [exact, `fit_info.json`]; the drawn composition is recorded by F3 | **not a discard, but it sets S8 and S9.** Audited by the H8 refit (with S7b-v2 eligibility) and by signal S6 |
| S9 | Verifier on OtherPlant-labelled boxes | box | conflict if the probe confidently calls a target | 6,256 conflicts. By predicted class: MorningGlory 2,051; PalmerAmaranth 1,883; Goosegrass 776; Purslane 521; CutleafGroundcherry 328; Waterhemp 286; Carpetweed 157; SpottedSpurge 132; Sicklepod 95; PricklySida 15; Eclipta 10; Ragweed 2. By name status: no name 1,818; numeric 1,796; named other 1,782; target-related 694; generic 166. | only in strata with no name information, and never for a relative of the predicted target (H2) |
| S10 | Image admission (`verify.image_verdict`) | image | admitted iff every species box is verified and every OtherPlant box is other_ok | Images: conflict 5,358; unknown 596. Verified target boxes admitted: 1,060 of 2,049. **989 were lost to the veto**: PalmerAmaranth 346 → 0; Waterhemp 1,044 → 437; Ragweed 27 → 3; MorningGlory 158 → 158. | yes: box-level admission with masking (R-V) |
| S11 | Base selection (`select.py`) | image | source has species crops ("evidence"); retrieval gate 0.10; OtherPlant budget | Of 96,966 admitted images: no_evidence 89,573; below_gate 4,578; no_feature 1,026; OtherPlant budget 911; selected 878. The gate rejects 5.8 % of train_core's own images (pass rate 0.942). | these images go to the increment pool (96,088), not to the bin |
| S12 | Increment evidence criterion (`select.load_evidence`, min 1) | source | at least 1 verified box | 33 of 38 sources; 94,649 of 96,088 pool images | derived from S7 and S8; recomputed after the audit |
| S13 | Zero-shot relevance (`relevance.py`) | source | median P(plant) ≥ τ | not used: it failed its own check (τ 0.0557; 34.5 % of train_core crops below 0.5) | retired (X9) |
| S14 [review] | The realloop gate (`gate.py`, chain arm) | increment | ACCEPT / HOLD / REJECT against the incumbent | realloop_v1: 0 of 6 accepted; gate–truth agreement 2 of 6 (`report.json /agreement/full`) [exact]. P_recipe 0.0 at every step, so the chain could not see data effects. | not a data filter. Its error rate is read from the truth arm (§9.1) and needs no new labels. It is listed so that the negative conclusion's full path is on the ledger. |

### 3.2 The target funnel in one line

Target-labelled boxes in the pool: 7,288 → embedded: 7,281 → verified: 2,049 → admitted: 1,060 → in base B: 854 [post hoc; image-rule query] → about 50 target boxes per realloop_v1 increment [RESEARCH_LOG].

### 3.3 Reconciliation notes

- **Per-prediction counts in the loop state file.** The file gives PalmerAmaranth 2,344, MorningGlory 2,061, … Those counts are taken over all 8,732 conflicts. Over the 6,256 OtherPlant-labelled conflicts alone, PalmerAmaranth is 1,883, because 461 PalmerAmaranth calls fall on target-labelled boxes.
- **Ragweed 1,166 and Waterhemp 938** are all conflicts of those labels. Of these, 1,142 and 489 are confidently OtherPlant.
- **The image-rule loss.** The image-rule query attributed 985 of the 989 lost boxes to a blocking box in the same image:
  - an OtherPlant box called a species: 564;
  - a species conflict: 276;
  - an unknown species box: 145.

  The remaining 4 are most likely the 4 images that `admit_summary.json` records as `images_unknown_only_for_small_species_boxes`: the query read crops.csv, which leaves out boxes below `MIN_BOX_PX`, so it could not see those images' blocking box. census_v1 (F3) confirms this exactly. The lost boxes sit in 457 images, with a median of 16 boxes per image.

### 3.4 Where the discards concentrate

**Conflicts by source.** greenhouse 1,744; rf_tuf 1,607; mh_weed16 1,576; weed_crop 1,215; aiml-35 643; imageweeds_aerial 540.

**Sources whose embedded boxes are more than half uninformative names:** 21 sources in all. The largest are:
- rf_tuf: no names; 6,662 boxes; 1,607 conflicts.
- mh_weed16: ids '0'–'14'; 46,811 boxes; 1,576 conflicts.
- rf_test-gzc3r crop-mfete: no names; 20,353 boxes; 210 conflicts.
- rf_srec crop-weed-poxtn: '0'–'3'; 5,132 boxes; 152 conflicts.
- rf_school: '0'; 3,443 boxes; 67 conflicts.
- rf_test-8qezo: '0', '1'; 271 boxes.
- The non-plant sources: csgo, uav-wqshy, tomato-leaf and crop-health-advisor.

**[review] What the filters kept is mostly cwd12-derived [exact; census_v0, `realloop_v1/build_summary.json`].**
- Of the 2,049 verified target boxes, 941 (45.9 %) come from two Roboflow sources with near_eval hits on cwd12's dev and test:
  - `rf_karthikeya-c8pvy__weed-detection-cwp10`: 647 verified; near_eval dev 12, test 120; 200 cwd12 copies;
  - `rf_zig-zag-lnodr__weed-detection-vanpe`: 294 verified; near_eval dev 6, test 53; 84 cwd12 copies.
- Of base B's 854 harvested target boxes, 731 (85.6 %) come from these two sources. Of B's 878 harvested images, 812 do.
- cwp10 also names *spurred anoda* and *swinecress*, two CottonWeedID15 classes that are not cwd12 species. It may therefore derive from the Lu lab's CottonWeedID15 collection rather than from cwd12 itself [unverified]. The CottonWeedID15 class list, with SpurredAnoda and Swinecress, was checked on its AgML card (https://huggingface.co/datasets/Project-AgML/CottonWeedID15_classification). vanpe's names (Carpet weed, Crabgrass, Eclipta, Goosegrass, Morning glory, Nutsedge) are also CottonWeedID15 classes.
- **Consequences:**
  - H6b (a leak of augmented dev or test copies into B) bears on realloop_v1's own base and conclusion, not only on recovery.
  - The "admitted" side of the audit is in-domain almost by construction, so recall under shift is measured only on the discarded side.

**[review] Sources that kept images and have at least one near_eval hit** (`pool_summary.json per_slug`). These are the candidates for re-exports with augmented copies that 6 dHash bits cannot see:
- rf_tuf (ood22 3, ood23 10);
- peradeniya (ood23 337);
- 5nvic (ood23 150);
- test-8qezo (ood23 142);
- cwp10 (dev 12, test 120, imageweeds 2);
- vanpe (dev 6, test 53);
- latvia (ood23 4);
- srec fqrtg (imageweeds 2, ood23 2);
- crop-health-advisor (ood23 2);
- csgo (ood23 2; probably false hits);
- uav-wqshy (ood22 1);
- yolo-gyfyq (ood23 1);
- itmo (ood23 1);
- weeds-qftz4 (ood22 1, ood23 1).

H6 covers all of them (§6).

---

## 4. Judges and their calibration on known truth

### 4.1 Known-truth sets

| Set | Content | Allowed use |
|---|---|---|
| KT1 | train_core crops: 5,029 boxes, 12 species, with capture sessions | in-domain calibration, leave-session-out |
| KT2 | cwd12 copies (`calibration.json`). The `rf_agrobot` wildcard copy holds about 4,743 boxes that are truly targets but labelled OtherPlant; the verifier called 83.3 % of them a conflict and 792 other_ok. Under the old wrong joins, the copies' true labels are known. | planted positives for hidden targets and for wrong class joins (H0) |
| KT3 | the three_season 2021 copy, with true names | in-domain; a re-exported source |
| KT4 | NDSU target boxes, 4,639 in all: `project_agml__weed_crop_detection` (Ragweed 1,086, Waterhemp 695) and `project_agml__greenhouse_crop_weed_detection` (Waterhemp 1,140, Palmer 755, Ragweed 963). The papers name the species: *Ambrosia artemisiifolia*, *A. tuberculatus*, *A. palmeri* (PMC11986624, PMC11599996). | shifted positives. ~~Split 50/50 by image: one half calibrates judges, the other estimates.~~ [review] **Not split. KT4 is the population H1 estimates** (note K1). Until H1 is evaluated its labels are *claimed*, not known (note K2). The source labels' own precision is measured (H1). |
| KT5 | Named non-targets from the same papers: redroot pigweed (*A. retroflexus*), kochia, horseweed and the crops. Also MH-Weed16's non-target ids from its card: *Euphorbia hirta* and *E. hypericifolia* (ids 8, 11), *Chamaecrista pumila* (14), *Digitaria* (9), *Cynodon* (13), *Cyperus rotundus* (1), and others. | specificity against probe attractors. [review] These labels are also *claimed*. H2b asks whether they are right, so KT5 never qualifies the RL for H2b or H4. KT5 boxes of a source never calibrate a judge that estimates that source's strata (disjointness below). |
| KT6 | MH-Weed16 card-resolved classes: id 12 = *Senna obtusifolia* (Sicklepod); id 5 = *Ipomoea obscura* (MorningGlory by class policy). Source: Data in Brief 61:111691, Table 2, PMC12179629, with the author class list in github.com/SPS-Dataset/MH-Weed16. | shifted positives, **only after the exact id check of H3a passes**. ~~split 50/50 by video~~ [review] Split by near-duplicate group (3 bits). The article describes still images, and no video structure is documented (§2.5), so video ids are used only if the upstream file names show them. The GitHub page fetched on 2026-09-28 did not display the author class list [unverified]; the class table is taken from the article's Table 2. KT6 never qualifies the RL for H3a's purity test. |
| KT7 [review] | **Independent species truth under shift.** iNaturalist Research Grade observations (§2.5) of the 12 targets and of the KT5 attractor taxa (*A. retroflexus*, *A. hybridus*, *Bassia scoparia*, *Erigeron canadensis*, *Ipomoea* spp., *Euphorbia hirta*, *Chamaecrista*, *Digitaria*, *Cyperus*), filtered to seedling or vegetative stage and drawn from US and Indian observations. About 30 per taxon, fetched and hashed by L11a, with observation ids and licences recorded. | **Qualifies the RL** at species and genus level (sentinels), and qualifies J-knn1, J-knn2 and J-vlm. **Never J1 or J-zs**: BioCLIP-2 trains on GBIF citizen-science images (§2.5). Not training data. |
| Forbidden | dev, test, ood22, ood23, ImageWeeds ground (`project_agml__imageweeds_weed_detection`) | never used to calibrate a judge, choose a threshold or label a sample |

**[review] Notes to the table.**
- **K1, KT4 is not split.**
  - An image split would break the disjointness rule below, which forbids a shared source and, since this review, a shared annotating lab.
  - The greenhouse system photographs the same potted plants on several days, so an image split would also share individual plants across the halves.
  - A judge used on an NDSU stratum is therefore calibrated without NDSU data: on KT1–KT3, KT6 after H3a, and KT7.
  - A weed_crop ↔ greenhouse cross-fit may be reported as a sensitivity analysis, labelled "same lab". It never qualifies a judge.
  - After H1 is supported, KT4 may shift-calibrate thresholds (H8, H9) for non-NDSU strata only.
  - Any within-source draw uses the seed `stable_int("funnel/v1/kt4")`.
- **K2, claimed truth.** KT4, KT5 and KT6 hold source or card labels, which H1, H2b and H3a test. They never qualify the RL for the hypothesis that tests them. They never shift-calibrate a threshold (H8, H9) before that hypothesis is supported.
- **K3, imageweeds_aerial.** This third NDSU source names its species, but it is not in KT4, because it shares provenance with the ImageWeeds exam (H6c).

**Disjointness, enforced in code (`funnel/qualify.py`, replay R12).** A judge or threshold calibrated on a set may not be used to estimate a stratum that shares with that set any of:
- a source;
- a near-duplicate group (3 dHash bits);
- a provenance group (a cwd12 copy and its train_core twin; the frames of one video);
- [review] an annotating lab. weed_crop, greenhouse, imageweeds_aerial and the ImageWeeds exam are one NDSU group. cwd12, CottonWeedID15, 3SeasonWeedDet10 (all Y. Lu's lab, §2.5) and the Roboflow re-exports with cwd12 hits (cwp10, vanpe) are one Lu-lab group. Without this, the one lab's labelling convention leaks between halves.

**[review] What "calibrated on" covers:** fitting, threshold choice, qualification, a kNN bank, and the exemplar strips shown to the RL.

**[review] Truth used to qualify the RL for a hypothesis never comes from the population that hypothesis tests.** This is the circularity rule:
- H1 (NDSU labels): qualify on KT1–KT3 and KT7, not KT4.
- H2b and H4 (named labels): qualify on KT1–KT3 and KT7, not KT5.
- H3a (card map): qualify on KT1–KT3 and KT7, not KT6.

Agreement with the claimed labels is still reported, as a description.

### 4.2 The verifier (judge J1) on known truth

**In domain.**
- On the cwd12 copies: verified precision 1.0 under the current join and 0.9996 under the old wrong joins.
- On seeded label swaps: 0.9997.
- Recall on correct labels: 0.9457 on the copies and 0.9473 on the swaps.

**Under shift, on KT4 [exact].**

| Source | Label | Boxes | Verified | Main confident call when not verified |
|---|---|---|---|---|
| weed_crop | Waterhemp | 695 | 381 | Palmer 102 |
| weed_crop | Ragweed | 1,086 | 13 | unknown 895 |
| greenhouse | Waterhemp | 1,140 | 101 | OtherPlant 411, Palmer 295 |
| greenhouse | Palmer | 755 | 346 | OtherPlant 265 |
| greenhouse | Ragweed | 963 | 14 | OtherPlant 493 |
| **all** | | **4,639** | **855 (18.4 %)** | |

**Calibration-transfer gap.** CTG = in-domain recall − recall on KT4 = 0.9457 − 0.184 = 0.76.

**On KT6 [post hoc; card-resolved].**
- id 12 (307 pool boxes): Sicklepod is the argmax for 108 (35 %) and a confident call for 50 (16 %).
- id 5 (1,634 boxes): MorningGlory is the argmax for 1,304 (80 %) and a confident call for 1,007.

**Attractors on KT5 [post hoc].** These are confident target calls on boxes whose source names a non-target:

| Source name (taxon) | Called as | Share of that source class |
|---|---|---|
| redroot pigweed | PalmerAmaranth | 0.25–0.69, varying by source |
| cotton | MorningGlory | 0.63 |
| crabgrass | Goosegrass | 0.16 / 0.54 |
| kochia | Waterhemp | 0.14 |
| black bean | MorningGlory | 0.10–0.13 |
| nutsedge | Goosegrass | 0.14 |

**A design cause of the attractors.** `verify.CWD12_RELATED_TOKENS` keeps every pigweed, amaranth, spurge, morning glory and similar name out of `_other_sample` (`N_OTHER` = 5,000). The probe has therefore never seen a non-target from a target's own genus labelled OtherPlant. Whether the OtherPlant sample is also confounded with source identity is pending: its source list is in `step1/verifier/fit_info.json` on the cluster.

**[review] Two more design causes.**
- **The OtherPlant sample is expected to be about 62 % "BroWeed"** (§1). The probe's reject class was then learned mostly from one maize-field source's generic broadleaf-weed boxes.
- **An OtherPlant call is cheaper than a target call.** In `verdicts()`, confident(j) needs cos_j ≥ σ[j] only when j is a species. OtherPlant has no prototype gate.
  - The contact sheet `funnel/sheets/greenhouse_Ragweed_conflict.jpg` captions confident OtherPlant calls at p = 0.30. So τ[OtherPlant] ≤ 0.30 [post hoc]. [exact, `step1/verifier/thresholds.json`] τ[OtherPlant] = 0.3017 and σ[OtherPlant] = 0.7046; `verdicts()` never applies σ[OtherPlant]. The species τ range from 0.40 (Waterhemp) to 0.78 (Sicklepod).
  - A target-labelled box can therefore become a confident "conflict → OtherPlant" at a probability no target call could pass.
  - The F3 diagnostic below records, for each such conflict, P_OtherPlant, τ[OtherPlant] and the probe's second choice.

**[review] Viewing that sheet is not a measurement, but it shows H1 can fail.**
- About 20 of its 30 crops show finely divided ragweed leaves.
- About 10 show a grass, another broadleaf plant, or an empty pot or wall.

So box validity, not only species identity, may decide H1.

### 4.3 The judge panel

| Judge | Shares with J1 | Role |
|---|---|---|
| J0: source label | nothing | truth for KT4 after H1; a prior elsewhere |
| J1: BioCLIP-2 probe | — | **never re-judges its own discards**; used only as a stratum key and as the G4 prioritisation score (Horvitz–Thompson weights keep that unbiased) |
| J-card: the class table in a data paper or card | nothing | class-level truth (MH-Weed16 Table 2; NDSU Table 2) |
| J-taxon: GBIF backbone (`api.gbif.org/v1/species/match`; version and responses cached) | nothing | resolves names to taxa. Scientific names resolve (*Amaranthus rudis* → *A. tuberculatus*; *Cassia obtusifolia* → *Senna obtusifolia*). Vernacular names are ambiguous ("goosegrass" is also *Galium aparine*; "pigweed" is also *Portulaca oleracea* and *Chenopodium album*), so a vernacular match alone never maps a class. GBIF keeps *Eclipta alba* separate from *E. prostrata*, which the project's alias table joins, so project policy overrides the backbone and the overrides are R4. |
| J-zs: BioCLIP-2 zero-shot | image encoder | Taxonomic + common-name prompts over a closed candidate set: 12 targets + the named non-targets in the pool + non-plant. For prioritisation. Reuses the existing embeddings and `relevance.BioclipTextEncoder`. |
| J-knn1: DINOv2 kNN to train_core crops, leave-session-out | reference domain | expected to be domain-bound, like J1 |
| J-knn2: DINOv2 kNN, multi-domain bank (train_core + KT4 + KT5 + KT6 calibration half + KT7; [review] every bank entry obeys the disjointness rule for the stratum being judged), ~~leave-source-out~~ [review] **leave-lab-out** | partly; [review] **not independent of J0** where its bank holds the same lab's claimed labels | best candidate for the shifted strata. Built from species-named crops only, never from the legacy `object_bank/`. [review] For an NDSU stratum its bank holds no NDSU crop, and for mh_weed16 no mh_weed16 crop. Otherwise J-knn2 partly echoes J0, which H1 and H3a test. |
| J-vlm: a cluster VLM (`model_router` role `labeling_vlm`, or the planner model if its vision capability is confirmed [unverified]) | nothing | Closed multiple choice: 12 targets + "other plant" + "not a plant", with reference strips. Tie-breaker only. [review] `labeling_vlm` resolves to `ollama:minicpm-v` (fallbacks llama3.2-vision, moondream; `model_router.py ROLES`). A small general VLM is not expected to qualify at species level on seedlings (VLM4Bio, §2.2). |

**Reference labeller (RL).** The RL gives the labels every estimate rests on.
- **Task.** Verify only: yes, no or unsure on a proposed label, plus "box covers one plant of this class: yes/no".
  - [review] **A machine RL answers blind multiple choice instead:** the 12 targets, the stratum's named attractors, "other plant", "not a plant", "box invalid" and "unsure", with no proposed label.
  - The reason: in G1 the proposed label *is* J1's prediction, so verify-only lets the filter's verdict reach the RL through the question.
  - A person keeps verify-only, for cost (Papadopoulos). A random 20 % of each person's items are asked as blind multiple choice, and the yes-rate difference between the two formats is reported as the anchoring effect.
- **What it sees.** The crop at ≥ 224 px, a context thumbnail with the box drawn, 6 train_core exemplars of the proposed class, and 6 exemplars of that class's main attractor from KT5.
- **Blinding.** It never sees the source name, the verdict, the score or the stratum.
- **Backends.** The L14 verify-only queue (a person), or a strong multimodal model outside every judged model family. Each label row records `labeller`.
  - [review] **Unresolved, and on the critical path.** No such model is in `model_router.ROLES`: the cluster VLMs are minicpm-v, llama3.2-vision and moondream. Who the RL is must be decided, as an R4 decision recorded before F7, because every estimate rests on it.
  - An external API model may see pool and known-truth crops only. G5 pairs, which include evaluation images, go to a person or to a cluster-local model (§5.2).
- **Qualification on sentinels.**
  - 20 % of every sheet is KT1/KT4/KT5/KT6 truth, and a quarter of those carry a deliberately wrong proposal (to measure anchoring).
  - [review] **Sentinel truth that counts toward qualification** is KT1–KT3 and KT7 only (the circularity rule, §4.1). KT4–KT6 sentinels are shown and scored, but only as agreement.
  - [review] **Wrong proposals** are drawn from J1's attractor pairs (for example, "PalmerAmaranth" on a KT7 *A. retroflexus*), not uniformly. A uniform wrong label is easy to reject and would overstate Sp.
  - [review] **Prevalence.** Sheets interleave strata, so no sheet has a target prevalence below about 20 % (Wolfe et al. 2005, §2.5). G4 items, whose true prevalence may be around 1 %, are never sheeted alone. Se is estimated from sentinels shown at the same sheet prevalence.
  - To serve as reference at species level, sensitivity (Se) and specificity (Sp) must both have lower bounds ≥ 0.85 on sentinels of the same kind.
  - Otherwise the RL is demoted to genus, then to plant vs non-plant, and every estimate states the level.
  - Pre-registered expectation: *Amaranthus* species may qualify only at genus level. H1 is then evaluated at genus for *Amaranthus* labels.
- **Unsure answers** are reported under both assignments.
- **Adjudication.** A person may adjudicate up to 50 disagreements between the RL and the best qualified judge. Without a person, those items are reported under both assignments.

**Qualification of a machine judge** (pre-registered; per stratum type, on the matching known truth):
- precision of its "target k" calls: lower bound ≥ 0.85 against the KT5 attractors; [review] for H2b and H4 strata, against the KT7 attractors instead;
- rescue rate (accuracy on J1's errors): lower bound ≥ 0.5; [review] measured on J1's errors on KT2 and KT7, where the truth does not come from the audited sources, and reported separately on KT4 as agreement;
- also reported: P(judge wrong | J1 wrong) next to P(judge wrong).

Two judges whose errors on known truth correlate at φ ≥ 0.5 count as one. A judge that fails qualification still prioritises sampling and still serves as a PPI predictor: a bad predictor widens the interval but cannot bias it.

**The first diagnostic needs no judge [exact, F3].** For each non-verified target-labelled box, census_v1 records which condition of `verdicts()` failed:
- argmax ≠ label;
- p < τ[k];
- argmax right but cos < σ[k] (the train_core prototype gate).
- [review] for a conflict whose argmax is OtherPlant: P_OtherPlant, τ[OtherPlant] and the second-ranked class with its probability.

This separates "the probe is wrong under shift" from "the prototype gate is bound to its domain", and [review] from "the reject class is too easy to enter".

---

## 5. Sampling and statistics

### 5.1 Units, the definition of a false negative, and class policy

**Units:**
- a box, for S6–S9;
- an image, for S2–S5 and S10;
- a (source, class) pair, for the join and class maps;
- a source, for S12.

**False negative (FN):** a discarded or mislabelled unit that is truly a correctly boxed target under `CWD12_BINOMIAL`. "Correctly boxed" means the box covers one plant of that class, as the RL answers.

**Class policy (pre-registered; changing it is card X10, R4, never after reference labels exist):**
- **MorningGlory = *Ipomoea* spp.** Any *Ipomoea*, including *I. obscura*, counts.
- **Every other class is one species:**
  - SpottedSpurge = *E. maculata* (*E. hirta* and *E. hypericifolia* are not);
  - Ragweed = *A. artemisiifolia* (giant ragweed, *A. trifida*, is not);
  - Sicklepod = *S. obtusifolia* (*Chamaecrista* is not);
  - *A. retroflexus* is not a target.

**Path-aware attribution.** Every unit carries its discard-path vector: its outcome at every stage. Three counts are reported:
- first-cause counts;
- sole-cause counts: the unit would pass if only this stage were fixed;
- joint recovery under each named recovery policy of §7.

Without this, recovery double-counts. For example, an NDSU Ragweed box fixed at the verifier is still vetoed at admission and excluded by S12.

**[review] Counting an answer given at genus level** (pre-registered, because the RL may qualify only at genus for some genera):
- A genus answer counts as a target only when the class policy defines the class at genus level. Today that is *Ipomoea* → MorningGlory.
- *Amaranthus*, *Euphorbia*, *Ambrosia* and *Senna*/*Chamaecrista* hold both targets and named non-targets in this pool. A genus answer there is "unsure" and is reported under both assignments.
- So H2a and H1 cannot be supported by genus-level answers for these genera. They can only be bounded.

### 5.2 Strata and budget

| Group | Frame (N) | Strata | Planned n (minimum) |
|---|---|---|---|
| G0 planted control | A mixed stratum drawn from KT2 hidden targets and KT5 named non-targets, at a seeded share hidden from everyone until estimation | one | 120 (60) |
| G1 OtherPlant → target conflicts | 6,256 boxes | name status (5) × predicted class (top 3 + rest). [review] Name status is `name_status_v2` (S7b), so "crop" (372 conflicts), BroWeed and NarWeed (33) and disease or object names (7) leave the "named other" stratum. | 300 (160) |
| G2 target-labelled, not verified | 5,232 boxes | source × label × failure mode (§4.3) | 300 (150) |
| G2v verified boxes in vetoed images | 989 boxes | source | 60 (30) |
| G2a [review] verified target boxes from sources outside the Lu-lab group | 1,108 boxes [exact]: imageweeds_aerial 253, weed_crop 394, greenhouse 461. The ones in vetoed images are shared with G2v; they are drawn once and carry joint inclusion probabilities. | source | 60 (30). The other direction of the audit: the verifier's precision under shift has never been measured. It is also the numerator of funnel recall. |
| G3 class level | (source, class) rows of uninformative, numeric and target-related names, plus visual clusters (k = 8, fixed before labelling) of each no-name source | class or cluster; 20–30 crops per unit of ≥ 100 boxes | 450 (250) |
| G4 other_ok | 531,781 boxes | name status × argmax-is-target × judge score band | 450 (300): prioritised draws with π ∝ judge target score, plus 300 uniform |
| G5 guard pairs | near_eval and cwd12_copy pairs from sources not derived from cwd12: all 13 rf_tuf hits, the 2 csgo hits, and 20 each from peradeniya, 5nvic, test-8qezo and vitif; plus exact_dup twins whose sha256 differs | source × split × dHash bits (0–2, 3–4, 5–6) | 160 pairs (100) |
| Sentinels | KT1, KT4, KT5, KT6, [review] KT7 | kind | about 400; [review] + about 150 KT7 |
| **Total** | | | **about 2,240 (about 1,450)**; [review] **about 2,450 (about 1,630)** with G2a and KT7 |

**[review] G5 shows evaluation images.** Judging "same photograph or consecutive frame" requires looking at the dev, test or exam image. That departs from the invariant of §8.7, which is amended there:
- the G5 sheets are rendered and judged only on the cluster, by a person or by a cluster-local model;
- the evaluation image never leaves the cluster and never reaches an external API;
- a G5 answer is used only for a guard decision, pair by pair, never as a label.

**If the budget is short.** G0 and the sentinels always run. The rest follows in the order G3 → G2 (with G2v) → G1 → G5 → G4, because one class decision can recover hundreds of boxes.

**The draw.**
- Seeded with `common.stable_int("funnel/v1/<group>/<stratum>")`.
- Every item's inclusion probability is written into `funnel/sample_v1.csv`.
- The file's sha256 is appended to `prereg_v1.json` as the "sample lock" amendment before any RL label is made.

### 5.3 Sample-size reference

| n | One-sided 95 % upper bound when 0 of n are positive | Wilson half-width at p = 0.5 | at p = 0.2 | at p = 0.1 |
|---|---|---|---|---|
| 30 | 9.5 % | 0.168 | 0.139 | 0.111 |
| 60 | 4.9 % | 0.123 | 0.100 | 0.077 |
| 100 | 3.0 % | 0.096 | 0.078 | 0.060 |
| 200 | 1.5 % | 0.069 | 0.055 | 0.042 |
| 400 | 0.7 % | 0.049 | 0.039 | 0.030 |

### 5.4 Estimators

All estimators live in `funnel/estimate.py`, the only producer of audit numbers. Nothing is counted by hand.

- **Per stratum:** p̂_h with a Wilson interval (Jeffreys when n_h < 30).
- **Aggregate:** θ̂ = Σ W_h p̂_h with W_h = N_h / N, and a finite-population correction. The interval is Korn–Graubard (Clopper–Pearson at the effective n).
- **Recoverable count:** R̂ = Σ N_h p̂_h.
- **Decisions** use one-sided 97.5 % bounds.
- **Prioritised draws:** Horvitz–Thompson with the recorded inclusion probabilities π_i.
- **Judge-assisted estimates:** stratified PPI, θ̂ = mean of f over all units + mean of (y − f) over labelled units, with PPI++ power tuning. The predictor f is the best-qualified judge in each stratum. PPI is used where it narrows the interval; otherwise the estimate is plain HT.
- **Reference-labeller error:** corrected with Rogan–Gladen, θ = (θ_obs + Sp − 1) / (Se + Sp − 1), with melded intervals (Bayer, Fay & Graubard, https://arxiv.org/abs/2205.13494). When Se + Sp − 1 < 0.7, the uncorrected estimate is reported with a flag.
- **Clustering:** a seeded cluster bootstrap by image (2,000 resamples) gives the design effect. Aggregates that span sources are bootstrapped by source.
- **Funnel recall for target boxes:** admitted true targets / (admitted true targets + Σ over discard strata of R̂_h), with Webber's beta-binomial Monte Carlo interval. Small boxes (S6) are outside the denominator and reported separately.
- **Label frequency per source (Elkan–Noto c):** among the RL-confirmed targets of a source, the share the join labelled as a target.
- **Screening across stages.** A stage is "suspect" when both hold:
  - the Holm-adjusted one-sided test of FN rate > 0.10 rejects;
  - the lower bound of R̂ is ≥ R_min.

  R_min is 500 boxes, about 25 % of the 2,049 verified. For a class with fewer than 200 train_core boxes it is 100: CutleafGroundcherry 50, Goosegrass 107 and Sicklepod 121 have fewer. The confirmatory hypotheses of §6 are each tested at one-sided 0.025.
  - [review] **Except H10**, whose "helps" rule (P ≥ 0.75 over 3 × 3 seeds) has a false-positive rate of 0.20 under no effect. With 3 seeds per arm, an exact rank test cannot reach one-sided 0.025 at all: its smallest p is 1/20 = 0.05. H10 therefore adds a 5-seed permutation test (§6 H10a, §9).
- **[review] Funnel recall's numerator** uses the precision of the admitted target boxes. That is the in-domain calibration for the Lu-lab-group sources, and G2a for the rest. It is never assumed to be 1.

---

## 6. Pre-registered hypotheses

**Common rules:**
- Every hypothesis is reported with the same prominence whichever way it comes out. "The filter was right here" is a result.
- The campaign starts from the suspicion that the filters were wrong. The symmetric falsifiers below are what keep that suspicion from deciding the outcome.
- Measurements come from `funnel/audit_v1.json` (estimation) and `funnel/census_v1.json` (exact) unless stated.

**H0 — the audit sees known failures (backtest; this gates everything).**
- (a) On G0, the 95 % interval of the estimated target share covers the planted share.
- (b) The class-relation audit of L11, run with names hidden:
  - maps every class of cottonweed_holdout and of the three_season 2021 copy that has ≥ 20 boxes to its true species;
  - flags cottonweed_holdout's old wrong join;
  - does not flag cottonweed_sp8's current join.
- *Falsified* if (a) misses or any class in (b) is wrong. The campaign then stops at F8: the estimator, the RL or the relation audit is defective, and nothing downstream is reported as a result.
- **[review] What H0 does and does not test.**
  - (b) runs on copies of train_core photographs, the reference domain itself, so it can fail only through a bug. It is a pipeline check, not validity under shift.
  - (a) as first written drew its negatives from KT5, whose labels are claimed (§4.1 K2), and its positives from in-domain KT2. G0 now draws its negatives from KT7 attractor taxa, and half its positives from KT7 target taxa, so the planted share has independent truth under shift.
- **[review] H0(c), reported; it gates only L11(b) visual class maps.** The relation audit, run with names hidden on a shifted named source outside the NDSU group (the aiml-35 named classes, or mh_weed16 after H3a):
  - must not map a named relative to a target (for example, *Euphorbia prostrata* or Crab Grass);
  - is reported with its class-level accuracy.
  - If it fails, L11 applies card maps only (J-card plus the H3a-type exact geometry match), and visual maps stay proposals for L14.

**H1 — where the verifier rejected paper-confirmed target labels, the labels are right** [the 18.4 % is exact and already seen; the precision is new].
- *Measurement:* RL precision of the source label (label right and box valid) on non-verified NDSU target boxes, from G2 ~~on the KT4 estimation half~~ [review] on G2's NDSU strata (KT4 is not split, §4.1 K1). Reported overall and per label, and [review] separately for "label right" and "box valid".
- **[review] H1-pre, exact, before any RL label.** The pool's weed_crop and greenhouse label files are matched by box geometry to the upstream Mendeley annotations (10.17632/mthv4ppwyw.2 and hs7d7kpd3z/2), as H3a does for mh_weed16.
  - The registry's class lists are alphabetical (Blackbean … Waterhemp). The order in the files is unchecked (`pilot.CLASS_ORDER_NOTE`).
  - An off-by-one order would produce exactly the observed Ragweed pattern. If the ids disagree, the finding is a join error, fixed through L11, and H1 is evaluated on the corrected labels.
- **[review] The RL's qualification for H1** uses KT1–KT3 and KT7 sentinels only (§4.1).
- *Supported* if the lower bound is ≥ 0.85 overall and for Ragweed.
- *Falsified* if the upper bound is < 0.85 overall. The verifier's rejections were then mostly correct.
- *Otherwise* inconclusive.
- *Consequence:* if supported, REC-AUTH (§7) is allowed at the measured precision, the CTG is recorded, and card X11 is raised.

**H2 — conflicts are hidden targets where names carry no information, and relatives where they do** [post hoc for mh_weed16, weed_crop and greenhouse].
- *H2a.* In the strata with no name information (no name 1,818 + numeric 1,796 + generic 166 = 3,780 boxes), the share that is truly a target is ≥ 0.5. *Falsified* if the upper bound is < 0.5.
- *H2b.* In the named strata (named other 1,782 + target-related 694 = 2,476 boxes), the share that is truly a target is ≤ 0.15. *Falsified* if the lower bound is > 0.15.
- **[review] The strata use `name_status_v2` (S7b), fixed before the draw.**
  - **H2a frame:** no name, numeric, and generic or unresolvable names, including BroWeed and NarWeed.
  - **H2b frame:** taxon-resolved names, target-related names, and role names ("crop"). Role names are reported as their own sub-stratum.
  - The v1 counts above are kept for reference. The confirmatory frames are the v2 frames, and their sizes are written into `prereg_v1.json` at the sample lock.
  - Genus-level answers follow §5.1: for *Amaranthus*, *Euphorbia* and *Ambrosia* they are "unsure".
- *Also reported:* per stratum, the precision of "relabel to the probe's prediction".
- *Consequences:*
  - H2b supported → named strata are never relabelled (the sibling guard).
  - H2a supported → those strata become R-J candidates under the gates of §7.

**H3 — sources with anonymous class names hold targets.**
- *H3a (exact first, then confirmatory; post hoc).*
  - Our pool's mh_weed16 label files are matched by box geometry to the upstream annotations (Mendeley d3n3mgjjbv v2, CC BY 4.0; fetched and hashed by L11a).
  - *Supported* if all of the following hold:
    - ≥ 95 % of pool boxes find a geometry match;
    - ≥ 99 % of matched boxes carry the same id under the identity alignment (AgML id = upstream id);
    - RL purity is ≥ 0.8 (lower bound ≥ 0.6) for id 12 = *Senna* and for id 5 = *Ipomoea*.
  - *Falsified* if exact agreement is < 99 % under every alignment tried (identity, ±1), or if either purity lower bound is < 0.6.
  - [review] **Alignments to try.** The article lists 16 classes (0–15) and the AgML card 15 (§2.5), so AgML dropped one class. If it dropped a class other than the last, the ids above it shift by −1, and neither a global identity nor a global ±1 alignment fits.
    - The alignment set is therefore the 16 monotone "drop class d" maps, plus identity and ±1.
    - The best map is chosen by geometry agreement on a seeded half of the matched boxes and confirmed on the other half.
  - [review] **Qualification.** RL purity for H3a is qualified on KT1–KT3 and KT7 sentinels, never on KT6 (§4.1).
- *H3b (confirmatory).* Frame: the anonymous sources other than mh_weed16 with ≥ 100 embedded boxes (rf_tuf, crop-mfete, poxtn, school, test-8qezo, crop-health-advisor, uav-wqshy, csgo, tomato-leaf). At least one of their classes, or one visual cluster in a no-name source, has a target-prevalence lower bound ≥ 0.5.
  - The platform fetches and hashes each source's card or project class list before any RL label (L11a).
  - *Falsified* if every upper bound is < 0.3.

**H4 — targets are rare among other_ok boxes.**
- The true-target prevalence in other_ok has an upper bound < 2 % in the named-other strata and < 10 % in the no-information strata.
- ~~*Falsified* otherwise~~ [review] *Falsified* if the lower bound is ≥ 2 % (named-other) or ≥ 10 % (no-information): the OtherPlant conflict rule is then too strict for no-information sources. *Otherwise* inconclusive. As first written, an under-powered sample would have "falsified" H4.
- [review] **The named-other stratum uses `name_status_v2`.** Under v1 it is dominated by BroWeed boxes, on which the probe's OtherPlant verdict is fixed by its own training sample (§1).
- [review] **Power.** With n = 225 in a stratum, the one-sided 97.5 % upper bound is below 2 % only when at most 1 target is found. The stratum budget is set so the bound can be reached.

**H5 — the dedup and eval guards are not material FN sources.**
- *H5a [exact].* Twins dropped by exact_dup that carry a target box the kept twin lacks: fewer than 100 boxes in total.
  - [review] Also reported, exactly: dropped twins whose class names are more informative under `name_status_v2` than the kept twin's (for example, a species-named fork dropped in favour of a numeric-named one). The alphabetically first slug is kept. Forks are common: seven slugs each lose 1,655 of their 1,661 images to exact_dup [exact, `pool_summary.json per_slug`].
- *H5b.* In sources not derived from cwd12, ≥ 80 % of near_eval hits are the same photograph as the eval image, or a consecutive frame of it. *Falsified* if the upper bound is < 0.8.
  - The guard radius is never lowered. A false hit may be recovered only pair by pair, with a person's sign-off.

**H6 — leak check, two-sided (it routes recovery rather than being supported or falsified).**
- *Detector.* A robust copy detector: SSCD descriptor (DINOv2 as fallback), plus dHash of flipped and rotated variants.
  - Calibrated before it judges anything: recall ≥ 0.95 on the 8,769 matched cwd12 copies (positives), and false-positive rate ≤ 1 % on distinct pairs at 7–10 dHash bits (negatives).
  - [review] **That calibration is circular in both directions.**
    - The 8,769 positives are the copies 6-bit dHash already found, the easy ones, so recall on them says nothing about the copies dHash misses.
    - A pair at 7–10 bits may itself be an augmented copy, which is exactly what H6 looks for.
  - [review] **Calibration instead:**
    - *Positives:* train_core images (never evaluation images) put through the augmentations Roboflow and Keras exports apply: flips, 90° rotations, crops of 0–20 %, brightness ±25 %, blur, shear, letterbox resize to 640 and JPEG re-encoding. The ImageWeeds paper applies such augmentations itself (§2.5). Recall ≥ 0.95 per augmentation family.
    - *Negatives:* pairs at 7–10 bits whose sources share no lab, country or collection (for example cwd12 train_core × MH-Weed16). FPR ≤ 1 %.
- [review] **Scope.** (a) is run for every source that kept images and has a near_eval hit (§3.4: 14 sources), not only rf_tuf. A source with a detected copy is quarantined from recovery as a whole, not image by image, because its undetected copies are the ones a detector misses.
- *(a)* Does rf_tuf hold at least one augmented copy of a never-train image? If yes:
  - rf_tuf is quarantined from recovery;
  - ood numbers of any arm that contains rf_tuf are void;
  - realloop_v1's UNVERIFIED "helps" is re-read with that finding.
- *(b)* Do base B and realloop_v1's increments hold no augmented copy of dev or test? The risk sources are cwp10 (dev and test near hits 12 and 120) and vanpe (6 and 53). If a copy is found, it is an R4 incident: B is rebuilt, and realloop_v1's dev results are void.
  - [review] (b) also covers ood22, ood23 and ImageWeeds. 812 of B's 878 harvested images come from cwp10 and vanpe (§3.4).
- **[review] (c) Provenance (same lab, site and season), which no copy detector can see** (Kapoor & Narayanan L3.2, §2.5). Two groups are recorded:
  - **NDSU:** weed_crop, greenhouse, imageweeds_aerial and the ImageWeeds exam. They share sites, seasons and camera model, and possibly the same greenhouse.
  - **Lu lab:** cwd12, CottonWeedID15, 3SeasonWeedDet10 (whose 2021 subset is derived from cwd12), cwp10, vanpe, and any source H6(a) links to them.
  - **Routing.** An arm whose training set holds a group's *harvested* images reports that group's exams as "same-lab", not out-of-distribution.
    - train_core itself does not trigger it: the train_core → ood22/ood23 relation is the designed temporal shift.
    - B already holds 49 weed_crop and 17 imageweeds_aerial images, so B's ImageWeeds number is "same-lab" too.
  - This is recorded before realloop_v2 runs. It does not block recovery, because exams make no decisions (§9.3).

**H7 — judges.**
- J-zs and J-knn1 fail qualification on ~~KT4/KT5~~ [review] KT7 and KT2 (the circularity rule, §4.1); J-knn2 qualifies. Agreement on KT4 and KT5 is reported beside it.
- Each judge's own claim is falsified separately.

**H8 — near-miss negatives and leave-source-out fitting fix the verifier's shifted recall** (card X11, R4).
- *The refit.* The probe is refitted with:
  - (i) KT5-type named relatives added to the OtherPlant sample;
  - (ii) the OtherPlant sample drawn leave-source-out with respect to the audited sources;
  - [review] (iii) OtherPlant eligibility by `name_status_v2`, so no generic abbreviation, role name or non-plant object enters the reject class;
  - [review] (iv) **cross-fitted by lab.** Relatives from the NDSU group are added only when the refit is scored on non-NDSU strata, and the reverse. Otherwise the named-relative conflicts below would fall because those very boxes were trained on.
- *Supported* if all three hold:
  - KT4 ~~estimation-half~~ recall rises by ≥ 0.20 absolute; [review] recall is counted against RL-confirmed labels (the G2 NDSU items with RL answers), not against the claimed labels;
  - in-domain precision stays ≥ 0.99 (KT1 CV and KT2);
  - named-relative conflicts (the G1 named strata) fall by ≥ 50 %, [review] measured only on strata whose lab was held out of the refit.
- *Falsified* if the upper bound of the recall rise is < 0.10.
- Its product is `verifier_v2/`, a candidate for a later Step 1 version. It is not used by realloop_v2.

**H9 — the target-count conclusion is fragile.**
- *Grid.* The number of target boxes the harvest yields at precision ≥ 0.95 (the estimated precision of the admitted strata × the admitted counts) is computed over 24 settings:
  - join: strict / card-mapped / card + taxonomy;
  - admission: image / box with masking;
  - thresholds: in-domain / shift-calibrated (Learn-then-Test on the KT4 and KT6 calibration halves, https://arxiv.org/abs/2110.01052); [review] each stratum's thresholds come from known truth outside its lab: KT4 only for non-NDSU strata, KT6 only for strata outside mh_weed16, and KT7 for all;
  - evidence criterion: on / off.
- *Supported* (the conclusion is fragile) if max/min ≥ 2. *Falsified* if < 2.
- The whole grid is reported.
- **[review] As posed, H9 can hardly fail.**
  - The admission axis alone moves the verified target boxes that survive from 1,060 to 2,049 (×1.93, exact and already known).
  - Any card map adds hundreds more.
  - The "× precision" definition is also ambiguous.
- **[review] H9 is therefore descriptive (a multiverse table) and carries no evidential weight.** The pre-registered test is H9′:
  - *Quantity.* For each grid cell, the one-sided 97.5 % lower bound of the expected number of true target boxes, Σ over admitted strata of N_h × precision_h. It counts only strata whose precision lower bound is ≥ 0.85.
  - *H9′ supported* (the scarcity conclusion is wrong at the pool level) if the best cell's lower bound is ≥ 2 × 2,049 = 4,098. That is more than twice what the filters verified, and about 80 % of train_core's 5,029 target boxes.
  - *Falsified* if the best cell's upper bound is < 2,049 × 1.5.
  - *Otherwise* inconclusive.
  - The thresholds axis uses shift-calibrated thresholds only after H1 and H3a are supported (§4.1 K2).

**H10 — recovered data raises dev accuracy** (realloop_v2, §9). Metric: dev mAP50-95 over the 12 species, 3 seeds, using the truth-arm rule `gate.truth_detail`.
- *H10a (primary).* The dose arm U (B ∪ every recovered image that passes the gates) against B: verdict **helps** (P ≥ 0.75 and the species guard passes).
  - [review] **The rule's false-positive rate is 0.20 under no effect** (exact permutation null of `p_greater`, 3 v 3). Simulated power at sd 0.007 is 0.82 at +0.011.
  - [review] **H10a is supported only if both hold:** the pinned rule says "helps", and a one-sided exact permutation test on the difference of means, at 5 seeds per arm, gives p ≤ 0.025. That needs 2 more B seeds and 5 U seeds. The smallest possible p at 5 v 5 is 1/252.
  - [review] With the rule alone, the result is reported as "helps (decision rule, α ≈ 0.20)", not as confirmed.
  - *Falsified* if the verdict is neutral or hurts. If it hurts (P ≤ 0.25 or the species guard fails), the recovered data lowers accuracy.
- *H10b (label effect).* U against U-ctl (the same images with join labels and no masks): P ≥ 0.75.
- *H10c (mechanism).* At least one M-size recovery increment is "helps" in the truth arm and, where a same-image control exists, beats it with P ≥ 0.75.
- *Secondary, no decision:*
  - per-species dev AP of the recovered classes;
  - gate-versus-truth agreement on the recovered steps;
  - exams read at the end only.
- *Power.* B's dev sd is 0.007 over 3 seeds, so the minimum detectable effect is about 0.011. H10a carries the large dose.
- **[review] H10d, domain dev (secondary; pre-registered so that the "neutral" branch of §9.3 can be tested).**
  - Before recovery, one capture group of each recovered source is held out of every recovered pool: a site-season for weed_crop, a capture date for greenhouse, a near-duplicate-group block for mh_weed16. The group is chosen by seed.
  - Its images are scored with each arm's existing checkpoints; no extra training is needed.
  - The labels are the source labels, with H1's measured precision attached. RL-verified labels are used where they exist.
  - H10d reads: U vs B on the domain dev, with the same rule and test as H10a.
  - Recovered data that helps on its own domain but not on cwd12 dev supports "domain mismatch" over "useless data".
  - Domain dev never makes a decision. The test-blindness guard is extended to cover it.

**H11 — the adversarial pass anticipates what the audit finds** (prospective test of the platform).
- The devil's-advocate (DA) reply is written on the cluster after D17 fires and before any estimate exists. It is committed as `funnel/prospective_da.json`.
- *Supported* if the stage with the largest lower bound on R̂ in `audit_v1.json` is among the stages the DA's counter-arguments predict.
  - [review] **As written, this almost cannot fail.** The positive fixture names about six stages, and there are about seven recoverable stages. So the DA must give a probability for each ledger stage, summing to 1.
  - [review] H11 is scored on the **top-1** stage, and by log-loss against the uniform distribution over recoverable stages. It is supported only if top-1 is right and the log-loss beats uniform.
  - [review] **The DA must be blind** to this document, to the R14 fixture and to D17's proposal list. It gets the allow-listed artifacts and the claim text only. Otherwise it can copy the hypotheses written here.
- *Falsified* if it is not.
- Every DA prediction is scored either way and enters the adversary role's track record (`outcome.py`).

**H12 [review] — discovery recall (stage S−1; reported, no recovery lever in this campaign).**
- *Known-item list.* Public datasets documented to hold at least one cwd12 species with boxes. The list is compiled on the lab from `docs/literature/`, the dataset surveys it cites and AgML's index, and it is hashed before the registry is compared.
- *Measurement:* the share of those datasets present in the registry, with a Wilson interval, plus a list of the missing ones.
- *Hypothesis:* at least 70 % are present. *Falsified* if the upper bound is < 70 %.
- *Why it is here.* The scarcity claim is about the harvest, not only about Step 1. A low recall moves the claim's weak point to discovery. That would be a harvest lever, outside this campaign.

---

## 7. Recovery levers and their precision gates

**Rules for every lever:**
- **Nothing is overwritten.** Source labels and `step1/labels` stay as they are. Recovery writes a content-addressed overlay to `INC_DIR/step1_r1/`:
  - `labels_overlay/`;
  - `images_masked/` (masked copies, each with its own sha256);
  - `recovered_pool.jsonl`;
  - `recovery.json`: per stratum, the policy, n labelled, the RL level, the precision estimate with its interval, boxes and images by class, and the input hashes.
- **Guards are never touched.** never_train, near_eval, cwd12_copy, quarantine and user flags are marked `recoverable: false`. Every recovered image passes the never-train index (6 bits) and the H6 detector.
  - [review] **The checks run on the unmasked source image,** and again on the masked copy. A mean-colour mask changes the dHash, so a masked copy of a near-eval image could otherwise pass the 6-bit index.
  - [review] **A source that H6(a) links to an evaluation set is excluded from recovery as a whole.**
  - [review] **Every recovered image records its H6(c) provenance group** (§6), which routes the exam reading in §9.3.
- **Box-level precision gate:** the RL precision of "label right and box valid", Rogan–Gladen corrected, must have a one-sided 97.5 % lower bound ≥ **0.85** and a point estimate ≥ **0.90** at the level used. That level is species, or genus for MorningGlory only.
  - *Why this level.* ~~It sits above the measured label precision of the pilot's `Breal` sources (0.737 and 0.650, Step 4).~~ [review] Struck: those numbers are a probe's agreement, not label precision (§1). The detection literature favours moderate thresholds over strict ones (OWL-ST; 2506.02359). realloop_v2 measures whether data at this precision helps.
- **Class-level gate:** RL purity has a lower bound ≥ 0.8 (Missing Link / UniDet style).
- **The Ragweed identity check comes first.** Before any NDSU Ragweed box is recovered, the RL checks that 30 train_core Ragweed crops are *A. artemisiifolia* (finely divided leaves), not giant ragweed (3–5 large lobes). The cwd12 papers and cards do not state the Ragweed species.

| Lever | Scope | Label policy | Gate |
|---|---|---|---|
| **R-A** authoritative source labels (REC-AUTH) | Non-admitted images of sources whose paper names each class's species (T1: weed_crop, greenhouse) | Target labels trusted where J1 said unknown or conflict. Named non-target boxes stay OtherPlant whatever J1 says (sibling guard). | H1-pre (exact class order) and H1 supported; box gate per (source, label) |
| **R-C** card class maps (REC-CLASS) | mh_weed16 images holding id 5 or 12 | id 12 → Sicklepod; id 5 → MorningGlory (class policy); all other ids OtherPlant | H3a supported; class and box gates. Draws group whole videos: 240 videos, with the video id taken from the upstream file names; if it cannot be derived, 3-bit dHash groups are used and that is recorded. [review] "240 videos" is [unverified]: the article describes still images (§2.5). Unless the upstream file names show a video or capture-sequence structure, the groups are 3-bit dHash groups. |
| **R-T** taxonomy synonyms (L12) | Source names that the GBIF backbone resolves as a synonym, or as a descendant at the target's rank, of a target | Overlay mapping only; the pinned alias table in `cwd12_species.py` is unchanged. Relatives are never mapped; a vernacular name alone never maps. | class gate |
| **R-V** box-level admission (REC-VETO) | the 457 images holding the 989 vetoed verified boxes | Verified and other_ok boxes are kept. Blocking boxes (unknown or conflict) are masked with a mean-colour fill, because YOLO has no ignore region. | box gate on G2v (the verified precision there was measured in-domain only) |
| **R-J** panel relabel (REC-JUDGE) | OtherPlant boxes in the no-information strata of sources that pass H6 | Relabelled only when the qualified judges agree unanimously on one target, and no source taxon is a relative of it. Boxes in the same image that any judge calls a target but that are not recovered are masked. | H7 qualification; H2a supported; box gate per (source, predicted class) |
| **R-F** re-fetch (optional) | the 1,656 MH-Weed16 images the cap left out | as R-C | as R-C; the licence is recorded (Mendeley and AgML: CC BY 4.0; the Kaggle copy says CC BY-NC-SA). [review] Re-fetched images have passed no guard, so they go through the whole Step 1 chain before anything else: the never-train index, the train_core copy check, exact_dup against the pool, H6(a) and the crop embedding. They are not "as R-C" until then. |

**Not recoverable:**
- guard stages;
- `no_boxes` images;
- small boxes;
- named relatives (under H2b);
- non-plant and leaf-disease sources.

After recovery, the evidence criterion (S12) is recomputed from the recovered counts. It has no lever of its own.

[review] **Recovered boxes at genus level.** A box whose RL answer is "*Amaranthus*, species unsure" is never recovered as PalmerAmaranth or Waterhemp. It is masked in any recovered image (§5.1).

---

## 8. The platform capability

### 8.1 Why today's autopilot cannot see this

1. **Evidence allow-list.** The allow-list (`inc_autopilot/evidence.py ALLOWED`) reads select, admit, increments and relevance files for Step 1. It cannot read `pool_summary.json` (label spaces and drops), `calibration.json` (what the known truth covers) or the verifier's fit record.
2. **D2 asks the opposite question.** It asks whether a source is off-domain, never whether the evidence was under-counted.
3. **The truth arm sees only filtered data.** Nothing reads the one step that was drawn from rejected data.
4. **The conclusion is not an object.** It exists only in RESEARCH_LOG prose. `campaign` ends with "nothing left to propose", and nothing can challenge a conclusion.
5. **The verifier has no out-of-domain recall check.** Relevance was the only filter with a fail-closed check.
6. **Derived filters chain their errors.** S11 and S12 are functions of S8's verdicts.

### 8.2 Components (new unless marked)

| Component | Runs on | Role |
|---|---|---|
| `funnel/` package: `ledger.py`, `strata.py`, `draw.py`, `estimate.py`, `qualify.py`, `leak.py`, `relation.py`, `sheets.py`, `recover.py`, `claims.py` | cluster and lab (pure Python; torch only in `leak.py` and `relation.py` embedding) | Domain-free engine. Every module reads only keys from a domain config. |
| `funnel/adapters/inc_step1.py` | cluster | Turns Step 1 into the ledger and the stratum frames. It reads `pool_meta.jsonl`, `crops.csv`, `pool_verdicts.npz`, `conflicts.csv`, `cwd12_copies.jsonl`, `cache/dhash/*.json`, `verifier/thresholds.json` and the summaries. |
| `funnel/domains/weed.json` | both | Holds: the target classes with taxon and rank; the other class; the taxonomy authority, its version and an overrides file (R4); the uninformative-name patterns (a copy of `verify.GENERIC_NAME_KEYS` and `NON_PLANT_WORDS`; `verify.py` itself is unchanged); card resolvers; known-truth tiers and their allowed uses; the judge panel; `recoverable` per stage; the class policy of §5.1. |
| `run_inc_funnel.sh` | cluster sbatch. GPU-shared for `leak` and `embed-judges`; RM-shared for the rest. | Verbs: `census`, `leak`, `embed-judges`, `qualify`, `draw`, `sheets`, `estimate`, `map`, `recover`. |
| `inc_autopilot/remote.py funnel` (new verb) | cluster login node | Prints aggregates only. `conflicts.csv` and `pool_verdicts.npz` are never shipped. |
| `inc_autopilot/evidence.py` allow-list additions | lab | `step1/pool_summary.json`, `step1/calibration.json`, `step1/verifier_fit_info.json` (a projection: OtherPlant sample sources and name counts), `funnel/funnel_ledger.json`, `funnel/audit_v1.json`, `funnel/class_maps.json`, `funnel/recovery.json`, `funnel/prospective_da.json` |
| `inc_autopilot/diagnose.py` D17, D18, D19; `thresholds.json` keys, each with its `why` | lab | §8.4 |
| `inc_autopilot/levers.json` L10–L14; cards X10–X12 | lab | §8.5 |
| `inc_autopilot/campaign.py`: context key `claims` | lab | the claims register (§8.3) |
| `model_router.ROLES["adversary"]` | cluster, async | Default `vllm:glm-4.7-flash`, a different family from the planner's `ollama:qwen3.8:27b`; fallback `ollama:gemma4`. A same-family model marks the reply `same_family`. [review] The check compares the models *actually resolved* at run time, not the configured defaults. The planner's first fallback is `vllm:glm-4.7-flash`, the adversary's default, and both fall back to `ollama:gemma4` (`model_router.py ROLES`). So a planner outage can make them the same model. |
| `inc_autopilot/brain_plan.py --role adversary`; `validate.py` DA rules | cluster job (`run_inc_plan.sh`); lab | §8.6 |
| `inc_autopilot/panel.py` | lab | Compares cold baseline experiments on dev only, by calling the pinned `gate.truth_detail` (§9). |
| `executor.REPLAY_REQUIRED` += R9–R14 | lab | Envelope autonomy needs the new replays to pass. |

**Ledger format `funnel-ledger/1`** (written through the adapter):

```
{"format":"funnel-ledger/1","domain":"<d>","inputs":{"<file>":"sha256"},
 "target_classes":[...], "other_class": id,
 "stages":[{"id","filter","version","unit":"box|image|source|class","depends_on":[...],
            "recoverable": bool, "in","kept","discarded":{"<reason>":n},
            "by_source":{src:{"in","kept","discarded":{}}},
            "by_label_pred":{"<label>|<pred>":n},
            "calibration":{"known_truth_sets":[{"id","sha256","sources","domain_score":[lo,hi]}],
                           "precision","recall","domains_covered"},
            "audit": null | {"sha256","fn_rate":{...}}}],
 "label_spaces":{src:{"classes","kinds":{"named","numeric","none","generic"},"boxes"}},
 "domain_scores":{src: median, "reference":{"q05","q50"}}}
```

### 8.3 Claims register

**Claims** are stored as `campaign/claims.json`, with fields `{id, text, polarity: scarcity|negative|positive, scope, made_by, cites[], status}`.
- A claim can be written by a person, by a COMPLETE card, or by a brain item.
- Status moves `open → challenged → tested_survives | refuted | accepted_open`, only through the DA machinery or by a person (card X12).
- The first claim is filed from the RESEARCH_LOG sentence of 2026-09-27, marked `made_by: human-transcribed`.
- [review] **C2** is filed from `figures_data.json /s1_gate_verdict_2026_08_25`: "web harvest at this scale supplies volume, not usable supervision" (polarity `scarcity`). It rests on probe agreement read as label precision (§1), so D17 treats it like C1. The platform had a second unaudited scarcity claim a month earlier, and that is a replay case (R9b).

### 8.4 Diagnoses

**Suspicion signals.** Each is cited. Today's values below are all [exact]; the fixture for S6 is pending.

| Signal | Definition | Cut point | Today |
|---|---|---|---|
| S1 reject/accept | target-class evidence rejected (target-labelled conflict + unknown + OtherPlant → target conflicts) / verified | ≥ 1.0 | (2,476 + 2,756 + 6,256) / 2,049 = **5.61** |
| S2 uninformative names | embedded boxes whose source name has status no name, numeric or generic, as a share of embedded boxes | ≥ 0.10 | 220,104 / 545,318 = **0.404**. [review] Under `name_status_v2` it is 417,717 / 545,318 = 0.766 [post hoc]. R9 pins the v1 value and the name-status version it used. |
| S3 class-yield outlier | a class with ≥ 100 name-joined boxes whose verified share is < 0.25 × the median over such classes | — | Ragweed 27 / 3,029 = **0.009**; median 0.43 over MorningGlory, Palmer, Waterhemp, SpottedSpurge, Sicklepod and Ragweed |
| S4 calibration domain gap | no known-truth set covers the domain range of the pool's sources | — | 6 of 6 sets are in-domain; 4 of the 6 species-bearing sources have retrieval medians below train_core's q05 (0.0902): 0.0388, 0.0276, 0.0125 and 0.0 |
| S5 derived dependency | a keep or discard criterion `depends_on` an unaudited stage | — | S12 → S8 (5 of 38 sources kept) |
| S6 reject-class confound | the OtherPlant sample's sources intersect the sources of rejected positives | — | [exact] the eligible pool's largest (source, name) cell is BroWeed of `rf_weed-tnf9e__weed-bqdok`, 0.620 of 316,417; whether a rejected-positive source is in the drawn sample is recorded by F3. [review] Also: the share of the reject-class sample taken by its largest (source, name) cell. About 0.62 is expected (BroWeed, §1). A reject class dominated by one source's generic name is a confound even when no source overlaps. |
| S7 truth-arm contradiction | a step drawn from filter-rejected data is "helps" while no verified step is | — | UNVERIFIED P 0.889; V1–V4 not "helps" |

S1, S2 and S4 were framed as questions before the numbers were computed. The S3 cut point and the D17 design as a whole are [post hoc]: R9 reproduces them, and it is not evidence that they generalise.

**D17 `scarcity_conclusion_unaudited`** (warn). It fires when all three hold:
- **(C) a negative conclusion exists**, meaning any of:
  - the latest finished real loop accepted nothing and no verified step is "helps";
  - the evidence sizing rule shrank M;
  - an open or challenged claim has polarity `scarcity` or `negative`.
- **(S)** at least one of S1–S7 holds.
- **(A)** no `audit_v1.json` exists whose input hashes match this Step 1.

*Proposes:* L10; L11 and L12 for the S2 sources; the DA pass; card X11 when S4 or S6 holds. It never proposes L13 (`only_after: D18`).

**D19 `filter_recall_unmeasured`** (warn; Step 1 level, needs no conclusion).
- *Fires* on S1, S4 or S6 for any recoverable stage that has no audit.
- *Proposes:* L10.
- On today's Step 1 files it would have fired before realloop_v1 (26.2 GPU-h). That is a reproduction, not evidence that it generalises.
- Ranking L10 ahead of L2 changes campaign behaviour, so it is an R4 decision.

**D18 `filter_false_negatives`** (reads `audit_v1.json`).
- *Fires* for a recoverable stage when both hold:
  - its FN lower bound is ≥ `fn_lb_min` = 0.10;
  - its recoverable lower bound is ≥ `recover_min` = 0.25 × the current verified target count. A stage that can recover less than a quarter of what was already admitted cannot overturn the conclusion.
- *Levers by kind of stratum:*
  - an uninformative label space → L11 apply (when the card and the purity check agree), otherwise L13 with R-J;
  - target-labelled but rejected → L13 with R-A, plus card X11;
  - OtherPlant-labelled but predicted as a target → L13 only where the resolved source taxon is not a relative of the prediction (L12 sibling guard). Sibling strata are recorded as known confusions, never relabelled.
- *When D18 stays silent,* the claim moves to `tested_survives`, and the COMPLETE card may say "scarcity (audited)".
- *An audit is invalid* if its calibration sets overlap an audited stratum (§4.1). D18 then reports `calibration_overlap`, and nothing may cite that audit.

### 8.5 Levers and cards

Writing each lever's code is R4 once; after that it runs at the risk shown.

| Lever | Exact action | Risk | Control → success / falsifier |
|---|---|---|---|
| L10 `funnel_audit` | `sbatch run_inc_funnel.sh {census\|leak\|embed-judges\|qualify\|draw\|sheets\|estimate} --prereg INC_DIR/funnel/prereg_v1.json --out INC_DIR/funnel/` | R2 | Known truth disjoint from every audited stratum. *Success:* H0 passes, and every recoverable stage has an FN interval with half-width ≤ 0.10. *Falsifier:* H0 fails; the audit is void. |
| L11 `resolve_label_spaces` | (a) lab: fetch and hash dataset cards, author class lists and paper tables → `class_maps.json` proposals with provenance. (b) `run_inc_funnel.sh map`: exact geometry match (H3a) and relation audit (H0b, G3). | (a) R0; (b) R2; applying a map is R3 | *Accepted* when the card and the geometry match agree and purity meets the class gate. *Refused* when the class count disagrees with the card; the class then goes to L14. |
| L12 `taxonomy_resolve` | GBIF backbone lookups, with the version and responses cached (`funnel/taxonomy_cache.json`) | R0 | Maps only synonyms or descendants at the target's rank. Relatives are listed as confusable. |
| L13 `recover_increment` | `sbatch run_inc_funnel.sh recover --audit … --maps … --policy R-A,R-C,R-T,R-V,R-J --out INC_DIR/step1_r1/` | R3 | The overlay plus recovered manifests. *Success:* H10. *Falsifier:* every recovered arm is neutral or hurts, which turns the claim into "scarcity of *useful* target data (audited)". |
| L14 `verify_queue` | Push yes/no crop tasks to the review surface | R2 | Answers go to `known_truth/<domain>/human_verify.jsonl`, split 50/50 by a seeded hash of the crop id into calibration and estimation halves. |

**Cards** (R4; a person decides):
- **X10 class-definition policy.** For example, whether *I. obscura* counts as MorningGlory. §5.1 holds until a person changes it.
- **X11 verifier recall out of domain.** Covers the H8 refit, retraining and thresholds.
- **X12 claim status sign-off.**

### 8.6 The devil's-advocate pass

**When it runs.** On any of:
- D17 fires;
- a claim of polarity `scarcity` or `negative` is written;
- before a COMPLETE card that carries such a claim.

**How.** It runs as `run_inc_plan.sh --role adversary` on the cluster, with the model from `model_router.resolve("adversary")`.

**Why it must be grounded and external:**
- self-correction without external feedback degrades (Huang 2024);
- a model's own critiques add little (CRITIC);
- judges favour their own family and the user's view (Panickssery; Sharma).

**Reply schema `inc-da-reply/1`:**

```
{"claim_id","counter_arguments":[{"argument","mechanism",
   "evidence_cites":[>=2, from >=2 distinct artifacts],"lit_cites":[...],
   "prediction":{"stage","stratum","metric":"fn_rate|purity|truth_verdict","direction","threshold"},
   "cheapest_test":{"lever":"L10|L11|L12|L14","params":{}} | {"card":"X10|X11"},
   "falsifier"}],
 "concessions":[{"claim_id","checked":[cites],"why"}]}
```

**Validation.** `validate.py` applies its existing rules plus four new ones. A counter-argument is dropped or not counted when:
- any cite does not resolve exactly, or a literature quote is not verbatim in the corpus;
- it names a non-dev split;
- its prediction does not name a stage in the ledger;
- its test is neither a menu lever whose preconditions hold nor an R4 card;
- it adds no artifact beyond the diagnosis's own cites (the echo filter).

**What surviving counter-arguments do.** They are filed as `tier2:adversary` proposals or cards, and the claim becomes `challenged`. `outcome.py` scores each prediction when its test lands (H11). A COMPLETE card may not call a negative claim "concluded" while any of its counter-arguments is still `challenged`.

**Positive fixture for R14.** This is what a correct reply says on today's evidence. Six counter-arguments:
1. Out-of-domain recall is unmeasured (S4; `calibration.json`, `select_summary.json`).
2. Paper-confirmed Ragweed was rejected (weed_crop join `"8": ["Ragweed","Ragweed"]`; the paper's Table 2).
3. Label spaces were never resolved (S2; MH-Weed16 Table 2).
4. The evidence criterion is circular (S5).
5. The truth arm contradicts the conclusion (S7).
6. The reject class may be confounded with source (S6, pending).

[review] The six items above were written by the author of this document. R14 therefore tests the validator's mechanics, not the DA's judgement. The DA's judgement is tested only by H11, with the DA blind to this list (§6).

Two concessions:
- the greenhouse OtherPlant → Palmer conflicts are mostly *A. retroflexus* per the paper;
- B vs B0 on dev (+0.005) is within noise, so data that exists may still not help.

### 8.7 Governance

| Tier | Items |
|---|---|
| R0 auto | ledger aggregation (`funnel` verb), card fetch, GBIF lookups, D17–D19 |
| R1 auto | – |
| R2 auto, within budget | L10 stages, L11(b), L14 |
| R3 approval, or the envelope rule once R9–R14 pass | applying a class map (L11), L13, the realloop_v2 builds (§9) |
| R4, a person | each new module the first time; D19 ranking L10 ahead of L2; X10; X11 and H8; X12; any change to `cwd12_species.py`, `verify.py`, `select.py`, the gate, LOCK or the never-train index |

**Invariants that stay:**
- The pinned gate decides every increment.
- Decisions read dev only. The audit reads pool and known-truth crops only, never dev, test or exam images.
  - [review] **Amended exception.** The H6 detector computes descriptors of evaluation images, and G5 shows evaluation images next to pool images. Both run only on the cluster, their outputs are used only for guard decisions, and no evaluation pixel reaches an external model or a label file (§5.2).
  - [review] H10d's domain dev is a new non-decision split. `evidence.py`, `remote.py` and `validate.py` must block it like an exam.
- The never-train guards stay.
- All mAP comes from `inc/scorer.py`; all audit numbers come from `funnel/estimate.py`.

### 8.8 Domain-agnostic design

- **Config, not code.** Everything domain-specific sits in `funnel/domains/<d>.json`.
- **A second domain** needs a config, a Step 1 adapter or a generic adapter (images, labels, class names, a filter log), and a known-truth set.
- **A grep test** (`tests/test_funnel_domain_free.py`) fails if any `funnel/*.py` outside `adapters/` and `domains/` contains a term from any domain config: cwd12, a species name, OtherPlant, BioCLIP.
- **Coupling that exists today** is to be migrated, not claimed as solved:
  - `inc_autopilot/model.py DOMAIN = "weed"`;
  - the `brain_plan` prompt header;
  - `verify.CWD12_RELATED_TOKENS` and `OTHER_ALLOWED_KEYS`;
  - the D2 summary strings.
  - [review] **The test-blindness guard names this domain's exams in code:**
    - `evidence.py NON_DEV_EXAMS` and `remote.py NON_DEV_EXAMS` = ("test", "ood22", "ood23", "imageweeds");
    - `validate.py`'s regex `\b(?:ood22|ood23|imageweeds)\b` and "cwd12 test".
    - In a second domain the guard would pass that domain's exam names silently. The exam list must come from the domain config (or `common.EVAL_SPLITS`), and R13 must assert that a vehicles-domain exam name is refused. The grep test covers only `funnel/*.py`, so it cannot catch this.
  - [review] Metric paths such as `/exams/dev/twelve` (diagnose.py) and `thresholds.json` `species_crops_max` name the domain's metric and unit.
  - [review] **Name informativeness from hand lists does not transfer**; it failed even in this domain (BroWeed). The domain config names an authority resolver instead (GBIF for organisms, WordNet or Wikidata otherwise), and informativeness means "resolves to a node of the class taxonomy".
  - [review] **A domain config must declare, per known-truth set, whether its labels are independent of every audited source** (the circularity rule of §4.1). A domain with no independent shifted truth gets D19's warning "filter recall cannot be measured", not a silent audit.

### 8.9 Replay and regression tests

**Fixtures.** They live under `tests/fixtures/inc_replay/funnel/`, with sha256 values pinned in MANIFEST.json:
- byte copies of `census_v0.json`, `step1/pool_summary.json`, `step1/calibration.json` and `realloop_v1/{exp,report,ledger,build_summary}.json`;
- the derived `funnel_ledger.json`;
- a synthetic claim fixture, marked as such;
- the cluster projection of `verifier_fit_info.json`, when it is pulled.

| Case | Asserts |
|---|---|
| **R9** today's evidence | D17 fires and cites S1 = 11,488/2,049, S2 = 220,104/545,318, S3 Ragweed 27/3,029, S4, S5 (5/38) and S7 (`/steps/2/truth/verdict` "helps"). L10 is ranked first with the exact argv. L11 and L12 are proposed for the S2 sources. **No L13.** The DA is staged; X11 is raised. R1–R8 outputs are unchanged apart from the rules version. It holds under the test-blindness perturbation. *Reproduction test.* |
| R9-early | The Step 1 files without realloop_v1: D19 fires, and L10 is proposed before L2. The case reports the 26.2 GPU-h L10 would have preceded. |
| R10 audited negative | A synthetic audit where every stratum's FN upper bound is < 0.05: D17 and D18 stay silent, and the claim becomes `tested_survives`. |
| R11 audited positive | A synthetic audit where weed_crop's rejected Ragweed stratum has FN lower bound 0.6, and the greenhouse OtherPlant → Palmer stratum has 0.4 but a card taxon of *A. retroflexus*. D18 fires. L13 covers only the Ragweed stratum; the sibling stratum is guarded. A card class map whose class count disagrees with the card goes to L14. |
| R12 disjointness | A judge calibrated on a source or near-dup group shared with a stratum makes the audit invalid, and `validate` refuses any item that cites it. |
| R13 other domain | A synthetic "vehicles" config (car, truck, bus + OtherObject; a hypernym table; a numeric-named source; no out-of-domain calibration). The same code and thresholds fire D17 and D19. Includes the grep test. |
| R14 DA | The positive fixture keeps 6 counter-arguments and 2 concessions. A sycophantic reply (concessions without checks) keeps 0, and the claim stays `challenged`. A fabricated cite is dropped, a test leak is dropped, and an echo is not counted. A same-family model cannot move a claim. |
| Negative controls | pilot_v1–v3, b0_v1 and base_b_v1: D17 is silent (no conclusion). A synthetic high-yield Step 1 with out-of-domain calibration: D19 is silent. |
| Mutation tests | The suite must kill each of: the conclusion condition removed; the probe used as its own judge; thresholds fitted on the audit sample; the sibling guard removed; a guard stage marked recoverable. [review] Also: the RL qualified on the claimed labels of the population being tested (KT4 for H1); a judge's kNN bank sharing a lab with the judged stratum; the never-train check run on the masked copy only; a source with a detected H6 copy recovered image by image; an exam name missing from the test-blindness list. |
| R9b [review] | The 2026-08-25 claim C2 with only the artifacts that existed then: D17 or D19 fires on C2 and asks for an out-of-domain known-truth set before the claim can be cited. A reproduction, not a generalisation. |
| Prospective | `prereg_v1.json`, `prospective_da.json` and the live D17/D19 outputs are committed before L10's estimate runs (H11). The first non-weed campaign is the second prospective test. |

---

## 9. The experiment the platform runs with recovered data: realloop_v2

**Question.** Does recovered data raise dev accuracy when added to B (H10)?

**Who runs it.** The platform, through L2 and the baseline builder, after L13. A person does not build it by hand.

**Base.**
- B = `step1/base_B.jsonl`, the same bytes as realloop_v1 (sha256 in its `exp.json`).
- Recovered data enters only as increments and in the union arms; B is not rebuilt.
- The base arm is B's three cold seeds.

### 9.1 Part 1: the recovered loop (chain and truth arm; realloop builder)

- **Builder.** `inc.realloop build --exp realloop_v2 --base <base_B> --step1-overlay INC_DIR/step1_r1 --increment-sources recovered --size M --replay-mode full --gate-flips-mode net`.
  - `--step1-overlay` and `--increment-sources recovered` are new. They make a new realloop protocol version (R4 code, once).
  - The pinned modules (`driver.py`, `gate.py`, `splits.py`, `common.py`, `cwd12_species.py`) are unchanged.
- **M.** 287, as in realloop_v1. If an arm's capacity is smaller, M becomes the smallest capacity, down to a floor of 5 % of B (196 images). An arm below the floor is dropped and reported.
- **Disjoint pools:**
  - VETO = the 457 vetoed images;
  - AUTH = non-admitted NDSU images minus VETO;
  - CLASS = mh_weed16 images holding id 5 or 12;
  - JUDGE = R-J images.
- **Sequence** (every step `clean: false`):
  1. REC-VETO
  2. REC-AUTH-1
  3. REC-CLASS-1
  4. REC-JUDGE-1
  5. REC-AUTH-2
  6. REC-CLASS-2

  **Substitutions.** An arm that cannot be drawn is replaced by the next draw of the arm with the most remaining capacity, in the order AUTH, CLASS, VETO. The replacement is recorded.
- **Truth arm.** T grows only by clean steps, so every step compares B ∪ D_k with B, a fixed-base design (`driver._truth_transitions`). The verdict is `gate.truth_detail`: P(with > without) over the 3 × 3 seed pairs; "helps" at P ≥ 0.75 with the species guard passing; "hurts" at P ≤ 0.25 or when the species guard fails.
- **Chain arm.** It uses realloop_v1's recipe, or the X1 recipe if X1 has landed. Its verdicts are secondary. On realloop_v1 the cheap recipe degraded the incumbent at every step: P_recipe was 0.0, and null (0.8164) was below inc (0.8189) at all six steps. So the chain cannot see data effects until X1 is resolved. D1 itself did not fire there, because no clean step was "helps" in the truth arm.

### 9.2 Part 2: the dose and same-image controls (cold baselines; existing builder)

Each arm is `inc.pilot build-baseline --exp rv2_<arm> --manifest <manifest>`:
- 3 seeds;
- the same recipe, init and seeds as B;
- the never-train guard applies.

`inc_autopilot/panel.py` compares each arm on dev only through `gate.truth_detail`. If `build-baseline` refuses a manifest that is not a select build, a recovery provenance record (the `recovery.json` and overlay hashes) is added to its refusal check (R4).

| Arm | Training set | Compared with | Isolates |
|---|---|---|---|
| U | B ∪ every recovered image that passes the gates (a seeded stratified draw capped at \|B\| = 3,927 images if larger) | B (base arm) | H10a: total value at full dose. [review] 5 seeds for U, and 2 extra B seeds so that B has 5 (§6 H10a). |
| U-ctl | the same images with join labels and no masks | U | H10b: the effect of the recovery labels and masks |
| CLASS-ctl | B ∪ REC-CLASS-1's images with join labels (all OtherPlant) | the truth "with" runs of REC-CLASS-1 | H10c: the class-map label change on identical images |
| JUDGE-ctl | B ∪ REC-JUDGE-1's images with join labels | the truth "with" runs of REC-JUDGE-1 | H10c: the panel relabel on identical images |

**Existing references, not re-run:**
- realloop_v1's T_final (dev 0.8093 ± 0.0027): data the old funnel admitted.
- realloop_v1's UNVERIFIED step (helps, 0.889): rf_tuf with join labels. Void if H6a finds a leak.

### 9.3 Reading and interpretation

- **Decisions read dev 12-class mAP50-95 only.** ood22, ood23, ImageWeeds and test are read once at the end, reported as description, and never used in a decision. Arms containing rf_tuf have void ood numbers if H6a holds.
  - [review] ImageWeeds is reported as "same-lab" for every arm with NDSU-group images, which includes B (H6c).
  - [review] H10d's domain dev is read with the exams, at the end, and is never used in a decision.
- **Interpretation, fixed in advance:**

| Audit (§6) | H10a | Conclusion recorded in the claims register |
|---|---|---|
| Large recoverable count (H1 or H2a or H3 supported; H9 fragile) | helps | The filters discarded useful target data; the scarcity claim is refuted. |
| Large recoverable count | neutral | Target data exists but does not raise dev at this dose and recipe: "scarcity of useful data under dev (audited)". Domain mismatch (ND greenhouse and fields, Indian soybean fields vs NC/MS cotton) is the first explanation to test. [review] H10d tests it: "helps" on the domain dev with "neutral" on cwd12 dev supports mismatch; "neutral" on both supports "not useful at this dose". |
| Small recoverable count (H1 falsified, H2a falsified, H3b falsified, H4 supported) | any | The filters were right; "scarcity (audited)". |
| any | hurts | Recovered data at this precision lowers dev; reported per class, with H10b saying whether labels or images carry the harm. |

- **Cost estimate.** Part 1 has the shape of realloop_v1 (26.2 GPU-h). Part 2 is 12 cold runs, about 13 GPU-h, since U and U-ctl are up to twice B's size. The total is about 40 SU, charged to the campaign envelope through `su_ledger`.
  - [review] Plus the H10a seeds: 2 B runs and 2 U runs beyond the 12, about 4–7 GPU-h. H10d needs inference only, under 1 GPU-h.
  - [review] realloop_v1 spent its 26.2 GPU-h in 2 h 52 min of wall clock (ledger 19:46 → report 22:38 UTC, 2026-09-27), so queue time, not compute, sets the calendar.
  - [review] The whole campaign's cost is in §10.1.

**Acceptance of realloop_v2:**
- at least 6 recovered increments decided;
- the truth arm decided at every step;
- all panel arms scored with 3 production seeds; [review] U and B with 5;
- ledger, `report.json` and `panel.json` written;
- H10 evaluated by `outcome.py`;
- the test-blindness replay extended to `panel.py` and passing.

---

## 10. Execution order and acceptance criteria per step

| Step | What | Who / tier | Artifacts | Acceptance |
|---|---|---|---|---|
| F0 | This contract and its JSON transcription | a person commits | `docs/FUNNEL_AUDIT.md`, `INC_DIR/funnel/prereg_v1.json` | Committed before F6 and before any RL label. Later changes are dated amendments marked post hoc. |
| F1 | Platform code (§8.2), fixtures and replays R9–R14 | R4, once | `funnel/`, adapter, `domains/weed.json`, autopilot changes, tests | The full existing INC and autopilot suite passes, plus the new tests. R9 reproduces S1–S5 and S7 exactly. Negative controls are silent. R13 and the grep test pass. A replay pass is recorded for the executor. |
| F2 | Live detection and the adversarial pass | platform ticker (R0); DA on the cluster | campaign ledger entries; `funnel/prospective_da.json` | D17 and D19 fire on the real Step 1 with resolvable cites. The DA reply is validated and committed before F8. Invalid replies are recorded as such, not retried until they pass. |
| F3 | L10 `census` | platform, R2 | `funnel/census_v1.json`, `funnel/funnel_ledger.json`, `funnel/ledger.jsonl` (one row per unit with its discard path) | Sums reconcile exactly with the Step 1 summaries: 545,318 crops; 2,049 / 8,732 / 2,756 / 531,781 verdicts; 1,060 admitted target boxes; 125,500 small; the 989 veto losses fully attributed. H5a and the verifier failure-mode split are computed. The ledger validates. [review] Also: `name_status_v2` (S7b) over every census name through J-taxon, frozen before F6; the quarantine reason of each S1 slug; the OtherPlant-sample composition (S8b, from `fit_info.json`). |
| F4 | L10 `leak`; L11a card fetch; the H3a geometry match | platform, R2 (the cards on the lab, R0) | `funnel/leak_v1.json`, `funnel/cards/` (hashed), `funnel/class_maps.json` (proposals) | The detector meets its calibration (recall ≥ 0.95, FPR ≤ 1 %) before judging any candidate. H6a, H6b and H3a (exact part) are recorded. [review] Calibration uses the synthetic-augmentation positives and provenance-disjoint negatives of §6 H6. H6(a) covers all 14 sources of §3.4. H6c groups, H1-pre (NDSU class order), the KT7 fetch (hashed) and the H12 known-item list are recorded. |
| F5 | L10 `embed-judges` and `qualify` | platform, R2 (GPU) | `funnel/judge_qualification.json` (locked) | The disjointness check passes. H7 is evaluated. The VLM judge is used only if its vision capability is confirmed. [review] Includes the lab rule and the circularity rule of §4.1. |
| F6 | L10 `draw` | platform, R2 | `funnel/sample_v1.csv` (sha in the prereg amendment "sample lock") | Stratum sizes match the ledger. A test recomputes the seeds. [review] Strata use the frozen `name_status_v2`. Sheets interleave strata (§4.3). |
| F7 | L10 `sheets` and reference labels | RL backend(s); L14 for a person | `funnel/sheets_v1/`, `funnel/gold_v1.csv` (with `labeller`) | G0 and the sentinels are complete. The RL level is fixed from the sentinels before any estimate is computed. Each group meets its minimum n, or the shortfall is reported. [review] The RL backend is an R4 decision recorded before F7. Qualification counts only KT1–KT3 and KT7 sentinels. |
| F8 | L10 `estimate` | platform, R2 | `funnel/audit_v1.json` and `.md` | **H0 passes, or the campaign stops.** H1–H5, H7 and H9 are evaluated. Every number carries its estimator, n and interval. D18 reads the file. H11 is scored. |
| F9 | L11 apply, L12, L13 | platform, R3 (envelope) | `funnel/taxonomy_cache.json`, `INC_DIR/step1_r1/` | The Ragweed identity check is done. No guard stage is touched. Every recovered image passes the never-train index and H6. Every recovered stratum meets its gate. Source labels are unchanged (hash check). |
| F10 | realloop_v2 parts 1 and 2 | platform, R3 (envelope) | `INC_DIR/realloop_v2/`, `INC_DIR/rv2_*/`, `panel.json` | As in §9. |
| F11 | H8 verifier refit | R4 (X11), in parallel after F8 | `INC_DIR/funnel/verifier_v2/` | H8 is evaluated. No Step 1 is rebuilt in this campaign. |
| F12 | Claim resolution and records | DA pass 2 (cluster); a person signs X12 | `campaign/claims.json`; RESEARCH_LOG, CHANGELOG, README; cross-references in INCREMENTAL_PROTOCOL.md and INC_AUTOPILOT.md | Every hypothesis is reported with its outcome. The claim status follows the §9.3 table. |

**Stop rules:**
- H0 fails → stop at F8.
- H6b finds a copy → an R4 incident before F9.
- Any never-train hit in a recovered set → stop F9.
- Campaign stop-losses and budget limits as in docs/INC_AUTOPILOT.md (d).
- [review] If H6(b) finds a copy in B, realloop_v1's conclusion (claim C1) is void rather than challenged. B is rebuilt, and B's baseline is re-run, before F10. realloop_v2 then compares against the rebuilt B.
- [review] If no RL qualifies at species level on KT7 for a genus, every hypothesis about that genus is reported as bounded, not as supported or falsified. The campaign continues for the other genera.

### 10.1 [review] Cost and time

Estimates, except where marked [exact]. The GPU-h figures assume Bridges-2 V100 GPU-shared. [exact] Step 1's BioCLIP-2 pass embedded shard 2 (125,844 crops of 28,345 images) in 643 s, so about 2,800 s for all 545,318 pool crops on one GPU.

| Step | Resource | Estimate | Basis |
|---|---|---|---|
| F0 | a person | 1 review session | this document plus `prereg_v1.json` |
| F1 | engineering, plus about 8 first-time R4 reviews (§8.7) | the largest item; not estimated here [gap] | 10 new modules, an adapter, autopilot changes, R9–R14 and the mutation tests |
| F2 | cluster LLM (glm-4.7-flash) | under 1 GPU-h per DA pass, 2 passes | one reply of a few thousand tokens |
| F3 | CPU (RM-shared) | under 1 h | reads existing files. Step 1 admit took 55 s, calibrate 244 s and select 401 s [exact]. The GBIF lookups for about 100 names take minutes. |
| F4 | GPU (descriptors) plus lab fetches | 2–5 GPU-h; wall clock dominated by Lustre reads | about 103 K pool images + 9.5 K evaluation images + about 15 K synthetic positives. The Step 1 pool build, hashing about 200 K listed images, took 5,472 s [exact]. |
| F5 | GPU | 2–6 GPU-h | DINOv2 over 545,318 crops + KT sets; J-vlm (minicpm-v) on about 2,600 items at 2–5 s each is 1.5–4 GPU-h |
| F6 | CPU | minutes | |
| F7 | reference labels | a person: about 2,600 items at 5–15 s = 4–11 h. Papadopoulos' 1.6 s is for easy yes/no on large objects, not fine-grained seedlings. A machine RL: GPU or API time proportional to about 2,600 multi-image prompts | §5.2 totals with G2a and KT7 |
| F8 | CPU | under 1 h | the estimator with 2,000 bootstrap resamples |
| F9 | CPU and I/O | 1–2 h | masked copies and overlays for a few thousand images |
| F10 | GPU | about 40 + 4–7 GPU-h (§9) | realloop_v1: 26.2 GPU-h in 2.9 h wall clock [exact] |
| F11 | CPU or GPU | under 1 h | a probe refit on existing embeddings. Step 1 fit plus calibrate took minutes. |
| **Total** | | **about 50–60 GPU-h**, 4–11 person-hours if a person labels, and about 8 R4 decisions | |

**Calendar.** The GPU steps take hours each. The critical path is F1 (engineering and R4 reviews) and F7 (the RL decision and its labels), not compute.

---

## 11. Open risks

- **Reference-labeller validity.** Seedlings are small, and *Amaranthus* species are hard to tell apart even for experts. The RL may be a strong multimodal model rather than a weed scientist. Mitigations: sentinel-measured Se and Sp, demotion by level, Rogan–Gladen correction, and optional adjudication by a person. Even so, a species-level claim about *Amaranthus* may end up genus-level.
- **Source labels as truth.** KT4 comes from one lab; H1 measures its labels rather than assuming them. The MH-Weed16 licence differs between copies: CC BY 4.0 on Mendeley and AgML, CC BY-NC-SA on Kaggle.
- **cwd12's Ragweed species is not stated** by the CottonWeedID15 paper, the CottonWeedDet12 record or the DCW README. `CWD12_BINOMIAL` assumes *A. artemisiifolia*. The check before F9 decides whether NDSU Ragweed may be recovered.
- **Domain mismatch.** Recovered data comes from ND greenhouse pots, ND fields and Indian soybean fields. Dev is cwd12 capture sessions. Real value may not show on dev, and the exams stay descriptive.
- **Effective sample size.** MH-Weed16 is about 28 frames per video over 240 videos, so its effective n is far below its image count. Grouping by video is required; if video ids cannot be derived, dHash groups under-merge. [review] The video structure is [unverified]: the article and both cards describe still images (§2.5). Effective n is then set by near-duplicate and field-date groups.
- **Power.** The minimum detectable effect is about 0.011 dev mAP50-95 at 3 seeds, so neutral outcomes are likely for the M-size arms. The dose arm exists for this reason, and a neutral verdict is reported as "not detectable", not as "no effect".
- **Post hoc exposure.** census_v0, the image-rule query and the dataset forensics were seen before this contract. H1, H2 and H3a are partly post hoc, and D17's cut points reproduce today's evidence. The prospective tests are H11, the confirmatory reference-labelled estimates, and the first non-weed campaign.
- **Confirmation pressure.** The campaign starts from the suspicion that the filters were wrong. Mitigations: symmetric falsifiers, concessions required from the DA, and prominent reporting of "the filter was right" outcomes (H2b, H4, H5).
- **Leak.** rf_tuf may be a re-export of 3SeasonWeedDet10 2022/2023 exam material; its predicted classes match that dataset's class names. H6 decides this before any recovery. realloop_v1's "helps" on UNVERIFIED stays uninterpreted until then.
- **Chain arm.** It is uninformative while the cheap recipe degrades the incumbent and X1 is open (§9.1).
- **Unresolved upstreams.** rf_tuf, crop-weed-poxtn, school and test-8qezo have unknown upstreams: Roboflow Universe returns 403, and their class names really are numbers. L11a through the platform's Roboflow API route is untested. H3b then rests on visual clusters and RL labels only.
- **Harvest cap.** The location of the 5,000-image cap is not identified in the local code. R-F depends on finding it.
- **Correlated judges.** J1 and J-zs share the BioCLIP-2 encoder, and J-knn1 shares J1's reference domain. Only J0, J-card, J-taxon, J-knn2 (partly) and J-vlm are independent.
- **Unverified tools.** It is not verified that SSCD weights can be installed on the cluster (DINOv2 is the fallback), or that the planner model has vision capability.
- **Governance load.** Many items are R4 the first time (§8.7). Each is recorded as a decision with its reason before it runs. None is folded into the envelope rule.
- **[review] Provenance leakage into exams.** The NDSU recovery sources and the ImageWeeds exam share lab, sites and seasons. B already holds NDSU images. A pixel detector cannot clear this (H6c), so ImageWeeds is "same-lab" for these arms.
- **[review] cwd12-derived admitted set.** 45.9 % of the verified target boxes, and 85.6 % of B's harvested target boxes, come from two re-exports with dev and test hits. H6(b) may void realloop_v1's conclusion before this campaign reaches it.
- **[review] Independent truth is thin.** KT7 (iNaturalist Research Grade) is community-verified, mostly of older plants, and possibly in BioCLIP-2's training data. It qualifies the RL and the DINOv2 judges only. Where it is too small for a genus, species-level claims about that genus stay bounded.
- **[review] Name-status v2 changes strata already described in this document.** The v1 counts are kept for traceability. The confirmatory frames are the v2 frames, whose sizes are locked at F6.
- **Zero-shot relevance stays retired.** It is not re-tuned on this pool (X9). A few-shot plant/non-plant probe is a candidate for a later relevance version, pre-registered on train_core first.

---

## 12. [review] Items the review could not settle in this document

Each item needs data from the cluster, a person's decision, or code. None is closed by editing this text.

1. **`prereg_v1.json`** does not exist yet. It must be generated from this version, including the [review] thresholds: H4's inconclusive band, H9′, H10a at 5 seeds, H11's top-1 and log-loss scoring, and H12.
2. **The reference labeller's backend** (§4.3) is an R4 decision with no candidate in `model_router`. Every estimate depends on it.
3. **KT7** (iNaturalist Research Grade) is specified but not fetched. Its size per taxon at seedling stage is unknown. If it is too small, species-level qualification fails for that genus.
4. ~~**Pending cluster reads**~~ Read on 2026-09-28 and entered above: the OtherPlant eligible pool (§1, S8b), τ[OtherPlant] (§4.2), the quarantine reasons (S1) and the embedding time (§10.1). The drawn 5,000-crop composition is still read by F3.
5. **The harvest-cap location** (S0) is still unidentified. `rf_weed-tnf9e__weed-bqdok`'s exact 10,000 suggests a second cap.
6. **MH-Weed16's dropped class.** Which of the 16 classes AgML dropped is unknown until L11a fetches the upstream annotations. So is the "240 videos" structure. [post hoc] The upstream annotation archive (`intel Real Sense Depth_Annotations.zip`, Mendeley d3n3mgjjbv v2) names its files `<stem>_<n>.xml`, for example `11087x7s4zk8382409_360`, with 240 distinct stems over 6,656 files. That is consistent with 240 capture sequences, but no source says so; L11a records the stems, and the grouping rule stays as written in §7 R-C (stems are used as groups only if the stem structure is confirmed, else 3-bit dHash groups).
7. **ImageWeeds and the NDSU sources.** Whether the ImageWeeds greenhouse images and the NDSU Waldron Greenhouse dataset show the same plants is [unverified]. Only the shared lab, sites, seasons and camera model are verified.
8. **The R14 positive fixture** was written by the author of this document, so it tests the validator's mechanics only (§8.6).
9. **H12's known-item list** has not been compiled. Its inclusion rule, "documented to hold at least one cwd12 species with boxes", needs a hashed source list before the registry is compared.

---

## 13. Decisions recorded before F1

These are the R4 decisions §8.7 and §12 leave open. They were made on 2026-09-28 under the project owner's standing delegation for this campaign, before any platform code, sample draw or reference label. Each is logged in the campaign ledger with `decided_by: human-delegated` when F1 lands. X12 (the final claim status) is not delegated: it stays with the project owner.

| Id | Decision | Reason |
|---|---|---|
| DEC-1 | **Two reference-labeller backends answer the same blind sheets.** RL-A is Claude Opus 5.5 (vision), a strong multimodal model outside every judged family (BioCLIP-2, DINOv2, qwen, glm). RL-B is `ollama:qwen3.8:27b` (vision) in a cluster job, the platform-native candidate. Both are qualified on the same KT1–KT3 and KT7 sentinels (§4.3). For each hypothesis the primary RL is the qualified backend with the larger sum of the Se and Sp lower bounds at the level that hypothesis needs; a tie goes to RL-B. The other backend's labels and its agreement with the primary are reported. | §12 item 2. A platform that must run without a person needs a platform-native RL, and whether it qualifies is itself a result. RL-A bounds what a strong model can do on these crops. Two backends from different families also expose correlated errors. RL-A is the same family as this document's author, so it only ever sees blind multiple choice with no source, verdict or stratum. |
| DEC-2 | **RL-A sees pool and known-truth crops only.** No dev, test, exam or H10d domain-dev pixel is sent to it. G5 pairs (evaluation images) are judged by RL-B only, on the cluster. If RL-B does not qualify, H5b is reported as "not evaluated". | §8.7 invariant. |
| DEC-3 | **Class policy** (card X10) is §5.1 as written: MorningGlory = *Ipomoea* spp.; every other class is one species. | Matches `CWD12_BINOMIAL`. It is fixed before any reference label. |
| DEC-4 | **D19 ranks L10 ahead of L2** when it fires. | A negative loop result read only through unaudited filters cost 26.2 GPU-h in realloop_v1 (§8.4). |
| DEC-5 | **The new modules of §8.2 are accepted once, subject to the F1 acceptance tests.** They run afterwards at the risk tier §8.7 gives them. The pinned INC modules (`driver.py`, `gate.py`, `splits.py`, `common.py`, `cwd12_species.py`, `verify.py`, `select.py`) are not edited. | The contract separates new code, which is reviewed once, from pinned decision code, which stays unchanged. |
| DEC-6 | **The H8 verifier refit (card X11) runs in F11.** Its product is a candidate only and is not used by realloop_v2. | It is cheap (existing embeddings), and it tests the design causes found in §4.2. |
| DEC-7 | **Licences.** Recovered images are used for research training only and are never redistributed. Each recovered image records its source licence in `recovery.json`. The weed_crop, greenhouse and MH-Weed16 data are CC BY 4.0 on Mendeley; the MH-Weed16 Kaggle copy says CC BY-NC-SA and is not the copy used. Use in a commercial product needs a separate licence review. | §12, critic item 5. |
| DEC-8 | **KT7** is at most 30 iNaturalist Research Grade observations per taxon, CC-licensed photos only. Each observation's id, licence and photo sha256 are recorded. The photos are used for sentinels and judge qualification only, never for training. | §4.1. |
| DEC-9 | **The judge panel for F5** is J-zs (BioCLIP-2 zero-shot with taxonomic plus common-name prompts over the closed candidate set), J-knn1 and J-knn2 (DINOv2 ViT-B/14 features of the same crops), and J-vlm = RL-B asked the same multiple choice. The copy detector for H6 is DINOv2 descriptors plus dHash of flipped and rotated variants, calibrated as in §6 H6. SSCD is not used. | SSCD weights are not installed on the cluster, and DINOv2 is §6's named fallback. qwen3.8:27b is the strongest vision model in the cluster store (§4.3's minicpm-v, llama3.2-vision and moondream are smaller). |
| DEC-10 | **Budget.** The campaign's GPU spend is charged to the INC envelope through `su_ledger`, with a campaign cap of 120 GPU-h (about twice §10.1's estimate). Reaching the cap stops new submissions and is reported. | §10.1. |
