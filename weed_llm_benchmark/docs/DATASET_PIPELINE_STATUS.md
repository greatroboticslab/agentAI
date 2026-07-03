# Dataset upload + analysis pipeline — status (2026-07-03)

Honest status of the **student dataset upload → analysis → ready-to-train** flow, built for the
professor's "get it done this weekend so we can run experiments" goal. Every ✅ item is verified by
automated tests (`tests/smoke_test.sh`, `tests/e2e_dataset.sh`) against the live lab server.

## ✅ Works end-to-end (verified)

**Upload (any format, no zipping required)**
- `.zip`, `.tar`, `.tar.gz`, `.tgz` archives, a **folder**, **multiple loose files**, or a **single image** —
  detected by magic bytes / multipart. Drag & drop, "Choose files", or "Choose folder" in the UI.
- A single redundant wrapper directory is stripped, so `images/train/<class>/` layouts land correctly
  (this also fixed the earlier "no training images found" bug).
- Understands: classification (class subfolders), YOLO (`labels/` + `data.yaml`), COCO/VOC (class names
  auto-extracted), or just images (no labels).
- Optional **goal/purpose** captured per dataset (typed or spoken).

**Analysis (read-only, lab-local, no GPU needed for EDA)**
- EDA: image count, modality mix, train/val split, class distribution, image dimension/aspect/filesize
  histograms, near-duplicate detection, per-class sample grid. Non-image modalities too (video via cv2,
  audio via wave, sensor CSV numeric stats, point-cloud counts, text length).
- **🤖 AI review**: grounded rule-based issue detection (no labels, class imbalance, too-few-per-class,
  >10% duplicates, tiny/huge images, no validation split, partial labels) MERGED with a local-LLM
  (qwen2.5:3b) plain-English summary + recommendations, plus a **training-readiness verdict** and a
  **fitness-for-goal** check. Degrades to a rules-only report if no LLM is reachable (clearly labelled).

**Voice** — self-hosted Whisper (`/api/voice/transcribe`, faster-whisper on the lab GPU) for the goal
and intent boxes; browser Web Speech as the instant fallback.

**Governance + training** — a student can upload/browse/analyze freely; training on the GPU is gated
(member → 403) until an admin grants cluster access, then jobs queue; verified a real classification
job runs to completion and writes metrics.

**Onboarding** — `/guide` page (formats, layouts, how analysis + training work), linked from the
launcher and empty states.

## ⚠️ Honest limits / scaffold
- The **collector / filter / labeler** agents run the shared harvest→label pipeline, which is still
  specialised for the weed/CWD12 domain — not yet per-arbitrary-field. The **trainer** is general (any
  uploaded dataset). Self-sup / RL / multi-strategy training are scaffolded (501).
- `/api/train/submit` is **synchronous** (stages data to the cluster inside the request, ~40–150s) — it
  works but the browser button waits; making it async is a known TODO.
- The AI review uses a small 3B local model — good for grounded summaries; a larger on-demand cluster
  model (deepseek-v3/glm) second pass is a planned bonus, not yet wired.

## How to demo (for the professor)
1. Open the dashboard, click **👋 New here? Guide**.
2. Create a project → drag a folder of class-subfolder images (or a zip) into the upload box, add a goal.
3. Click **📊 Analyze** on the dataset → read the EDA → click **Analyze with AI** for the review +
   readiness + fitness.
4. (Admin) grant a student cluster access → the student clicks Train → a real job runs and writes metrics.

## Test coverage
- `tests/smoke_test.sh` — regression suite (pages/APIs/auth/RBAC/project+agent lifecycle/upload formats/
  analysis/AI review/goal/voice). Green.
- `tests/e2e_dataset.sh` — full student journey + every upload path + edge cases (unlabeled, imbalanced,
  non-image, bad/empty upload) + permission gate + real training submit.
