# Project Handoff

## Current Milestone

Milestone 11 — Portfolio Finalization. Completed on 2026-08-24. The final
production model is the locked V2-A balanced ResNet18; no new modeling work was
performed.

## Milestone 11 Finalization

- Preserved the complete research progression and every earlier metric in its
  original dataset/split context; no completed result JSON was overwritten or
  reinterpreted.
- Updated the existing Streamlit inference integration to accept only the
  selected `v2a_balanced_resnet18` checkpoint format and changed the default to
  `models/v2a_balanced_resnet18.pt`.
- Kept preprocessing fixed at symmetric edge padding, 224×224 bilinear resize,
  and ImageNet normalization; made the locked occupied threshold 0.5 explicit.
- Added candidate, architecture, experiment-config-hash, and pre-selection
  boundary validation before the demo loads a checkpoint.
- Re-verified the production checkpoint SHA-256
  `97b039fa7d4125e993903c4d1b485a7bc8e58d47cf7917c5fef8515e6982d5f9`,
  selection lock, and completed fresh-final result without reopening evaluation.
- Passed a real V2-A CPU inference smoke test and browser-verified the existing
  Streamlit initial/upload/result states. The sample returned `OCCUPIED`; this
  was a UI smoke test, not a new evaluation result.
- Preserved the original V1 demo screenshots/GIF and Milestone 10 demo report;
  added separate V2-A screenshots and `images/v2_parking_occupancy_demo.gif`.
- Rebuilt `README.md` as a bilingual portfolio README with final highlights,
  model comparison, research progression, experiment boundaries, error summary,
  demo instructions, limitations, and application-material links.
- Added `docs/MODEL_COMPARISON.md`, retaining separate fair-comparison sections
  for CNR validation, V1 adaptation, V2 validation selection, and fresh final.
- Added the bilingual `docs/FINAL_RESEARCH_SUMMARY.md`, 1–2 page
  `docs/GRADUATE_APPLICATION_ABSTRACT.md`, and concise
  `docs/CV_PROJECT_DESCRIPTION.md` with a technical-skills matrix.
- Added `docs/FINAL_DEMO.md` for the production V2-A inference contract and
  browser validation record.
- Generated `images/research_workflow.svg`, `images/domain_shift.svg`, and
  `images/v1_v2_improvement.svg` directly from locked result JSON values.
- Browser-rendered and visually checked all three SVGs; corrected comparison
  labels before finalizing them.
- Added `images/repository_qr.png`, pointing to the configured GitHub origin,
  and `src/generate_portfolio_assets.py` for reproducible portfolio assets.
- Did not retrain either model, evaluate EfficientNet on fresh final, reopen
  fresh final, change splits, calibrate confidence, change threshold, or revise
  a model from final results.
- The historical V1 registry now reports expected mismatches only for
  `app/app.py` and `src/inference.py`, because Milestone 11 explicitly
  authorized the production-demo checkpoint update. All other 26 registered V1
  checkpoints, results, local manifests, documentation files, and original demo
  media still match their frozen hashes.
- Final verification: 23 automated tests passed, Python compilation passed, all
  repository-local Markdown links resolved, the QR decoded to the configured
  GitHub origin, `git diff --check` passed, and no private SSD path was found in
  portfolio artifacts.

### Milestone 11 files

New:

- `docs/MODEL_COMPARISON.md`
- `docs/FINAL_RESEARCH_SUMMARY.md`
- `docs/GRADUATE_APPLICATION_ABSTRACT.md`
- `docs/CV_PROJECT_DESCRIPTION.md`
- `docs/FINAL_DEMO.md`
- `images/research_workflow.svg`
- `images/domain_shift.svg`
- `images/v1_v2_improvement.svg`
- `images/repository_qr.png`
- `images/v2_demo_initial.png`
- `images/v2_demo_inference_result.png`
- `images/v2_parking_occupancy_demo.gif`
- `src/generate_portfolio_assets.py`

Modified:

- `README.md`
- `app/app.py`
- `src/inference.py`
- `tests/test_baseline.py`
- `HANDOFF.md`

Dataset source, size, labels, and split assignments did not change during
Milestone 11. Large images remain only on external storage. Final labels remain
`0=EMPTY`, `1=OCCUPIED`; fresh final remains exactly 154,669 samples and closed.

## Completed

- Created the planned repository directory structure.
- Added `.gitkeep` files so empty directories can be tracked by Git.
- Added the README project skeleton.
- Added the initial Python dependencies to `requirements.txt`.
- Updated `.gitignore` for local environments, datasets, caches, secrets, and
  model checkpoints.
- Preserved `PROJECT_PLAN.md` without modification.
- Evaluated PKLot and CNRPark+EXT using their official dataset pages, READMEs,
  papers, archive descriptions, and license declarations.
- Selected CNRPark+EXT as the primary dataset for the initial classification and
  held-out-camera work.
- Reserved PKLot as a later external domain for cross-location evaluation.
- Documented the comparison, limitations, licensing, acquisition strategy, and
  leakage risks in `docs/DATASET_RESEARCH.md`.
- Downloaded only the official CNRPark+EXT metadata CSV (18,132,695 bytes) and
  recorded its source and SHA-256 in `data/DATA_MANIFEST.md`.
- Confirmed 157,549 unique metadata rows with no duplicate image paths.
- Added `src/prepare_splits.py` to generate split manifests and fail on defined
  date, frame, camera, or slot leakage.
- Defined a primary date-grouped CNR-EXT protocol and a separate camera-and-slot
  holdout protocol.
- Generated and audited the local split manifest and summary; all automated
  leakage checks passed.
- Documented the split counts, rules, limitations, and reproduction command in
  `docs/DATA_SPLIT.md`.
- Added `src/data_paths.py` to require and validate the configurable
  `PARKING_DATA_ROOT` environment variable.
- Rejects missing variables, relative paths, repository-internal paths, missing
  directories, and unreadable or unwritable storage roots with clear errors.
- Added tests for missing, relative, repository-local, and valid external paths.
- Documented that code stays in GitHub while archives and extracted images stay
  on the external SSD.
- Confirmed that split manifests contain relative paths and IDs only; no physical
  train/validation/test image copies will be created.
- Downloaded the two official 150×150 patch archives directly to external
  storage and recorded exact sizes and SHA-256 checksums.
- Passed ZIP integrity checks and extracted 12,584 CNRPark plus 144,965 CNR-EXT
  JPGs under the external root.
- Added `src/audit_image_paths.py`; all 157,549 metadata paths resolve to files on
  the external SSD.
- Removed 157,549 macOS-generated `._*.jpg` AppleDouble sidecars from the exFAT
  extraction. They were not dataset images; real JPGs and source ZIPs remain.
- Added `src/audit_images.py` and scanned every image with Pillow and OpenCV;
  there were no decode errors.
- Audited dimensions, color modes, file sizes, exact hashes, uniform images, and
  train-only RGB channel statistics.
- Added `data/IMAGE_EXCLUSIONS.csv` for 39 confirmed uniformly black placeholders
  and regenerated both split protocols with `excluded_quality` assignments.
- Confirmed that no exact duplicate group crosses an active date or camera split.
- Documented EDA findings and preprocessing decisions in `docs/EDA_REPORT.md`.
- Added a manifest-backed dataset loader that resolves images through
  `PARKING_DATA_ROOT`, filters quality exclusions, and preprocesses in memory.
- Added a 98,178-parameter `SimpleCNN`, binary metrics, checkpoint selection, and
  a reproducible MPS training command.
- Added tests for dataset filtering/transforms, model output shape, and metrics.
- Trained for 5 epochs on 106,833 primary date-split samples and used only the
  18,938-sample validation split for model selection.
- Selected epoch 3 with validation occupied-class F1 `0.980611`; the primary test
  split was not loaded.
- Saved the small tracked config/history/metrics JSON under `results/` and the
  ignored local checkpoint under `models/`.
- Verified the checkpoint by loading it into a fresh model instance and checking
  its config, best epoch, sample counts, and untouched-test flag.
- Documented the complete experiment in `docs/SIMPLE_CNN_BASELINE.md`.
- Added an ImageNet-normalized ResNet18 data transform and transfer-learning
  model with explicit backbone freeze/unfreeze helpers.
- Kept the same CNR-EXT date-grouped train/validation assignments and quality
  filters used by SimpleCNN; the primary test split was not instantiated.
- Trained the ImageNet-1K V1 pretrained ResNet18 for one frozen-backbone head
  epoch followed by two full fine-tuning epochs on Apple MPS.
- Selected epoch 3 with validation occupied-class F1 `0.995199`, accuracy
  `0.995248`, and confusion matrix `[[9519, 24], [66, 9329]]`.
- Improved validation occupied-class F1 by `0.014589` and accuracy by `0.014363`
  over SimpleCNN under the same validation assignments.
- Stored the pretrained-weight cache on external storage, the ignored local
  checkpoint under `models/`, and the small tracked result JSON under `results/`.
- Loaded the checkpoint into a fresh model and verified its best epoch, sample
  count, state dictionary, and untouched-test flag.
- Documented the experiment and baseline comparison in
  `docs/RESNET18_TRANSFER.md`.
- Downloaded the official 4,898,276,304-byte `PKLot.tar.gz` directly to external
  storage and recorded SHA-256
  `e89bbc1dc735298c478688d50c7a682fb3b0076a87b6634923132709f2d2fa9b`.
- Passed complete gzip/tar integrity and archive-path safety checks, then
  extracted the original full-frame/XML and `PKLotSegmented` structures.
- Removed only 721,277 macOS-generated `._*` AppleDouble sidecars, recovering
  SSD space while retaining the archive and every real JPG/XML file.
- Confirmed 12,417 full frames, 12,416 XML files, and 695,851 segmented patches;
  documented the one full frame lacking XML and the 48-patch difference from the
  paper's reported total.
- Added portable PKLot path resolution and `src/prepare_pklot_manifest.py`.
- Found 78 image IDs duplicated across conflicting Empty/Occupied folders. Both
  copies of each conflict were excluded; all pairs have identical content.
- Generated a 695,695-row gzip manifest with unique IDs, 12,508 source frames,
  only relative image paths, and `used_for_model_selection: false`.
- Documented acquisition, storage, conflict exclusions, distributions, and the
  cross-domain boundary in `docs/PKLOT_PREPARATION.md`.
- Added a gzip-manifest-backed PKLot dataset loader and frozen cross-domain
  evaluation pipeline with overall, location, site, and weather aggregation.
- Locked the epoch-3 ResNet18 checkpoint and verified SHA-256 before evaluation;
  no weights, preprocessing, or hyperparameters changed afterward.
- Opened the reserved 19,155-sample CNR-EXT test split once after model selection;
  accuracy was `0.988567` and occupied F1 was `0.988692`.
- Evaluated all 695,695 eligible PKLot patches without target-domain model
  selection; accuracy was `0.743097` and occupied F1 was `0.788652`.
- Measured absolute cross-domain drops of `0.245470` accuracy and `0.200040`
  occupied F1 relative to the one-time CNR-EXT test.
- Identified 174,485 PKLot false positives as the dominant error, with UFPR04
  performing worst and UFPR05 best among sites.
- Saved `results/cross_domain_evaluation.json` and documented the complete frozen
  evaluation in `docs/CROSS_DOMAIN_EVALUATION.md`.
- Added a deterministic SHA-256-ranked PKLot adaptation split grouped by complete
  `(site, capture_date)` units and precommitted it before changing weights.
- Assigned 47,742 samples from six dates to adaptation train, 32,300 samples from
  six dates to adaptation validation, and 615,653 samples from 89 dates to
  held-out evaluation.
- Passed site-date, source-frame, and duplicate-image leakage checks with zero
  violations; no physical image copies were created.
- Evaluated the frozen checkpoint on the dedicated held-out set, obtaining
  accuracy `0.762163` and occupied F1 `0.809347`.
- Ran exactly one head epoch and one full fine-tune epoch under the precommitted
  configuration; selected epoch 2 with validation occupied F1 `0.971022`.
- Fine-tuned held-out accuracy reached `0.986387` and occupied F1 reached
  `0.986545`, improvements of `0.224224` and `0.177197` respectively.
- Reduced held-out false positives from 142,636 to 1,040 while occupied recall
  decreased by `0.011291`; CNR-EXT test was not reopened.
- Saved `results/pklot_finetuning.json`, the ignored local fine-tuned checkpoint,
  and full documentation in `docs/PKLOT_ADAPTATION_SPLIT.md` and
  `docs/PKLOT_FINETUNING.md`.
- Added `src/analyze_errors.py` to replay the locked fine-tuned model on the
  existing PKLot held-out split and save only wrong predictions with portable
  metadata and confidence values.
- Verified the checkpoint and adaptation manifest SHA-256 values before
  analysis; weights and the 0.5 threshold were unchanged, and CNR-EXT test was
  not reopened.
- Reproduced the exact Milestone 8 confusion matrix
  `[[300027, 1040], [7341, 307245]]` across all 615,653 held-out samples.
- Saved an 8,381-row compressed local error manifest with 1,040 false positives
  and 7,341 false negatives. It contains relative paths only and has SHA-256
  `c3ae7d4c7652a0fc4e393ec39431295eb619f6459d4d4f3d6f8631d096d97afe`.
- Found that UFPR04 contributes 6,117 false negatives and 73.08% of all errors;
  its occupied recall is 0.854565 versus above 0.995 for PUC and UFPR05.
- Found that sunny metadata accounts for 68.58% of errors, while noting that
  weather is confounded with site/date and is not a causal lighting result.
- Created two worst-confidence contact sheets and one site/weather representative
  sheet without copying or reorganizing the underlying dataset.
- Documented potential annotation/crop ambiguities visible among the most
  confident errors; no labels were changed and no unsupported lighting,
  occlusion, or shadow rates were invented.
- Added unit coverage for portable error rows, deterministic ranking, and
  representative selection; all 16 tests pass.
- Saved the aggregate analysis in `results/error_analysis.json` and documented
  the protocol, findings, artifacts, and limitations in
  `docs/ERROR_ANALYSIS.md`.
- Added `src/inference.py` as the reusable single-image inference layer for the
  locked PKLot fine-tuned ResNet18 checkpoint.
- Added a minimal Streamlit interface in `app/app.py` for one cropped
  parking-space image, returning `EMPTY` or `OCCUPIED`, predicted-class
  confidence, and both class scores.
- Reused the exact 224×224 edge-padding, resizing, and ImageNet normalization
  from the completed ResNet18 experiments; no weights or threshold changed.
- Added configurable `PARKING_MODEL_PATH` checkpoint resolution with the local
  ignored Milestone 8 checkpoint as the repository-relative default.
- Added content-based JPEG/PNG/WebP decoding, a 10 MB upload limit, empty/corrupt
  upload handling, and explicit missing/incompatible-checkpoint errors.
- Confirmed the demo does not require `PARKING_DATA_ROOT`, access the external
  SSD, save uploaded images, or implement full-frame parking-space detection.
- Installed and recorded Streamlit as the only new project dependency for this
  milestone.
- Loaded the real checkpoint into a fresh CPU model and passed a direct
  single-image inference smoke test.
- Opened the app at localhost and verified the initial state, 10 MB limit,
  supported formats, file upload, image preview, label, confidence, and both
  class scores through the browser.
- Added a small demonstration sample, initial/result screenshots, and
  `images/parking_occupancy_demo.gif`. These are documentation derivatives, not
  evaluation data or new metrics.
- Added inference/upload/checkpoint tests; all 18 automated tests pass.
- Documented setup, inference contract, validation, and limitations in
  `docs/DEMO.md`, and added launch instructions and the GIF to `README.md`.
- Inserted Milestone 10.5 between Demo and Portfolio Finalization in
  `PROJECT_PLAN.md` at the user's explicit request.
- Added `src/freeze_v1_artifacts.py` and created
  `data/V1_ARTIFACT_LOCK.json`, a byte-size/SHA-256 registry for all completed
  V1 checkpoints, result JSONs, local adaptation/error manifests, and milestone
  reports. Later runs verify the registry instead of overwriting it.
- Locked the V1 PKLot checkpoint SHA-256
  `e61137bf2fbc259a2b7bd22ecc9840dda8c2668d9f73b7352bbaa3d47809e7ce`
  and retained every completed V1 metric and error-analysis statement without
  modification or reinterpretation.
- Added `src/prepare_v2_protocol.py`; it reads only the clean V1 adaptation
  manifest and does not read or reuse the held-out error manifest.
- Precommitted a deterministic SHA-256-ranked V2 split grouped by complete
  `(site, capture_date)` units. Final dates are reserved first and can only come
  from dates excluded from V1 adaptation.
- Assigned 78,376 samples from 12 dates to V2 train, 42,148 samples from six
  dates to V2 validation, 154,669 samples from 21 dates to fresh final, and
  420,502 samples from 62 dates to unused reserve.
- Passed all V2 leakage checks: zero site-date, source-frame, or image-ID leaks;
  zero fresh-final dates used by V1 adaptation or V2 development.
- Saved the tracked authoritative 101-row date assignment in
  `data/PKLOT_V2_DATE_SPLIT.csv` and the ignored reproducible full manifest in
  `data/processed/pklot_v2_protocol_manifest.csv.gz` with SHA-256
  `4aa2e976d67983b44708d304343ed6c72d7f054f294106ebf3b14e93dd00c5d7`.
- Precommitted the complete training/evaluation configuration in
  `data/PKLOT_V2_EXPERIMENT_CONFIG.json`: site-and-class-balanced sampling, mild
  augmentation, one ResNet18 candidate, one controlled EfficientNet-B0
  candidate, validation-only selection, and a one-time V1/V2 fresh-final
  comparison.
- Explicitly forbade the error manifest, aggregate error JSON, and error contact
  sheets as V2 training, sampling, validation, or selection inputs.
- Precommitted the final robustness gate: at least +0.02 absolute UFPR04
  occupied recall with no more than -0.005 absolute overall occupied F1. The
  gate is reported only after selection and cannot drive post-final changes.
- Documented the design and important limitation in
  `docs/MODEL_ROBUSTNESS_PROTOCOL.md`: every PKLot image previously received a
  V1 prediction, so fresh final means V2-development-disjoint and
  V1-adaptation-disjoint, not never previously inferred by V1.
- Saved the machine-readable protocol audit in `results/v2_protocol.json`.
- Added deterministic V2 assignment tests; all 19 tests pass. No images were
  loaded, no model was trained, and fresh-final inference was not opened.
- Added `src/v2_training.py` with the locked augmentation, development-only
  dataset boundary, site-and-label sampler, EfficientNet-B0 constructor, MPS
  training loop, per-site validation metrics, and lexicographic validation rank.
- Added `src/train_v2_candidates.py`; it requires the exact precommitted config,
  V2 manifest, and V1 checkpoint hashes, refuses fresh-final dataset names,
  refuses checkpoint/selection-lock overwrites, and records partial results
  after every epoch.
- Trained V2-A ResNet18 for exactly two full fine-tuning epochs using only
  78,376-sample balanced draws from `v2_train` and selected its epoch 2 using
  only the 42,148-sample `v2_validation` set.
- V2-A epoch 2 validation accuracy/F1 are 0.999478/0.999401 with confusion
  `[[23777, 14], [8, 18349]]`; macro-site F1 is 0.999489 and minimum-site recall
  is 0.999512.
- V2-A UFPR04 validation recall is 1.0 with confusion
  `[[5257, 2], [0, 1551]]`.
- Saved `models/v2a_balanced_resnet18.pt` with SHA-256
  `97b039fa7d4125e993903c4d1b485a7bc8e58d47cf7917c5fef8515e6982d5f9`.
- Trained V2-B EfficientNet-B0 for exactly one head epoch and two full
  fine-tuning epochs under the same split, sampler, augmentation, seed, loss,
  and validation-selection protocol; selected its epoch 3.
- V2-B epoch 3 validation accuracy/F1 are 0.998909/0.998748 with confusion
  `[[23756, 35], [11, 18346]]`; macro-site F1 is 0.999104 and minimum-site
  recall is 0.998048.
- Saved `models/v2b_balanced_efficientnet_b0.pt` with SHA-256
  `d3baa8374b012954aa24892bd53313a6d0735cc87b4ae3d4d8bd6ce688650185`.
- Applied the precommitted validation ranking exactly. V2-A ResNet18 epoch 2 won
  on primary macro-site F1 and is the only selected V2 checkpoint.
- Locked the selection, both candidate rank tuples, config/manifest hashes, and
  selected checkpoint SHA in `data/V2_SELECTED_CHECKPOINT.json` with SHA-256
  `60743fb7594e05cd7c2726a2a1029853f99473cfe20179e15cae1c78ee084e9e`.
- Saved exact configs, seed 105, all five epoch histories, overall/per-site
  Accuracy/Precision/Recall/F1/confusion matrices, checkpoint hashes,
  MPS/runtime information, and library versions in
  `results/v2_training_selection.json`.
- Total training, validation, and selection runtime was 5,621.12 seconds. V2-A
  used 2,037.25 seconds and V2-B used 3,536.19 seconds.
- Verified that fresh-final dataset instantiated=false, fresh-final images
  opened=0, error artifacts read=false, and Streamlit Demo updated=false in the
  result, both checkpoints, and selection lock.
- Re-verified all 30 frozen V1 artifacts after training; no V1 file or completed
  metric changed.
- Added candidate/sampler/model tests; all 21 tests pass.
- Documented the complete training histories and validation-only selection in
  `docs/V2_TRAINING_SELECTION.md`.
- Re-verified all 30 frozen V1 artifacts immediately before the fresh-final run;
  all registered hashes matched and no completed V1 result changed.
- Re-verified the selected V2-A checkpoint SHA-256
  `97b039fa7d4125e993903c4d1b485a7bc8e58d47cf7917c5fef8515e6982d5f9`
  against `data/V2_SELECTED_CHECKPOINT.json` before opening final.
- Wrote `data/V2_FRESH_FINAL_OPENED.json` immediately before the first image was
  read, then evaluated V1 and V2-A together in one deterministic traversal of
  exactly 154,669 `v2_fresh_final_evaluation` samples.
- Used the same 224×224 edge-padding, bilinear resizing, ImageNet normalization,
  label definitions, evaluation code, sample order, and threshold 0.5 for both
  immutable ResNet18 models.
- V1 fresh-final accuracy/precision/recall/occupied-F1 are
  0.991834/0.998594/0.985315/0.991910; its confusion matrix is
  `[[75977, 109], [1154, 77429]]` and macro-site F1 is 0.973232.
- V2-A fresh-final accuracy/precision/recall/occupied-F1 are
  0.998894/0.998766/0.999058/0.998912; its confusion matrix is
  `[[75989, 97], [74, 78509]]` and macro-site F1 is 0.998763.
- UFPR04 occupied recall improved from 0.859053 to 1.000000, an absolute gain
  of 0.140947. Overall occupied F1 improved by 0.007002.
- Both precommitted robustness conditions passed: UFPR04 recall gain was at
  least +0.02 and overall occupied-F1 degradation was no worse than -0.005.
- Saved exact overall, per-site, per-weather, FP/FN, hashes, runtime/device,
  comparison, and gate data in `results/v2_fresh_final_comparison.json`.
- Saved the human-readable comparison table and audit summary in
  `docs/V2_FRESH_FINAL_COMPARISON.md`.
- The one-time run used Apple MPS and took 726.74 seconds. EfficientNet-B0 was
  not evaluated; neither model, the Streamlit demo, nor any V1 artifact changed.
- Added the locked final evaluator and gate tests; all 23 automated tests pass.

## Current State

- The CNRPark+EXT metadata CSV is present under ignored local storage.
- Each new shell must set `PARKING_DATA_ROOT` to the existing writable directory
  on the external SSD before image data can be used.
- The user-provided external storage root passed the validation check; its actual
  machine-specific path is intentionally not recorded in the repository.
- The two patch archives and extracted JPGs are present only on the external SSD.
- No CNRPark full-frame archive or published split bundle has been downloaded.
- The official ResNet18 pretrained-weight cache is stored outside the repository
  under the configurable external dataset root.
- The PKLot archive and extraction are present only on the external SSD. The
  extraction occupies about 91 GiB on exFAT; approximately 94 GiB remained after
  AppleDouble cleanup.
- The compressed PKLot manifest, summary, and small exclusion CSV are local
  repository metadata; they contain no private absolute storage path.
- The Simple CNN checkpoint exists locally and is ignored by Git.
- The ResNet18 checkpoint exists locally and is ignored by Git.
- The primary CNR-EXT test split has now been evaluated once with the locked
  ResNet18 checkpoint. It must not be reused for model selection.
- PKLot cross-domain metrics are final for the frozen Milestone 7 checkpoint and
  must not be used to retroactively tune that model.
- The fine-tuned PKLot checkpoint exists locally and is ignored by Git. Its
  SHA-256 is
  `e61137bf2fbc259a2b7bd22ecc9840dda8c2668d9f73b7352bbaa3d47809e7ce`.
- Milestone 8 before/after metrics use the same 615,653 held-out samples. Those
  held-out labels must not be used to revise the completed fine-tuning run.
- The local ignored error manifest contains only held-out mistakes and portable
  metadata; it must not be promoted into adaptation train/validation data.
- Three tracked contact sheets are small documentation derivatives. The source
  PKLot images and all large archives remain only on external storage.
- The Streamlit demo runs locally with the ignored selected V2-A checkpoint. It
  is not deployed and does not bundle model weights.
- The demo accepts only a pre-cropped single parking-space image; full-frame
  parking-space localization remains out of scope.
- V1 artifacts are frozen by `data/V1_ARTIFACT_LOCK.json`. V2 work must create
  new checkpoint and result filenames and must never overwrite registered paths.
- The V2 split/config hashes are precommitted. Any change requires a separately
  versioned protocol before training, not an in-place edit.
- Fresh final has been opened exactly once and the comparison is complete. It
  must not be reopened or used for further model updates.
- The Streamlit interface layout remains unchanged; its authorized Milestone 11
  checkpoint integration now defaults to and validates only selected V2-A.
- The selected V2 checkpoint is `models/v2a_balanced_resnet18.pt`; it is ignored
  by Git and must match the selection-lock SHA before any final evaluation.
- The unselected EfficientNet-B0 checkpoint and full validation history are
  retained for audit but cannot replace the selected model after final opens.

## Key Decision

CNRPark+EXT is the primary dataset because it supplies official 150×150 patches,
has a smaller acquisition footprint, and exposes camera, capture time, weather,
and slot information. Since it represents one physical parking area, a held-out
camera must be reported as camera-domain generalization rather than true
cross-location generalization. PKLot provided the Milestone 7 location-domain
test.

## Split Decision

- Use the date-grouped CNR-EXT protocol for the future in-domain baseline.
- Keep all patches from a date and shared timestamp group in one split.
- Use the camera-and-slot holdout only as a separately labeled viewpoint-domain
  experiment.
- Keep CNRPark cameras A/B as a cross-subset diagnostic, not a cross-location
  claim.
- Use PKLot for the completed genuine cross-location evaluation.
- For Milestone 10.5, use `data/PKLOT_V2_DATE_SPLIT.csv` as the authoritative
  date grouping: 12 train, six validation, 21 fresh-final, and 62 unused-reserve
  site-date groups.
- Balance V2 training draws across `(site, occupancy)` cells while keeping
  validation/final distributions natural.
- Select between the two precommitted V2 candidates using validation metrics
  only, then lock one checkpoint before the one-time V1/V2 final comparison.

## External Storage Setup

Set the variable in the local shell, using the actual external SSD directory:

```bash
export PARKING_DATA_ROOT="/absolute/path/to/external-ssd/parking-datasets"
python3 src/data_paths.py
```

Do not commit the actual machine-specific path.

## Next Step

Stop. Milestone 11 and the planned portfolio deliverables are complete. The
repository is ready for user review and normal Git/GitHub publication workflow.
Do not reopen fresh final or start another modeling cycle from its results.

## User Decision Required

No research or modeling decision is pending. Optional publication actions such
as committing, pushing, deploying, or editing personal application details were
not performed and remain under user control.
