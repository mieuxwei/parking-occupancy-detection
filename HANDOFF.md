# Project Handoff

## Current Milestone

Milestone 8 — Fine-tuning (completed on 2026-08-22).

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

## External Storage Setup

Set the variable in the local shell, using the actual external SSD directory:

```bash
export PARKING_DATA_ROOT="/absolute/path/to/external-ssd/parking-datasets"
python3 src/data_paths.py
```

Do not commit the actual machine-specific path.

## Next Step

Milestone 9 — Error Analysis. Use the already saved held-out results and selected
fine-tuned checkpoint to define a read-only error manifest containing image ID,
relative path, true label, prediction, confidence, site, weather, and date.
Analyze false positives and false negatives by site and weather, with particular
attention to UFPR04's recall tradeoff. Sample representative worst-confidence
errors for documented visual inspection. Do not retrain, recalibrate, or start
the demo milestone during error analysis.
