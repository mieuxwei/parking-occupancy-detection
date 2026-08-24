# Milestone 10 — Single-image inference demo

## Purpose

The Streamlit demo classifies one already cropped parking-space image as
`EMPTY` or `OCCUPIED` and displays the model confidence and both class scores.
It uses the selected Milestone 8 fine-tuned ResNet18 without retraining or
changing its 0.5 decision rule.

The demo is deliberately limited to classification. It does not detect parking
spaces in a full parking-lot image, copy dataset files, or require
`PARKING_DATA_ROOT`.

## Setup and launch

Install the repository requirements and make the ignored fine-tuned checkpoint
available locally:

```bash
python -m pip install -r requirements.txt
export PARKING_MODEL_PATH="/absolute/path/to/resnet18_pklot_finetuned.pt"
streamlit run app/app.py
```

If `PARKING_MODEL_PATH` is unset, the app looks for
`models/resnet18_pklot_finetuned.pt` relative to the repository root. A missing
or incompatible checkpoint produces a visible setup error rather than silently
falling back to another model.

## Inference contract

- Accepted formats: JPEG, PNG, and WebP
- Maximum upload size: 10 MB
- Input: one cropped parking-space image
- Preprocessing: edge-pad to square, resize to 224×224, convert to tensor, and
  apply ImageNet mean/std normalization
- Model: selected PKLot fine-tuned ResNet18, dropout 0.20, two output classes
- Output: `EMPTY` or `OCCUPIED`, predicted-class softmax confidence, and both
  class scores
- Uploaded image handling: decoded in memory; not saved by the app

`src/inference.py` owns checkpoint resolution, upload decoding, preprocessing,
model loading, and prediction so UI code does not duplicate model logic.

## Validation

- The real ignored fine-tuned checkpoint loaded successfully in a fresh model.
- A CPU smoke test returned two finite class scores summing to one.
- The Streamlit page was opened at localhost and tested through the browser.
- The initial state showed supported formats and the 10 MB limit.
- A local cropped demonstration sample uploaded and decoded successfully.
- The page displayed `OCCUPIED`, confidence, and both class scores for that
  sample. This is a UI smoke test, not a new evaluation result.
- Invalid-image decoding and missing-checkpoint setup guidance are covered by
  automated tests.

The validated two-state demonstration is saved as
`images/parking_occupancy_demo.gif`; the initial and result screenshots are
stored beside it.

## Limitations

- The displayed confidence is an uncalibrated softmax score.
- The app expects a single parking-space crop and cannot process a complete lot.
- Unfamiliar sites, viewpoints, crop boundaries, or image conditions can reduce
  reliability.
- The selected checkpoint remains ignored by Git and is not bundled with the
  application.
- The demonstration sample verifies the interface only; it does not add to or
  alter any reported test metrics.

