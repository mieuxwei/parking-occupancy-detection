# Final V2-A Streamlit Demo / 最終展示版本

The final portfolio demo retains the original minimal single-crop interface and
changes only the selected checkpoint integration. 最終介面維持原始設計，只將模型指向
Milestone 10.5 選出的 V2-A ResNet18。

## Production inference contract

- Checkpoint: `models/v2a_balanced_resnet18.pt`
- SHA-256: `97b039fa7d4125e993903c4d1b485a7bc8e58d47cf7917c5fef8515e6982d5f9`
- Required metadata: candidate `v2a_balanced_resnet18`, architecture `ResNet18`, locked V2 config hash
- Input: one cropped JPEG, PNG, or WebP image, at most 10 MB
- Preprocessing: symmetric edge-pad to square, bilinear resize to 224×224, ImageNet mean/std normalization
- Decision: occupied softmax probability ≥ 0.5 → `OCCUPIED`; otherwise `EMPTY`
- Upload handling: decoded in memory and not saved by the app

The checkpoint path remains configurable through `PARKING_MODEL_PATH`, but the
loader rejects checkpoints that do not identify the selected V2-A candidate.
If unset, the repository-relative V2-A path is used.

```bash
export PARKING_MODEL_PATH="/absolute/path/to/v2a_balanced_resnet18.pt"
streamlit run app/app.py
```

## Validation

- Re-verified the V2-A checkpoint and selection-lock SHA-256 values.
- Loaded the real V2-A checkpoint into a fresh CPU predictor.
- Ran the existing cropped occupied demo sample; output was `OCCUPIED` with
  finite scores summing to one.
- Verified the initial state, supported formats, 10 MB limit, upload preview,
  label, confidence, and both class scores through the browser.
- All automated tests pass after the checkpoint-format update.

![V2-A demo](../images/v2_parking_occupancy_demo.gif)

This is a UI smoke test, not a new evaluation. It does not reopen fresh final or
add a metric. The earlier Milestone 10 V1 demo files remain preserved separately.

## Limitations

- The input must already be cropped to one parking space; no full-frame detection is implemented.
- Confidence is an uncalibrated softmax score.
- The model checkpoint is ignored by Git and must be supplied locally.
- The locked fresh-final result cannot be used for further calibration or model development.

