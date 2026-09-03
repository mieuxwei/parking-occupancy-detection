# Final Frozen Demo

**Independent AI Research Project · Completed and Frozen.**

[Open the public Streamlit app](https://parking-occupancy-detection-hk9l6wzyvtkrqjr6tkvftc.streamlit.app/).

## Locked demo inference contract

- Final frozen demo model: `models/v2a_balanced_resnet18.pt`.
- SHA-256: `97b039fa7d4125e993903c4d1b485a7bc8e58d47cf7917c5fef8515e6982d5f9`.
- Required metadata: candidate `v2a_balanced_resnet18`, architecture `ResNet18`,
  and the locked V2 config hash.
- Input: one pre-cropped JPEG, PNG, or WebP image, at most 10 MB.
- Preprocessing: symmetric edge padding to square, bilinear antialiased resize
  to 224×224, ImageNet mean/std normalization.
- Labels: `0=EMPTY`, `1=OCCUPIED`.
- Decision: occupied softmax score ≥ 0.5 → `OCCUPIED`; otherwise `EMPTY`.
- Uploads are decoded in memory, not saved by the app.
- Hosted inference explicitly uses CPU.

The path is configurable through `PARKING_MODEL_PATH`, but the loader rejects
any checkpoint other than the exact locked V2-A artifact. The repository keeps
only this 44,790,987-byte checkpoint (42.72 MiB) in ordinary Git for immediate
local use. It needs no external SSD, dataset, credential, or API key.
The small result JSON is read only to display existing metrics; no evaluation
is run.

## Interactive use

1. **Try a Sample:** select one of ten demonstration scenes (five EMPTY, five
   OCCUPIED), or use **Try another sample** to advance through them.
2. **Upload Your Own:** upload one already cropped parking-space image.
3. Read the label, predicted-class confidence, and both class scores.

Sample labels appear after inference as demonstration ground truth, never as
evaluation metrics. Samples are not from any experiment split.
[Source and attribution](../images/demo_samples/README.md).

This is single-crop classification, not automatic parking-space localization,
full-lot detection, or real-time CCTV. Manual cropping is not implemented.
Confidence is an uncalibrated softmax score.

## Offline fallback and local use

![Recorded V2-A demo](../images/v2_parking_occupancy_demo.gif)

[Initial screenshot](../images/v2_demo_initial.png) ·
[Inference screenshot](../images/v2_demo_inference_result.png)

These preserved recordings predate the sample-gallery addition; they show the
same locked V2-A inference contract. If the app sleeps, use its wake-up button.
If it remains unavailable, the recording and local commands remain usable:

```bash
python -m pip install -r app/requirements.txt
streamlit run app/app.py
```

Use Python 3.11. See [reproduction and engineering checks](REPRODUCTION.md).
The original [V1 demo record](DEMO.md) remains historical evidence.

## Hosting settings

Streamlit Community Cloud: repository `mieuxwei/parking-occupancy-detection`,
branch `main`, entrypoint `app/app.py`, Python 3.11,
dependencies `app/requirements.txt`, no secrets. Anonymous viewing must be
enabled in the app's sharing settings; code changes do not replace that setting.

[Publication checklist](PUBLICATION_CHECKLIST.md) records what was actually
verified and which external actions remain pending.
