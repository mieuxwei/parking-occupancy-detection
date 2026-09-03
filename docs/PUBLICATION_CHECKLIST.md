# Frozen Portfolio Publication Checklist

This records the completed publication and its verification evidence, not a new
research milestone. The research and final model remain frozen.

- Published release: [v1.0.0 — Frozen Portfolio Release](https://github.com/mieuxwei/parking-occupancy-detection/releases/tag/v1.0.0).
- Frozen tag target: `47cb05f6e6d835e850c8dfb40ba71363d3e802c0`.
- Published at: **2026-09-04 03:13:11 Asia/Taipei (UTC+08:00)**, verified from
  GitHub's publication timestamp. This is the release date, not an inferred
  research completion date.

## Starting state and preservation

- Branch: `main`.
- Pre-change HEAD: `bea45ed03a7662a9cc265f04ff01145f9a4e92ef` (`Sample Gallery`).
- Existing work: modified root README and documentation index, plus an untracked
  Chinese technical overview. Their content and entry points were integrated.
- No reset, history rewrite, model update, or external repository edit.
- The GitHub API initially reported this repository public and not archived,
  with empty description/homepage/topics. The owner browser session subsequently
  applied and verified the exact About values below; visibility and archival
  settings were not changed.
- Before publication, local `v1.0.0` was absent and public tags/releases were
  empty. The new tag and release were created without overwriting any version.

## Verification status

- Local unittest suite: **26 tests passed** in the existing macOS environment.
- `pip check`, Python compilation, workflow YAML syntax, and `git diff --check`
  passed. All 87 Markdown relative links resolved; the root README has exactly
  one table. These are local checks, not a substitute for a hosted CI run.
- JPEG/PNG/WebP decoding passed locally; empty, corrupt, and over-10-MB payloads
  were rejected. The existing inference implementation was not edited.
- All 57 tracked frozen/source/demo/test artifacts captured before editing
  retained identical SHA-256 values. The checkpoint still matches
  `97b039fa7d4125e993903c4d1b485a7bc8e58d47cf7917c5fef8515e6982d5f9`.
- README, abstract, summary, and comparison values were checked against the
  existing result JSON by read-only inspection; no metric was recomputed.
- Linux CPU workflow: **passed on GitHub** for the frozen tag target above.
  [Run 33794482408](https://github.com/mieuxwei/parking-occupancy-detection/actions/runs/33794482408)
  passed dependency installation, CPU runtime verification, checkpoint SHA-256,
  portable unit tests, Python compilation, and whitespace checks. This hosted
  Linux result is separate from the local macOS tests.
- Live browser: app was asleep; the wake button loaded the interface without a
  sign-in prompt. Sample A showed EMPTY; switching to Sample B showed OCCUPIED.
- Live JPEG upload: the existing demonstration crop displayed a preview,
  OCCUPIED, confidence, and both class scores. The uploader displayed JPEG,
  PNG, WebP and a 10 MB limit.
- Streamlit's Share dialog showed **Make this app public** already checked.
  No Streamlit sharing or model setting needed to be changed.
- Clean anonymous session: **independently confirmed by the project owner**
  before release publication, including sample and upload operation. The
  automated browser used an existing session; it is not presented as the
  source of the clean-session verification.
- Two legacy document URLs returned HTTP 200. Neutral local compatibility pages
  preserve both paths after publication.
- Social preview: [PNG](../images/social_preview.png), 1280×640, with
  [editable SVG](../images/social_preview.svg). Visually checked; derived from
  the existing domain-shift bars and V1→V2 recall chart, with their palette,
  exact values, evaluation boundaries, and no unrelated scene imagery.
  Uploaded to GitHub; the General settings page displayed the correct preview.

No passing badge has been added. No research result has been recomputed.

## GitHub About — applied and verified

The repository's **About** edit control was used to save these exact values:

**Description**

```text
Independent PyTorch study of leakage-aware cross-domain parking-space occupancy classification from CNRPark+EXT to PKLot.
```

**Website**

```text
https://parking-occupancy-detection-hk9l6wzyvtkrqjr6tkvftc.streamlit.app/
```

**Topics**

```text
python pytorch computer-vision image-classification transfer-learning domain-adaptation model-robustness reproducible-research resnet streamlit
```

No object-detection, YOLO, real-time, or localization topics were added.
**Settings → General → Social preview** now uses `images/social_preview.png`.
Only these explicitly requested metadata settings were changed, using the
existing owner session without a new login. The repository was not archived.

## Anonymous demo recheck procedure

The release acceptance check is complete (owner-confirmed above). These steps
remain available for future hosting checks; they are not outstanding tasks.

1. Open a new Chrome Incognito / Firefox Private window with no Streamlit login.
2. Open the exact Live Demo URL above. If asleep, use the wake button.
3. Switch sample A to B; confirm image, label, confidence, and both scores.
4. Upload one pre-cropped JPEG/PNG/WebP; confirm preview and result.
5. If a login/access request actually blocks viewing, use the owner's Streamlit
   workspace, open this app's **Share / sharing settings**, and enable the public
   viewing option (anyone with the link), then repeat steps 1–4.
6. If the owner lacks that setting, resolve account/workspace access with
   Streamlit; do not change model code or weights as a workaround.

The GIF and local launch instructions remain available during hosting downtime.

## Commit and push — completed

- `0a8becb178b34ea4d81ec5de08b555c4a5a40462`: presentation, documentation, and CPU CI.
- `47cb05f6e6d835e850c8dfb40ba71363d3e802c0`: workflow trailing-blank-line cleanup.
- Both commits were pushed to `main`; the latter passed the hosted CI run above.

## Tag and Release — published

[v1.0.0 — Frozen Portfolio Release](https://github.com/mieuxwei/parking-occupancy-detection/releases/tag/v1.0.0)
is a published, non-prerelease version at the exact verified commit above.
The release body records the checkpoint hash, scoped results, commit-pinned
evidence links, CI success, owner-confirmed anonymous access, and limitations.

The [original preparation draft](releases/v1.0.0.md) is retained as historical
pre-publication material; the published release is the authoritative version.
Subsequent documentation maintenance does not move or recreate the frozen tag.
No research artifact, model, split, threshold, or evaluation result changed.
