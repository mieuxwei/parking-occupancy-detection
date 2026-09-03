# Frozen Portfolio Publication Checklist

This records local preparation and the external settings actually applied; it
is not a new research milestone or a published code release. No completion date
is inferred from repository creation or commit timestamps.

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
- Local `v1.0.0` tag was absent; public tags and releases returned empty lists.
  Recheck immediately before publishing; never overwrite a later-created tag.

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
- Linux CPU workflow: prepared, **not yet run on GitHub**. Local Docker daemon
  was unavailable, so local tests are not represented as a Linux CI pass.
- Live browser: app was asleep; the wake button loaded the interface without a
  sign-in prompt. Sample A showed EMPTY; switching to Sample B showed OCCUPIED.
- Live JPEG upload: the existing demonstration crop displayed a preview,
  OCCUPIED, confidence, and both class scores. The uploader displayed JPEG,
  PNG, WebP and a 10 MB limit.
- Streamlit's Share dialog showed **Make this app public** already checked.
  No Streamlit sharing or model setting needed to be changed.
- Clean anonymous session: **still needs independent confirmation**. The
  available in-app browser has no fresh-cookie/incognito capability; no Chrome
  provider was available. A new tab without a sign-in prompt does not prove
  cookie isolation. Do not label this as a completed clean-session audit.
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

## Final anonymous demo check

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

## Commit and push — manual, after review

Run only in this repository. Review the staged diff and ensure it contains only
the intended presentation work; do not include unrelated staged work.

```bash
git status --short --branch
git diff --check
git add -- README.md THIRD_PARTY_NOTICES.md .github/workflows/cpu-tests.yml docs/README.md docs/PROJECT_OVERVIEW_AND_STATUS.md docs/PROJECT_ABSTRACT.md docs/PROJECT_HIGHLIGHTS.md docs/GRADUATE_APPLICATION_ABSTRACT.md docs/CV_PROJECT_DESCRIPTION.md docs/FINAL_RESEARCH_SUMMARY.md docs/MODEL_COMPARISON.md docs/FINAL_DEMO.md docs/REPRODUCTION.md docs/PUBLICATION_CHECKLIST.md docs/releases/v1.0.0.md images/demo_samples/README.md images/social_preview.svg images/social_preview.png
git diff --cached --stat
git diff --cached --check
git commit -m "docs: finalize frozen research presentation and add CPU CI"
git push origin HEAD:main
```

Then open GitHub **Actions → Linux CPU tests** and inspect the run for that
exact commit. Resolve environment/dependency/code failures explicitly. Only
after success may a status badge be added. Do not claim the local macOS test
run is the hosted Linux result.

## Tag and Release — only after the publication gates pass

1. Confirm the published commit and successful CI; copy `git rev-parse HEAD`.
2. Recheck local and remote tags, and the GitHub Releases page:

```bash
git tag --list v1.0.0
git ls-remote --tags origin refs/tags/v1.0.0 'refs/tags/v1.0.0^{}'
```

3. If either tag or Release exists, stop and report the conflict. Do not force,
   delete, or replace it.
4. In GitHub **Releases → Draft a new release**, create tag `v1.0.0` at the
   verified commit (not an unreviewed newer HEAD). Use title
   **v1.0.0 — Frozen Portfolio Release**.
5. Copy the [release draft](releases/v1.0.0.md), replace its commit placeholder
   and relative links as instructed, and remove draft-only instructions. Include
   the verified CI run URL. Publish only after the anonymous check passes.

No tag or Release was created during this task. The draft deliberately does not
invent the SHA of an uncreated commit.
