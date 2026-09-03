# Reproduction and Evidence

The research and final demo model are **completed and frozen**. Reproduction
here means inspecting recorded evidence and running portable engineering checks
or single-image demo inference. It does not authorize rerunning a terminal
evaluation, regenerating splits, or modifying an experiment artifact.

## Local demo

Use Python 3.11 from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r app/requirements.txt
python -m pip check
streamlit run app/app.py
```

No dataset, SSD, secret, or API key is required. The default checkpoint is
`models/v2a_balanced_resnet18.pt`. Only that frozen checkpoint is accepted.
See the [locked inference contract](FINAL_DEMO.md).

## Portable tests

```bash
python -m unittest discover -s tests -v
python -m compileall -q src app tests
git diff --check
```

Tests use temporary synthetic images/manifests and small repository fixtures.
They do not run training loops, download pretrained weights or datasets, or
open evaluation image sets. The test named `test_evaluate_v2_fresh_final` tests
the gate arithmetic only; it does not execute fresh-final evaluation. Reading
the locked result JSON to validate display values is not model evaluation.

## Clean Linux CPU CI

The [workflow](../.github/workflows/cpu-tests.yml) uses Ubuntu 24.04 and Python
3.11. It installs the matching PyTorch 2.8.0 / torchvision 0.23.0 CPU wheels,
then the pinned demo dependencies. No CUDA, external SSD, dataset archive, or
training is involved. It runs the same unittest command and checks the frozen
checkpoint hash. A passing badge must only be added after the actual published
commit has passed in GitHub Actions.

The CPU-wheel pairing follows the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/).
The runner setup follows [GitHub's Python workflow guidance](https://docs.github.com/en/actions/tutorials/build-and-test-code/python).

## Frozen evidence map

- [Baseline](SIMPLE_CNN_BASELINE.md) and [ResNet18](RESNET18_TRANSFER.md): same-split comparison.
- [Cross-domain report](CROSS_DOMAIN_EVALUATION.md): the unchanged source model across domains.
- [V1 adaptation](PKLOT_FINETUNING.md) and [error analysis](ERROR_ANALYSIS.md): completed held-out evidence.
- [V2 protocol](MODEL_ROBUSTNESS_PROTOCOL.md) and [selection](V2_TRAINING_SELECTION.md): development boundaries and candidate histories.
- [Final comparison](V2_FRESH_FINAL_COMPARISON.md): closed one-time result.
- [Exact results](../results/) and [portable configs/locks](../data/): machine-readable evidence.
- [Full model tables](MODEL_COMPARISON.md): comparisons retain their original sample sets.

Historical commands are retained for audit, not as instructions to rerun the
closed final protocol. Records retain their original tense and stage-specific
status. Large datasets and non-final checkpoints remain local-only. Public
manifests contain paths/IDs, not train/validation/test image copies.

## Checkpoint policy

The one public checkpoint is 44,790,987 bytes (42.72 MiB). Normal Git storage
lets a clone run the demo without an additional download service. Git LFS is not
used; no other model weights or dataset archives are added.

```text
97b039fa7d4125e993903c4d1b485a7bc8e58d47cf7917c5fef8515e6982d5f9
```

Threshold `0.5`, RGB image input, labels `0=EMPTY` / `1=OCCUPIED`, symmetric edge padding,
224×224 bilinear resize, and ImageNet normalization are immutable. The repository
remains maintainable and is not archived; the research itself is not reopened.
