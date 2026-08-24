# Purpose-Created Demo Samples

These two lightweight images exist only to let reviewers try the public
classifier immediately. They were purpose-created with OpenAI's built-in image
generation tool on 2026-08-25 and were not copied from CNRPark, CNR-EXT, PKLot,
or any train/validation/held-out/fresh-final split.

| File | Demonstration label | Source | Evaluation evidence |
|---|---|---|---|
| `empty-purpose-created.jpg` | `EMPTY` | Purpose-created image generation | No |
| `occupied-purpose-created.jpg` | `OCCUPIED` | Purpose-created image generation | No |

The labels describe the deliberately composed scenes. The app reveals them only
after inference and identifies them as demonstration ground truth, never as an
accuracy or evaluation result. The images are included as project documentation
under the repository [MIT License](../../LICENSE).

## Prompt specifications

### EMPTY

Create a realistic, surveillance-style cropped view of exactly one empty
outdoor parking space. Center the complete painted space, use neutral daylight,
and include no vehicle, person, logo, text, watermark, bounding box, or UI.

### OCCUPIED

Create a realistic, surveillance-style cropped view of exactly one occupied
outdoor parking space. Center one unbranded compact car fully inside the painted
boundaries, using neutral daylight and no person, readable plate, logo, text,
watermark, bounding box, or UI.
