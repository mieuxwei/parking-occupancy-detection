import json
import unittest
from pathlib import Path

from src.inference import decode_uploaded_image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_MANIFEST = REPOSITORY_ROOT / "app/demo_samples.json"


class DemoSampleTests(unittest.TestCase):
    def test_demo_samples_are_small_purpose_created_non_evaluation_assets(self) -> None:
        samples = json.loads(SAMPLE_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(len(samples), 10)
        self.assertEqual(
            {label: sum(sample["known_label"] == label for sample in samples) for label in ("EMPTY", "OCCUPIED")},
            {"EMPTY": 5, "OCCUPIED": 5},
        )
        self.assertEqual({sample["known_label"] for sample in samples}, {"EMPTY", "OCCUPIED"})
        self.assertEqual(len({sample["id"] for sample in samples}), len(samples))

        for sample in samples:
            relative_path = Path(sample["image"])
            self.assertEqual(relative_path.parts[:2], ("images", "demo_samples"))
            self.assertIn("Purpose-created", sample["source"])
            self.assertIs(sample["evaluation_evidence"], False)
            image_path = REPOSITORY_ROOT / relative_path
            self.assertLess(image_path.stat().st_size, 500_000)
            image = decode_uploaded_image(image_path.read_bytes())
            self.assertEqual(image.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
