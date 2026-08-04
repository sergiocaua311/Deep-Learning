import tempfile
import unittest
from pathlib import Path

import numpy as np

from esc50.evaluation import (
    classification_metrics,
    expected_calibration_error,
    fit_temperature,
    perturb_audio,
    save_prediction_artifacts,
)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([0, 1, 2, 3])
        self.logits = np.array(
            [
                [8.0, 0.0, 0.0, 0.0],
                [0.0, 8.0, 0.0, 0.0],
                [0.0, 0.0, 8.0, 0.0],
                [0.0, 0.0, 0.0, 8.0],
            ]
        )

    def test_metrics_for_perfect_predictions(self):
        metrics = classification_metrics(self.labels, self.logits)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["f1_macro"], 1.0)
        self.assertEqual(metrics["top5_accuracy"], 1.0)
        self.assertGreaterEqual(metrics["ece"], 0.0)

    def test_temperature_uses_validation_logits(self):
        temperature = fit_temperature(self.labels, self.logits * 4)
        self.assertGreaterEqual(temperature, 0.25)
        self.assertLessEqual(temperature, 4.0)

    def test_noise_is_deterministic(self):
        audio = np.linspace(-0.5, 0.5, 16_000, dtype=np.float32)
        first = perturb_audio(audio, "noise_10", "clip.wav", seed=42)
        second = perturb_audio(audio, "noise_10", "clip.wav", seed=42)
        np.testing.assert_array_equal(first, second)

    def test_artifacts_are_written(self):
        with tempfile.TemporaryDirectory() as directory:
            names = [f"class-{index}" for index in range(4)]
            save_prediction_artifacts(directory, self.labels, self.logits, names)
            for filename in (
                "metrics.json",
                "predictions.csv",
                "classification_report.csv",
                "confusion_matrix.csv",
                "confusion_matrix.png",
                "calibration.csv",
                "calibration.png",
            ):
                self.assertTrue((Path(directory) / filename).exists(), filename)


if __name__ == "__main__":
    unittest.main()
