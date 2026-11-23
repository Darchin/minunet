import json
import math
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from nnunetv2.evaluation.convert_summary import convert_empty_pairs


def metrics(tp: int, fp: int, fn: int) -> dict:
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Dice": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0,
        "IoU": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
    }


def example_summary() -> dict:
    return {
        "metric_per_case": [
            {"metrics": {"0": metrics(4, 0, 0), "(1, 2)": metrics(0, 0, 0), "2": metrics(2, 0, 0)}},
            {"metrics": {"0": metrics(4, 0, 0), "(1, 2)": metrics(1, 1, 0), "2": metrics(0, 1, 0)}},
            {"metrics": {"0": metrics(4, 0, 0), "(1, 2)": metrics(0, 0, 0), "2": metrics(1, 0, 0)}},
        ],
        "mean": {"0": {}, "(1, 2)": {"TP": 1 / 3}, "2": {}},
        "foreground_mean": {"TP": 2 / 3},
    }


class TestConvertSummary(unittest.TestCase):
    def test_excludes_only_empty_pairs_and_averages_foreground_targets_equally(self):
        summary = example_summary()
        convert_empty_pairs(summary)

        cases = summary["metric_per_case"]
        self.assertTrue(math.isnan(cases[0]["metrics"]["(1, 2)"]["Dice"]))
        self.assertTrue(math.isnan(cases[0]["metrics"]["(1, 2)"]["IoU"]))
        self.assertEqual(cases[0]["metrics"]["2"], metrics(2, 0, 0))
        self.assertEqual(cases[1]["metrics"]["2"], metrics(0, 1, 0))
        self.assertEqual(summary["mean"]["(1, 2)"], {"Dice": 2 / 3, "IoU": 0.5, "TP": 1 / 3})
        self.assertAlmostEqual(summary["foreground_mean"]["Dice"], 2 / 3)
        self.assertAlmostEqual(summary["foreground_mean"]["IoU"], (0.5 + 2 / 3) / 2)
        self.assertEqual(summary["foreground_mean"]["TP"], 2 / 3)

    def test_entirely_empty_target_propagates_nan_to_foreground_mean(self):
        summary = example_summary()
        for case in summary["metric_per_case"]:
            case["metrics"]["(1, 2)"] = metrics(0, 0, 0)
        for _ in range(2):
            convert_empty_pairs(summary)
            for metric in ("Dice", "IoU"):
                self.assertTrue(math.isnan(summary["mean"]["(1, 2)"][metric]))
                self.assertTrue(math.isnan(summary["foreground_mean"][metric]))

    def test_directory_cli_mirrors_paths_and_refuses_overwrites(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            relative = Path("experiment/fold_0/validation/summary.json")
            source = root / "source" / relative
            destination = root / "converted" / relative
            source.parent.mkdir(parents=True)
            original = json.dumps(example_summary())
            source.write_text(original, encoding="utf-8")
            command = [
                sys.executable, "-m", "nnunetv2.evaluation.convert_summary",
                str(root / "source"), str(root / "converted"),
            ]

            subprocess.run(command, check=True, capture_output=True)
            converted = destination.read_text(encoding="utf-8")
            self.assertIn("NaN", converted)
            self.assertEqual(source.read_text(encoding="utf-8"), original)
            rerun = subprocess.run(command, check=False, capture_output=True, text=True)
            self.assertNotEqual(rerun.returncode, 0)
            self.assertIn("Output already exists", rerun.stderr)
            self.assertEqual(destination.read_text(encoding="utf-8"), converted)
