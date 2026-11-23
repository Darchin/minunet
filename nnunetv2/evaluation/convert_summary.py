"""Convert saved Dice and IoU scores to nnU-Net's empty-mask convention."""

import argparse
import json
from math import isnan, nan
from pathlib import Path
from statistics import fmean


OVERLAP_METRICS = ("Dice", "IoU")


def convert_empty_pairs(summary: dict) -> None:
    """Update an already loaded nnU-Net summary in place."""
    cases = summary["metric_per_case"]
    means = summary["mean"]
    foreground = summary["foreground_mean"]
    foreground_targets = [target for target in means if target != "0"]
    if not cases or not foreground_targets:
        raise ValueError("Expected per-case metrics and at least one foreground target")

    for case in cases:
        case_metrics = case["metrics"]
        if case_metrics.keys() != means.keys():
            raise ValueError("Per-case and aggregate targets differ")
        for metrics in case_metrics.values():
            if metrics["TP"] == metrics["FP"] == metrics["FN"] == 0:
                for metric in OVERLAP_METRICS:
                    metrics[metric] = nan

    for target, target_means in means.items():
        for metric in OVERLAP_METRICS:
            scores = [case["metrics"][target][metric] for case in cases]
            valid_scores = [score for score in scores if not isnan(score)]
            target_means[metric] = fmean(valid_scores) if valid_scores else nan

    # Upstream uses an ordinary mean across foreground targets. An entirely
    # empty target therefore makes the foreground mean NaN for that summary.
    for metric in OVERLAP_METRICS:
        foreground[metric] = fmean(means[target][metric] for target in foreground_targets)


def convert_file(source: Path, destination: Path) -> None:
    try:
        with source.open(encoding="utf-8") as stream:
            summary = json.load(stream)
        convert_empty_pairs(summary)
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise ValueError(f"{source}: invalid nnU-Net summary ({error})") from error

    # Like nnU-Net, emit JSON with the literal NaN for undefined scores.
    payload = json.dumps(summary, indent=4, allow_nan=True) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    stream = destination.open("x", encoding="utf-8")
    try:
        with stream:
            stream.write(payload)
    except OSError:
        destination.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="A summary JSON file or a directory to scan for summary.json files")
    parser.add_argument("destination", type=Path, help="A new JSON file or a directory for the converted copies")
    args = parser.parse_args()

    if args.source.is_file():
        conversions = [(args.source, args.destination)]
    elif args.source.is_dir():
        conversions = [
            (source, args.destination / source.relative_to(args.source))
            for source in sorted(args.source.rglob("summary.json"))
            if source.is_file()
        ]
        if not conversions:
            parser.error(f"No summary.json files found in {args.source}")
    else:
        parser.error(f"Source does not exist: {args.source}")

    try:
        for _, destination in conversions:
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"Output already exists: {destination}")
        for source, destination in conversions:
            convert_file(source, destination)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")

    print(f"Converted {len(conversions)} summary file(s) to {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
