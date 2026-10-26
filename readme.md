# MinuNet

MinuNet is a 3D medical image segmentation network built on nnU-Net v2. It uses depthwise-separable inverted bottleneck blocks and a strided stem to reduce the cost of processing volumetric images. Its planner chooses target spacing and stem strides together to account for image anisotropy.

This repository contains the implementation used for *MinuNet: Efficiency-driven design for volumetric medical image segmentation*, by Mehdi Rejali, Maryam Jabbari, Mohammad Moein Firouzi, Mohammad Reza Ketabi Delshad, and Fatemeh Shakeri. Author order is provisional.

## What is included

- The MinuNet architecture, with static and dynamic convolution variants.
- Spacing and stem planning, plus presets for width, spatial resolution, and block ablations.
- An AdamW trainer with linear warm-up and cosine decay.
- Utilities for scheduling training across GPUs, measuring CUDA execution cost, and reporting results.
- The nnU-Net preprocessing, training, validation, and inference pipeline on which these experiments are built.

The Python package remains `nnunetv2`, and the data paths use nnU-Net's environment variables. The installed distribution is named `minunet`. Use a separate environment from upstream nnU-Net because both provide the same Python package.

## Installation

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). The dependency configuration selects CUDA 13.0 PyTorch wheels; training and the timing utility are intended for NVIDIA GPUs with a compatible driver.

```bash
git clone https://github.com/Darchin/minunet.git
cd minunet
uv python install 3.12
uv python pin 3.12
uv sync --extra dev
```

Set the three data directories before running commands. For example, in a POSIX shell:

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
```

In PowerShell, use `$Env:nnUNet_raw = "C:/path/to/nnUNet_raw"`, and likewise for the other two variables. See [dataset format](documentation/dataset_format.md) and [path setup](documentation/setting_up_paths.md) for the expected layout. Each dataset belongs in a directory such as `nnUNet_raw/Dataset001_MyDataset`.

## Plan, preprocess, and train

The commands below use dataset ID `1` as an example. Replace it with your dataset ID.

First extract the fingerprint with the custom extractor, then generate MinuNet plans:

```bash
uv run extract_fingerprint -d 1 -fpe PercentileFingerprintExtractor --verify_dataset_integrity
uv run plan_experiment -d 1 -pl MinuNetPlanner
uv run preprocess -d 1 -plans_name MinuNetPlans -c 2x 3x 4x
```

If the dataset already has a fingerprint from the standard nnU-Net extractor, rerun the extraction command with `--clean`. MinuNet planning needs the `spacing_percentiles` field added by `PercentileFingerprintExtractor`.

Plans are written to `nnUNet_preprocessed/DatasetXXX_Name/MinuNetPlans.json`. They include shared preprocessing presets (`base`, `2x`, `3x`, `4x`), architecture presets (`MN`, `MN-2x`, `MN-3x`, `MN-4x`), and trainable size presets such as `MN-4x-M`. The shared presets are intentionally incomplete. Before training a custom child configuration, set its `patch_size_multiplier` and architecture `channels`; the planner's size presets already supply these values.

Train one fold with:

```bash
uv run train 1 MN-4x-M 0 -p MinuNetPlans -tr nnUNetTrainerAdamW
```

Use folds `0` through `4` for five-fold cross-validation. Check patch size, batch size, and memory use for your dataset before launching a full run.

For the paper's cross-dataset experiments, use `KiTSPlanner`, `AMOSPlanner`, or `BraTSPlanner` in place of `MinuNetPlanner` during planning. These inherit from `BenchmarkPlanner` and set the dataset-specific patch size multipliers. They write `MinuNetPlans` by default and include these configurations:

| Paper configuration | Plan configuration |
| --- | --- |
| Static base | `MN-4x-M_Static_EP1000` |
| Wider static model | `MN-4x-L_Static_EP1000` |
| Finer spatial processing | `MN-2x-M_Static_EP1000` |
| Dynamic convolution | `MN-4x-M_Dynamic_EP1000` |

Preprocess with the chosen planner's plans before training. Training settings can be overridden in a configuration's `trainer` dictionary; see the [AdamW trainer notes](documentation/adamw_training.md).

Useful training options:

- `--disable-tta` disables test-time augmentation during post-training validation.
- `--disable_train_val` skips per-epoch validation while preserving post-training validation.
- `--ckpt-interval 50` controls how often `checkpoint_last.pth` is overwritten. The same checkpoint is saved again at the end of training.

## Inference and timing

Predict from a trained model, for example using fold `0`:

```bash
uv run predict -i /path/to/imagesTs -o /path/to/predictions -d 1 -c MN-4x-M -p MinuNetPlans -tr nnUNetTrainerAdamW -f 0 -chk checkpoint_last.pth
```

Input images follow the [nnU-Net inference format](documentation/dataset_format_inference.md).

Measure synthetic training-step or forward-pass cost with:

```bash
uv run benchmark -d 1 -p MinuNetPlans -c MN-4x-M
uv run benchmark -d 1 -p MinuNetPlans -c MN-4x-M --inference
```

These measurements use dummy tensors with the configuration's patch size and batch size. They measure model execution, not full-volume inference or end-to-end training time. See the [benchmark guide](documentation/how-to/run-benchmarks.md) for details. `uv run batch_train --help` describes the GPU scheduler; `uv run report_results --help` describes result reporting.

## Existing research runs

The architecture is `nnunetv2.network_architecture.minunet.MinuNet`, the planner is `MinuNetPlanner`, and current plans use `MinuNetPlans`. The `MN-*` configuration names and model parameter keys are unchanged.

To update earlier research runs, use the [one-time migration utility](documentation/how-to/migrate-minunet-results.md). It backs up and verifies the selected artifacts before updating their paths, plans, and checkpoint metadata. Archived results are excluded.

## Development and documentation

```bash
uv sync --extra dev
uv run pytest nnunetv2/tests
```

The [documentation directory](documentation/README.md) also contains inherited nnU-Net guides. Use the commands and MinuNet planning workflow in this README for this fork; some inherited guides describe upstream defaults and `nnUNetv2_*` command names.

## Attribution and citation

MinuNet is a research fork of [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet), developed by Fabian Isensee and colleagues at the Applied Computer Vision Lab of Helmholtz Imaging and the Division of Medical Image Computing at the German Cancer Research Center (DKFZ). Much of this repository's framework, utilities, and documentation comes from that project. The MinuNet work adds the architecture, planners, and experimental tooling described above. Original copyright notices are retained.

If you use MinuNet in research, please cite the MinuNet manuscript and the nnU-Net paper. The manuscript's current citation is:

```bibtex
@misc{rejali2026minunet,
  title = {MinuNet: Efficiency-driven design for volumetric medical image segmentation},
  author = {Rejali, Mehdi and Jabbari, Maryam and Firouzi, Mohammad Moein and Ketabi Delshad, Mohammad Reza and Shakeri, Fatemeh},
  year = {2026},
  howpublished = {Research manuscript},
  url = {https://github.com/Darchin/minunet}
}
```

The underlying framework is described in:

> Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., and Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*, 18, 203-211. [doi:10.1038/s41592-020-01008-z](https://doi.org/10.1038/s41592-020-01008-z).

For the residual encoder baselines, see [nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation](https://arxiv.org/abs/2404.09556).

## License

This repository is distributed under the [Apache License 2.0](LICENSE), inherited from nnU-Net.
