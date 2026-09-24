<div align="center">

# Cryomni：Cryo-EM Foundation Model Using Self-Supervised Learning


[Overview](#overview) · [Installation](#installation) · [Quick Start](#quick-start) · [Usage](#usage) · [Citation](#citation)

</div>

## Overview

Cryomni uses a 3D Swin Transformer masked autoencoder to reconstruct masked regions of cryo-electron microscopy (cryo-EM) density maps. This repository provides the model architecture, checkpoint loading utilities, MRC preprocessing, and command-line inference.

The inference workflow consists of:

1. Standardizing map axes and resampling the density map to a 1 Å voxel grid.
2. Selecting the map region using a contour threshold, normalizing densities, and extracting 3D boxes.
3. Reconstructing masked regions with the pretrained model.
4. Assembling the visible input and reconstructed regions into output MRC maps.

**Scope:** This release contains inference code. Training losses and training/validation entry points have been removed. Use `model.forward_pred()` for prediction; `model(...)` is not a supported entry point. The model parameter layers are retained for checkpoint compatibility. Distributed pretraining scripts and a complete pipeline for reproducing paper experiments are not included.

## Repository Structure

```text
Cryomni/
├── checkpoint/
│   └── Cryomni.pt              # Pretrained weights
├── configs/
│   └── config.json             # Model configuration
├── cryomni/
│   ├── __init__.py
│   ├── checkpoint.py           # Configuration and checkpoint loading
│   ├── protein_swin_mae3d.py   # 3D Swin masked autoencoder
│   └── model/                 # Neural network building blocks
├── data_processing/           # Map preprocessing and box extraction
├── example/
│   └── 8796/
│       ├── emd_8796.mrc        # Example input density map
│       ├── 5wcb.pdb
│       └── 5wcb.fasta
├── scripts/
│   ├── infer.py               # Inference entry point
│   └── test.sh                # Shell wrapper for inference
├── requirements.txt
└── README.md
```

## Installation

Use Python 3.10 or newer. Install compatible `torch` and `torchvision` builds for your CPU or CUDA environment, then install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

Dependencies include PyTorch, torchvision, NumPy, mrcfile, and tqdm. Dependency versions are currently unpinned; a validated environment specification is not yet provided.

Check that the inference entry point is available:

```bash
python scripts/infer.py --help
```

All commands below assume that your current working directory is the repository root.

## Pretrained Weights

Place the pretrained checkpoint at:

```text
checkpoint/Cryomni.pt
```

The pretrained checkpoint is available on [Hugging Face](https://huggingface.co/Jeffiry/Cryomni_pretrain/tree/main).

The default model configuration is [`configs/config.json`](configs/config.json). Use the configuration matching your checkpoint. For weights stored elsewhere, specify `--checkpoint /path/to/Cryomni.pt`.

## Quick Start

Run inference on the example map in `example/8796`:

```bash
python scripts/infer.py example/8796/emd_8796.mrc \
  --checkpoint checkpoint/Cryomni.pt \
  --config configs/config.json \
  --output-dir "$PWD/outputs/8796" \
  --device cuda:0 \
  --seed 0
```

Only the MRC file is required; the PDB and FASTA files are not used by the inference script. To run the model on CPU, replace `--device cuda:0` with `--device cpu`.

This example omits `--contour` and therefore uses the implementation's fallback value of `-1.0`. This is not a validated contour level for EMD-8796. If you know the appropriate density threshold for your map, supply it explicitly using `--contour`.

### Outputs

```text
outputs/8796/
├── mask_emd_8796.mrc
├── recon_emd_8796.mrc
└── preprocessed/
    └── emd_8796/               # Normalized map and preprocessing artifacts
```

| File | Description |
| --- | --- |
| `mask_<map_id>.mrc` | Visible, unmasked input density assembled from processed boxes. |
| `recon_<map_id>.mrc` | Visible input density combined with the model predictions for masked regions. |

`<map_id>` is the input filename without its extension. Outputs use the preprocessed map grid and normalized density scale; they may differ from the original input in dimensions and density range. Unified, resized, and segmented intermediate maps are removed by default. Use `--keep-intermediate` to retain them.

## Usage

### Single map with a contour threshold

The value `0.5` below is illustrative. Replace it with a threshold appropriate for your input map.

```bash
python scripts/infer.py /path/to/input.mrc \
  --contour 0.5 \
  --output-dir "$PWD/outputs" \
  --device cuda:0
```

### Multiple maps

Provide a JSON file mapping each input filename stem to its contour value. For example, `contours.json` might contain:

```json
{
  "map_a": 0.42,
  "map_b": 0.38
}
```

These values are examples, not recommended thresholds for a particular dataset.

```bash
python scripts/infer.py /path/to/map_a.mrc /path/to/map_b.mrc \
  --contour-json contours.json \
  --output-dir "$PWD/outputs" \
  --device cuda:0
```

An explicit `--contour` overrides the JSON values for all inputs. Maps without a matching JSON entry use `-1.0`. Use distinct filename stems for inputs in the same output directory to avoid overwriting results.

### Command-line options

| Option | Default | Description |
| --- | --- | --- |
| `input_maps` | Required | One or more `.mrc` or `.map` input files. |
| `--checkpoint` | `checkpoint/Cryomni.pt` | Checkpoint file or a `save_pretrained` directory. |
| `--config` | `configs/config.json` | Model configuration for a checkpoint file. |
| `--output-dir` | `outputs` | Output directory. |
| `--contour` | Unset | Density threshold shared by all input maps. |
| `--contour-json` | Unset | JSON mapping input filename stems to thresholds. |
| `--device` | CUDA if available; otherwise CPU | Model inference device, such as `cuda:0` or `cpu`. |
| `--box-size` | `64` | Box edge length in voxels. |
| `--stride` | `32` | Step size for box extraction. |
| `--masking-prob` | `0.75` | Masking probability during inference. |
| `--seed` | `0` | Python and PyTorch random seed, including random masking. |
| `--keep-intermediate` | Disabled | Retain unified, resized, and segmented maps. |

The default checkpoint, configuration, and output paths resolve relative to the repository root. Explicit relative paths resolve relative to the current working directory.

Keep `--box-size 64` for the supplied configuration, whose model resolution is 64. Overlapping boxes are written in processing order; overlapping predictions are not averaged. A fixed seed controls Python and PyTorch randomness but does not guarantee identical results across hardware and software environments.

## Troubleshooting

- **Missing Python module:** Install `requirements.txt` using the same Python environment used to run inference.
- **Checkpoint not found:** Confirm that `checkpoint/Cryomni.pt` exists, or pass its path with `--checkpoint`.
- **CUDA out of memory:** Free GPU memory or try `--device cpu`. Memory requirements and runtime have not yet been benchmarked for this release.
- **No meaningful boxes found:** Check the input density map and contour threshold. Box selection excludes regions with insufficient density above the threshold.

## Citation

Paper metadata and a BibTeX citation will be added when available.

<!-- Before publication, add the verified paper title, authors, venue/year,
     paper or preprint URL, and BibTeX citation. -->

## License

A license has not yet been specified for this repository.

<!-- Before publication, add the chosen LICENSE file and clarify the terms
     for pretrained weights and example data. -->
