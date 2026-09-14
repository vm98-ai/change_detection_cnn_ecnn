# Rotation-Equivariant SAR Flood Change Detection

Does baking exact rotation symmetry into a CNN's architecture actually make it more robust to rotated inputs than a plain CNN of matched capacity — on **real** Sentinel-1 SAR flood-mapping data, not synthetic benchmarks?

This project trains two Siamese change-detection networks with matched parameter counts on the [S1GFloods](https://github.com/) benchmark (Sentinel-1 SAR pre/post-flood image pairs) and measures how each holds up when the input scene is rotated at test time, despite both models only ever seeing upright (0°) training patches.

- **BaselineCNN** — ordinary `nn.Conv2d` layers, no rotation symmetry.
- **EquivariantCNN** — [e2cnn](https://github.com/QUVA-Lab/e2cnn) steerable convolutions built on the C4 group, giving it an *exact, mathematically guaranteed* 90°/180°/270° rotation equivariance.

## Results

Both models trained on upright-only patches, evaluated on 805 held-out, never-seen-during-training test scenes at 11 rotation angles:

| Angle | Baseline IoU | Baseline F1 | Equivariant IoU | Equivariant F1 |
|------:|-------------:|------------:|-----------------:|----------------:|
| 0°    | 0.6647 | 0.7728 | **0.6788** | **0.7855** |
| 15°   | 0.4952 | 0.6376 | **0.5324** | **0.6724** |
| 30°   | 0.4271 | 0.5729 | **0.4545** | **0.6007** |
| 45°   | 0.4092 | 0.5549 | **0.4342** | **0.5807** |
| 60°   | 0.4264 | 0.5721 | **0.4539** | **0.6000** |
| 75°   | 0.4931 | 0.6357 | **0.5317** | **0.6718** |
| 90°   | 0.6619 | 0.7709 | **0.6788** | **0.7855** |
| 135°  | 0.4082 | 0.5540 | **0.4342** | **0.5807** |
| 180°  | 0.6647 | 0.7728 | **0.6788** | **0.7855** |
| 225°  | 0.4091 | 0.5549 | **0.4342** | **0.5807** |
| 270°  | 0.6620 | 0.7709 | **0.6788** | **0.7855** |

**Summary**
- Mean IoU drop across all non-zero rotation angles: **baseline 0.1591 vs. equivariant 0.1477**.
- At exact 90° multiples (0/90/180/270), the equivariant model's IoU is **numerically constant** (std = 0.0000), matching its mathematical guarantee exactly. The baseline drifts slightly (std = 0.0014) purely from boundary/padding effects on non-square real scenes.
- The equivariant model outperforms the baseline at **every single tested angle**, including the training orientation (0°) — not just at rotated angles, which is the stronger and less commonly reported result.

Full sweep: `outputs/rotation_robustness.json`. Qualitative comparison figure: `python visualize_comparison.py`.

## A bug we caught and fixed during review

An earlier version of `EquivariantCNN`'s fusion head used a `kernel_size=1` convolution to combine the pre-/post-flood branches, while `BaselineCNN`'s equivalent head layer used `kernel_size=3`. This silently gave the equivariant model **less receptive field exactly at the step that matters most** (comparing the two SAR images) — an unfair capacity handicap unrelated to rotation equivariance, and invisible to the equivariance unit test (a 1×1 conv is still perfectly equivariant). Before the fix, the equivariant model underperformed the baseline at *every* angle, including 0°, which pointed to an architecture bug rather than an equivariance-vs-capacity tradeoff. After matching the head kernel size (and re-tuning field widths to restore parameter parity — 32,769 vs. 32,228 params), the equivariant model wins outright, as shown above.

This is a useful case study in how a fair "matched capacity" comparison for equivariant architectures needs to be checked at every layer, not just totalled at the end — parameter-count parity doesn't guarantee receptive-field parity.

## Dataset

[S1GFloods]([https://github.com/](https://github.com/Tamer-Saleh/S1GFlood-Detection)) — real Sentinel-1 SAR flood-mapping change-detection pairs.

```
<DATA_ROOT>/train/{A,B,GT}/*.png   pre-flood / post-flood / binary mask
<DATA_ROOT>/val/{A,B,GT}/*.png
<DATA_ROOT>/test/{A,B,GT}/*.png
```

`prepare_data.py` extracts, splits, and validates the full dataset (~5,360 scenes) from a flat download into this layout. `data_utils.py` auto-switches between exhaustive patch extraction (small datasets) and on-the-fly random-crop sampling (full-scale) so memory stays bounded regardless of dataset size.

## Repo structure

```
models.py               BaselineCNN and EquivariantCNN (C4-equivariant, e2cnn)
data_utils.py            Dataset classes, patch extraction, rotation utilities
prepare_data.py           One-time dataset extraction / splitting / validation
train.py                  Trains both models (matched hyperparameters, pos-weighted BCE loss)
evaluate_rotation.py       11-angle rotation robustness sweep on held-out test scenes
visualize_comparison.py    Qualitative 0°-vs-90° prediction comparison figure
```

## Running it

```bash
pip install torch torchvision e2cnn tqdm numpy pillow

# One-time: point at your downloaded S1GFloods data and stage it
python3 prepare_data.py --source /path/to/S1GFloods --dest data_full

# Train both models
python3 train.py both --data-root data_full --num-workers 4

# Rotation robustness sweep (the headline result)
python3 evaluate_rotation.py --data-root data_full

# Qualitative figure
python3 visualize_comparison.py --data-root data_full
```

## Requirements

- Python 3.9–3.12
- PyTorch, torchvision
- [e2cnn](https://github.com/QUVA-Lab/e2cnn)
- numpy, Pillow, tqdm, matplotlib

## Limitations / future work

- Only exact 90°/180°/270° rotations are architecturally guaranteed; arbitrary-angle robustness (15°–75°) is empirical, not provable, though it's still consistently better than the baseline here.
- SAR backscatter has physically anisotropic properties (look angle, layover/shadow) that a C4-symmetric model cannot represent by design — the gains here suggest the benefits of exact equivariance outweigh that cost on this dataset, but that tradeoff may not generalize to every SAR domain.
- A group with finer angular resolution (e.g. C8 or a fully continuous group) could be tried for smoother robustness across the non-90°-multiple angles.
