"""
Qualitative comparison figure: pick one real test scene, run both trained
models on it upright and on a 90-degree-rotated version, un-rotate the
90-degree predictions back for display, and show all four prediction maps
next to the ground truth. Also prints the exact pixel-level difference
between "predict upright" and "predict rotated then un-rotate", which is
the cleanest possible demonstration of what equivariance buys you.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data_utils import FullSceneDataset
from models import BaselineCNN, EquivariantCNN
from evaluate_rotation import rotate_scene

DEVICE = torch.device("cpu")


def build_arg_parser():
    p = argparse.ArgumentParser(description="Qualitative rotation comparison figure.")
    p.add_argument("--data-root", default=os.environ.get("SAR_CD_DATA_ROOT", "data"))
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--scene-index", type=int, default=0,
                    help="Index into the test split (after any --max-test-scenes "
                         "subsampling used elsewhere, this script always looks at "
                         "the full test split by index).")
    return p


def main():
    args = build_arg_parser().parse_args()

    test_ds = FullSceneDataset(args.data_root, "test")
    if args.scene_index >= len(test_ds):
        raise IndexError(
            f"--scene-index {args.scene_index} out of range, "
            f"test split has {len(test_ds)} scenes."
        )
    a, b, gt, sid = test_ds[args.scene_index]
    print("scene:", sid)

    baseline = BaselineCNN(base_ch=16)
    baseline.load_state_dict(torch.load(
        os.path.join(args.out_dir, "baseline_best.pt"), map_location=DEVICE))
    baseline.eval()

    equiv = EquivariantCNN(fields=(13, 19, 19))
    equiv.load_state_dict(torch.load(
        os.path.join(args.out_dir, "equivariant_best.pt"), map_location=DEVICE))
    equiv.eval()

    a0, b0, gt0 = a.unsqueeze(0), b.unsqueeze(0), gt.unsqueeze(0)
    a90, b90, gt90 = rotate_scene(a0, b0, gt0, 90)

    with torch.no_grad():
        pred_base_0 = torch.sigmoid(baseline(a0, b0))[0, 0]
        pred_base_90 = torch.sigmoid(baseline(a90, b90))[0, 0]
        pred_equiv_0 = torch.sigmoid(equiv(a0, b0))[0, 0]
        pred_equiv_90 = torch.sigmoid(equiv(a90, b90))[0, 0]

    # Un-rotate the 90-degree predictions (-90 = 3*90) so all four prediction
    pred_base_90_unrot = torch.rot90(pred_base_90, k=3, dims=(-2, -1))
    pred_equiv_90_unrot = torch.rot90(pred_equiv_90, k=3, dims=(-2, -1))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    axes[0, 0].imshow(a[0], cmap="gray")
    axes[0, 0].set_title("Pre-flood SAR (A)")
    axes[0, 1].imshow(b[0], cmap="gray")
    axes[0, 1].set_title("Post-flood SAR (B)")
    axes[0, 2].imshow(gt[0], cmap="gray")
    axes[0, 2].set_title(f"Ground truth flood mask\n({sid})")
    axes[0, 3].axis("off")

    axes[1, 0].imshow(pred_base_0, cmap="viridis", vmin=0, vmax=1)
    axes[1, 0].set_title("Baseline pred, trained orientation (0 deg)")
    axes[1, 1].imshow(pred_base_90_unrot, cmap="viridis", vmin=0, vmax=1)
    axes[1, 1].set_title("Baseline pred, input rotated 90 deg\n(then unrotated for display)")
    axes[1, 2].imshow(pred_equiv_0, cmap="viridis", vmin=0, vmax=1)
    axes[1, 2].set_title("Equivariant pred, 0 deg")
    axes[1, 3].imshow(pred_equiv_90_unrot, cmap="viridis", vmin=0, vmax=1)
    axes[1, 3].set_title("Equivariant pred, input rotated 90 deg\n(then unrotated for display)")

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, "rotation_comparison.png")
    plt.savefig(out_path, dpi=130)
    print(f"saved {out_path}")

    diff_base = (pred_base_0 - pred_base_90_unrot).abs()
    diff_equiv = (pred_equiv_0 - pred_equiv_90_unrot).abs()
    print(f"Baseline: mean|pred(0) - pred(90, unrotated)| = {diff_base.mean().item():.6f}, "
          f"max = {diff_base.max().item():.6f}")
    print(f"Equivariant: mean|pred(0) - pred(90, unrotated)| = {diff_equiv.mean().item():.6f}, "
          f"max = {diff_equiv.max().item():.6f}")


if __name__ == "__main__":
    main()