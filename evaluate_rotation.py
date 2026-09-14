"""
Both models were trained on upright (0-degree) patches only. Here we take
the held-out TEST scenes (never touched during training or model
selection) and evaluate both trained models at:

  0, 15, 30, 45, 60, 75, 90, 135, 180, 225, 270 degrees

For the exact 90-degree multiples we rotate losslessly (torch.rot90) so
there is no interpolation artifact to confound the comparison. For the
non-multiple-of-90 angles we use bilinear interpolation for the images and
nearest-neighbor for the ground truth mask, which introduces some
resampling blur; this is noted in the results rather than hidden.

Hypothesis: the C4-equivariant model's IoU should stay essentially flat
across all 90-degree multiples (that's an exact mathematical guarantee of
the architecture) and should degrade much less than the baseline at
in-between angles too, since its filters were built from a rotation-
symmetric basis rather than learned from scratch in one fixed orientation.

"""
import argparse
import json
import os

import numpy as np
import torch
import torchvision.transforms.functional as TF

from data_utils import FullSceneDataset
from models import BaselineCNN, EquivariantCNN
from train import iou_f1

DEVICE = torch.device("cpu")
ANGLES = [0, 15, 30, 45, 60, 75, 90, 135, 180, 225, 270]


def rotate_scene(a, b, gt, angle):
    if angle % 90 == 0:
        k = (angle // 90) % 4
        a2 = torch.rot90(a, k, dims=(-2, -1))
        b2 = torch.rot90(b, k, dims=(-2, -1))
        gt2 = torch.rot90(gt, k, dims=(-2, -1))
    else:
        a2 = TF.rotate(a, angle, interpolation=TF.InterpolationMode.BILINEAR)
        b2 = TF.rotate(b, angle, interpolation=TF.InterpolationMode.BILINEAR)
        gt2 = TF.rotate(gt, angle, interpolation=TF.InterpolationMode.NEAREST)
    return a2, b2, gt2


def evaluate_at_angle(model, dataset, angle):
    model.eval()
    ious, f1s = [], []
    with torch.no_grad():
        for a, b, gt, sid in dataset:
            a = a.unsqueeze(0)
            b = b.unsqueeze(0)
            gt = gt.unsqueeze(0)
            a, b, gt = rotate_scene(a, b, gt, angle)
            logits = model(a, b)
            iou, f1 = iou_f1(logits, gt)
            ious.append(iou)
            f1s.append(f1)
    return float(np.mean(ious)), float(np.mean(f1s))


def build_arg_parser():
    p = argparse.ArgumentParser(description="Rotation robustness evaluation.")
    p.add_argument("--data-root", default=os.environ.get("SAR_CD_DATA_ROOT", "data"),
                    help="Same folder passed to train.py. Can also be set via "
                         "the SAR_CD_DATA_ROOT environment variable.")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--max-test-scenes", type=int, default=None,
                    help="Evaluate a random subset of this many test scenes "
                         "instead of the full split. Useful once the test "
                         "split has hundreds of scenes; leave unset to use "
                         "every test scene.")
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_arg_parser().parse_args()

    test_ds = FullSceneDataset(args.data_root, "test",
                                max_scenes=args.max_test_scenes, seed=args.seed)
    print(f"test scenes evaluated: {len(test_ds)}")

    baseline = BaselineCNN(base_ch=16)
    baseline.load_state_dict(torch.load(
        os.path.join(args.out_dir, "baseline_best.pt"), map_location=DEVICE))
    baseline.to(DEVICE)

    equiv = EquivariantCNN(fields=(13, 19, 19))
    equiv.load_state_dict(torch.load(
        os.path.join(args.out_dir, "equivariant_best.pt"), map_location=DEVICE))
    equiv.to(DEVICE)

    results = {"baseline": {}, "equivariant": {}}
    for angle in ANGLES:
        b_iou, b_f1 = evaluate_at_angle(baseline, test_ds, angle)
        e_iou, e_f1 = evaluate_at_angle(equiv, test_ds, angle)
        results["baseline"][angle] = {"iou": b_iou, "f1": b_f1}
        results["equivariant"][angle] = {"iou": e_iou, "f1": e_f1}
        print(f"angle={angle:4d}   baseline IoU={b_iou:.4f} F1={b_f1:.4f}   "
              f"equivariant IoU={e_iou:.4f} F1={e_f1:.4f}", flush=True)

    with open(os.path.join(args.out_dir, "rotation_robustness.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Summary: drop relative to the 0-degree score, averaged over all
    # non-zero angles.
    base_0 = results["baseline"][0]["iou"]
    equiv_0 = results["equivariant"][0]["iou"]
    base_drops = [base_0 - results["baseline"][a]["iou"] for a in ANGLES if a != 0]
    equiv_drops = [equiv_0 - results["equivariant"][a]["iou"] for a in ANGLES if a != 0]
    print("\n=== Summary ===")
    print(f"Baseline:     IoU@0deg={base_0:.4f}   mean IoU drop over rotations={np.mean(base_drops):.4f}")
    print(f"Equivariant:  IoU@0deg={equiv_0:.4f}   mean IoU drop over rotations={np.mean(equiv_drops):.4f}")

    # 90-degree multiples specifically (exact equivariance regime)
    mult90 = [a for a in ANGLES if a % 90 == 0]
    base_90 = [results["baseline"][a]["iou"] for a in mult90]
    equiv_90 = [results["equivariant"][a]["iou"] for a in mult90]
    print(f"\nAt exact 90-degree multiples {mult90}:")
    print(f"  Baseline IoU:    {[round(v,4) for v in base_90]}  (std={np.std(base_90):.4f})")
    print(f"  Equivariant IoU: {[round(v,4) for v in equiv_90]}  (std={np.std(equiv_90):.4f})")


if __name__ == "__main__":
    main()