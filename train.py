"""
Training script for both the baseline CNN and the C4-equivariant CNN on the
S1GFloods SAR change-detection data.
"""
import locale
locale.setlocale(locale.LC_NUMERIC, "C")
import argparse
import copy
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import SARChangeDataset, list_scene_ids, _load_mask
from models import BaselineCNN, EquivariantCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_pos_weight(root, split, max_sample_scenes=300, seed=0):
    scene_ids = list_scene_ids(root, split)
    rng = random.Random(seed)
    if len(scene_ids) > max_sample_scenes:
        scene_ids = rng.sample(scene_ids, max_sample_scenes)

    total_pos = 0.0
    total_pix = 0.0
    for sid in tqdm(scene_ids, desc="Computing Pos Weight", leave=False):
        gt = _load_mask(os.path.join(root, split, "GT", sid))
        total_pos += gt.sum()
        total_pix += gt.size
    pos_frac = total_pos / total_pix
    pos_weight = (1 - pos_frac) / max(pos_frac, 1e-6)
    return torch.tensor(pos_frac), torch.tensor(pos_weight)


def iou_f1(logits, gt, thresh=0.5):
    probs = torch.sigmoid(logits)
    pred = (probs > thresh).float()
    tp = (pred * gt).sum().item()
    fp = (pred * (1 - gt)).sum().item()
    fn = ((1 - pred) * gt).sum().item()
    iou = tp / (tp + fp + fn + 1e-8)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    return iou, f1


def evaluate(model, loader, criterion, tag="Eval"):
    model.eval()
    losses, ious, f1s = [], [], []
    
    val_pbar = tqdm(loader, desc=tag, leave=False)
    with torch.no_grad():
        for a, b, gt in val_pbar:
            a, b, gt = a.to(DEVICE), b.to(DEVICE), gt.to(DEVICE)
            logits = model(a, b)
            loss = criterion(logits, gt)
            losses.append(loss.item())
            iou, f1 = iou_f1(logits, gt)
            ious.append(iou)
            f1s.append(f1)
            
            val_pbar.set_postfix(iou=f"{iou:.4f}", loss=f"{loss.item():.4f}")
            
    return float(np.mean(losses)), float(np.mean(ious)), float(np.mean(f1s))


def train_one_model(model, train_ds, val_ds, pos_weight, epochs, lr=1e-3,
                    batch_size=16, num_workers=0, tag="model",
                    log_every=5, out_dir="outputs"):
    model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)

    best_val_iou = -1.0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        
        train_pbar = tqdm(train_loader, desc=f"[{tag}] Ep {epoch}/{epochs} Train", leave=False)
        for a, b, gt in train_pbar:
            a, b, gt = a.to(DEVICE), b.to(DEVICE), gt.to(DEVICE)
            optimizer.zero_grad()
            logits = model(a, b)
            loss = criterion(logits, gt)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        scheduler.step()

        val_loss, val_iou, val_f1 = evaluate(model, val_loader, criterion, tag=f"[{tag}] Ep {epoch}/{epochs} Val")
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_loss,
            "val_iou": val_iou,
            "val_f1": val_f1,
        })

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, os.path.join(out_dir, f"{tag}_best.pt"))

        if epoch % log_every == 0 or epoch == 1:
            print(f"[{tag}] epoch {epoch:3d}  train_loss={np.mean(train_losses):.4f}  "
                  f"val_loss={val_loss:.4f}  val_iou={val_iou:.4f}  val_f1={val_f1:.4f}",
                  flush=True)
            with open(os.path.join(out_dir, f"{tag}_history.json"), "w") as f:
                json.dump(history, f, indent=2)

    model.load_state_dict(best_state)
    print(f"[{tag}] best val IoU = {best_val_iou:.4f}", flush=True)
    return model, history


def build_arg_parser():
    p = argparse.ArgumentParser(description="Train SAR change detection models.")
    p.add_argument("--which", nargs="?", default="both", choices=["baseline", "equivariant", "both"])
    p.add_argument("--data-root", default=os.environ.get("SAR_CD_DATA_ROOT", "data"),
                   help="Folder containing train/val/test subfolders with A/B/GT images. "
                        "Can also be set via the SAR_CD_DATA_ROOT environment variable.")
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--stride", type=int, default=32,
                   help="Only used in exhaustive (small-dataset) mode.")
    p.add_argument("--epochs-baseline", type=int, default=60)
    p.add_argument("--epochs-equivariant", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader worker processes. Increase this once the "
                        "dataset is large enough that PNG decoding, not compute, "
                        "is the bottleneck (e.g. --num-workers 4).")
    p.add_argument("--steps-per-epoch", type=int, default=None,
                   help="Only meaningful in random-crop mode (full dataset). "
                        "Defaults to one crop per training scene per epoch.")
    p.add_argument("--val-crops-per-scene", type=int, default=4,
                   help="Only meaningful in random-crop mode. Validation crops "
                        "are fixed once at startup so val metrics are comparable "
                        "across epochs.")
    p.add_argument("--max-pos-weight-scenes", type=int, default=300,
                   help="Cap on how many scenes to sample from disk when "
                        "estimating the flood/no-flood class balance.")
    p.add_argument("--out-dir", default="outputs")
    return p


def main():
    args = build_arg_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    set_seed(0)

    train_ds = SARChangeDataset(
        args.data_root, "train", patch_size=args.patch_size, stride=args.stride,
        augment=True, seed=0, epoch_size=args.steps_per_epoch,
    )
    n_val_scenes = len(list_scene_ids(args.data_root, "val"))
    val_epoch_size = (args.val_crops_per_scene * n_val_scenes
                      if train_ds.mode == "random" else None)
    val_ds = SARChangeDataset(
        args.data_root, "val", patch_size=args.patch_size, stride=args.stride,
        augment=False, seed=1, epoch_size=val_epoch_size,
    )
    # In random-crop mode, freeze the validation crop coordinates by
    # pre-materializing them once, so val_iou is comparable across epochs
    # rather than measured against a different random sample every time.
    if val_ds.mode == "random":
        frozen = [val_ds[i] for i in tqdm(range(len(val_ds)), desc="Caching val crops")]
        val_ds = frozen  # plain list is a valid Dataset-like for DataLoader via TensorDataset-style access
        from torch.utils.data import TensorDataset
        a_stack = torch.stack([x[0] for x in frozen])
        b_stack = torch.stack([x[1] for x in frozen])
        gt_stack = torch.stack([x[2] for x in frozen])
        val_ds = TensorDataset(a_stack, b_stack, gt_stack)

    print(f"data root: {args.data_root}", flush=True)
    print(f"train mode: {train_ds.mode}   train dataset size (per epoch): {len(train_ds)}", flush=True)
    print(f"val dataset size: {len(val_ds)}", flush=True)

    pos_frac, pos_weight = compute_pos_weight(
        args.data_root, "train", max_sample_scenes=args.max_pos_weight_scenes)
    print(f"train flood pixel fraction (sampled): {pos_frac.item():.4f}   "
          f"BCE pos_weight: {pos_weight.item():.4f}", flush=True)

    if args.which in ("baseline", "both"):
        set_seed(1)
        baseline = BaselineCNN(base_ch=16)
        baseline, _ = train_one_model(
            baseline, train_ds, val_ds, pos_weight, epochs=args.epochs_baseline,
            batch_size=args.batch_size, num_workers=args.num_workers,
            tag="baseline", out_dir=args.out_dir,
        )
        torch.save(baseline.state_dict(), os.path.join(args.out_dir, "baseline.pt"))

    if args.which in ("equivariant", "both"):
        set_seed(1)
        equiv = EquivariantCNN(fields=(13, 19, 19))
        equiv, _ = train_one_model(
            equiv, train_ds, val_ds, pos_weight, epochs=args.epochs_equivariant,
            batch_size=args.batch_size, num_workers=args.num_workers,
            tag="equivariant", out_dir=args.out_dir,
        )
        torch.save(equiv.state_dict(), os.path.join(args.out_dir, "equivariant.pt"))

    print("DONE", flush=True)


if __name__ == "__main__":
    main()