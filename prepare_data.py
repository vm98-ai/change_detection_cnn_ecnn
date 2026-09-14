"""
Run this once after you get the full dataset onto disk, before training.

Two ways to point it at your data:

1. You already have an extracted folder with A/, B/, Label/ subfolders:
       python3 prepare_data.py --source /path/to/S1GFloods --dest data_full

2. You have a .zip (e.g. downloaded from Google Drive and re-uploaded here):
       python3 prepare_data.py --source /path/to/S1GFloods.zip --dest data_full

Once this passes, point train.py at the validated folder:
    python3 train.py both --data-root data_full --num-workers 4
"""
import argparse
import os
import random
import shutil
import zipfile

import numpy as np

from data_utils import list_scene_ids, _load_mask


def find_data_root(base):
    current = base
    for _ in range(4):
        # Already formatted?
        if os.path.isdir(os.path.join(current, "train")):
            return current
        # Flat structure with Label?
        if all(os.path.isdir(os.path.join(current, d)) for d in ["A", "B", "Label"]):
            return current
        # Flat structure with GT?
        if all(os.path.isdir(os.path.join(current, d)) for d in ["A", "B", "GT"]):
            return current
            
        entries = [e for e in os.listdir(current)
                   if os.path.isdir(os.path.join(current, e))]
        if len(entries) == 1:
            current = os.path.join(current, entries[0])
        else:
            break
    return current


def extract_if_zip(source, work_dir):
    if os.path.isdir(source):
        return find_data_root(source)
    if zipfile.is_zipfile(source):
        extract_dir = os.path.join(work_dir, "_extracted_source")
        os.makedirs(extract_dir, exist_ok=True)
        print(f"Extracting {source} -> {extract_dir} ...")
        with zipfile.ZipFile(source) as zf:
            zf.extractall(extract_dir)
        return find_data_root(extract_dir)
    raise ValueError(f"--source {source} is neither a directory nor a .zip file.")


def create_splits(source_root, dest_root, ratios, seed=42):
    # Detect the name of the ground truth folder
    gt_source_name = "Label" if os.path.isdir(os.path.join(source_root, "Label")) else "GT"

    if os.path.isdir(os.path.join(source_root, "train")):
        print(f"Dataset at {source_root} already has 'train' split. Skipping split creation.")
        if os.path.abspath(source_root) != os.path.abspath(dest_root):
            print(f"Copying pre-split data to {dest_root}...")
            # Copy but ensure Label gets renamed to GT
            for split in ["train", "val", "test"]:
                for folder in ["A", "B", gt_source_name]:
                    src_dir = os.path.join(source_root, split, folder)
                    dst_folder = "GT" if folder in ["Label", "GT"] else folder
                    dst_dir = os.path.join(dest_root, split, dst_folder)
                    if os.path.isdir(src_dir):
                        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        return

    print(f"\nCreating train/val/test splits in {dest_root}...")
    
    a_dir = os.path.join(source_root, "A")
    if not os.path.isdir(a_dir):
        raise FileNotFoundError(f"Could not find 'A' directory in {source_root}. Cannot create splits.")
        
    if not os.path.isdir(os.path.join(source_root, gt_source_name)):
        raise FileNotFoundError(f"Could not find '{gt_source_name}' directory in {source_root}.")

    scene_ids = sorted([f for f in os.listdir(a_dir) if f.endswith('.png') or f.endswith('.tif')])
    
    # Shuffle for random distribution
    rng = random.Random(seed)
    rng.shuffle(scene_ids)
    
    total = len(scene_ids)
    train_end = int(total * ratios[0])
    val_end = train_end + int(total * ratios[1])
    
    splits = {
        "train": scene_ids[:train_end],
        "val": scene_ids[train_end:val_end],
        "test": scene_ids[val_end:]
    }
    
    for split_name, ids in splits.items():
        for folder in ["A", "B", "GT"]:
            os.makedirs(os.path.join(dest_root, split_name, folder), exist_ok=True)
        
        for sid in ids:
            for folder in ["A", "B", "GT"]:
                # Map destination "GT" folder to source "Label" or "GT" folder
                src_folder = gt_source_name if folder == "GT" else folder
                
                src_file = os.path.join(source_root, src_folder, sid)
                dst_file = os.path.join(dest_root, split_name, folder, sid)
                
                if not os.path.exists(src_file):
                    print(f"  [WARNING] Missing expected file: {src_file}")
                    continue
                
                if not os.path.exists(dst_file):
                    shutil.copy2(src_file, dst_file)
                    
    print(f"Splits successfully created: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")


def validate_and_report(root, sample_scenes=200, seed=0):
    print(f"\nValidating dataset at: {root}\n")
    ok = True
    for split in ("train", "val", "test"):
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            print(f"  [MISSING] {split}/ folder not found under {root}")
            ok = False
            continue
        try:
            ids = list_scene_ids(root, split)
        except FileNotFoundError as e:
            print(f"  [ERROR] {split}: {e}")
            ok = False
            continue

        rng = random.Random(seed)
        sample_ids = ids if len(ids) <= sample_scenes else rng.sample(ids, sample_scenes)
        flood_fracs = []
        for sid in sample_ids:
            gt = _load_mask(os.path.join(root, split, "GT", sid))
            flood_fracs.append(gt.mean())
        flood_fracs = np.array(flood_fracs)

        print(f"  {split:5s}: {len(ids):5d} scenes   "
              f"flood pixel fraction (sampled n={len(sample_ids)}): "
              f"mean={flood_fracs.mean():.3f} "
              f"min={flood_fracs.min():.3f} max={flood_fracs.max():.3f}")

    if ok:
        total = sum(len(list_scene_ids(root, s)) for s in ("train", "val", "test"))
        print(f"\nLooks good. {total} total scenes found.")
    else:
        print("\nFix the issues above before training. Expected layout:\n"
              "  <root>/train/A/*.png  <root>/train/B/*.png  <root>/train/GT/*.png\n"
              "  <root>/val/{A,B,GT}/*.png\n"
              "  <root>/test/{A,B,GT}/*.png")
    return ok


def main():
    p = argparse.ArgumentParser(description="Split, validate, and stage the full dataset.")
    p.add_argument("--source", required=True,
                   help="Path to either an already-extracted dataset folder or a .zip file.")
    p.add_argument("--dest", default="data_full",
                   help="Where the fully split and validated dataset should live.")
    p.add_argument("--split-ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15],
                   help="Ratios for train, val, and test splits (e.g., 0.7 0.15 0.15).")
    p.add_argument("--sample-scenes", type=int, default=200,
                   help="How many scenes per split to sample when reporting class balance.")
    args = p.parse_args()

    if not (0.99 <= sum(args.split_ratios) <= 1.01):
        raise ValueError("--split-ratios must sum to 1.0")

    raw_data_root = extract_if_zip(args.source, args.dest)
    
    create_splits(raw_data_root, args.dest, ratios=args.split_ratios)

    ok = validate_and_report(args.dest, sample_scenes=args.sample_scenes)

    if ok:
        print(f"\nUse --data-root {args.dest} with train.py "
              f"(or set SAR_CD_DATA_ROOT={args.dest}).")


if __name__ == "__main__":
    main()