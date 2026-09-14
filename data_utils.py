"""
Data loading for the S1GFloods benchmark (real Sentinel-1 SAR flood mapping
data), scaled to work with either:

  (a) the small 10-scene example subset committed in the paper's GitHub repo
      (used for the first pass of this project), or
  (b) the full ~5,360-pair benchmark (4,300 train / 530 val / 530 test),
      downloaded separately from Google Drive / HuggingFace / Baidu since
      those hosts are not reachable from this sandboxed environment.

Expected directory layout under DATA_ROOT (matches what the benchmark ships
as, and what the example subset already used):

    <DATA_ROOT>/train/A/*.png   pre-flood SAR image
    <DATA_ROOT>/train/B/*.png   post-flood SAR image
    <DATA_ROOT>/train/GT/*.png  binary ground truth mask (0 / 255)
    <DATA_ROOT>/val/{A,B,GT}/*.png
    <DATA_ROOT>/test/{A,B,GT}/*.png

Any of .png/.jpg/.jpeg/.tif/.tiff is accepted for the image files.

IMPORTANT DESIGN NOTE (flaw caught during review, first pass):
Patches must never be drawn from more than one split for the same
underlying scene. We only patchify train and val images. Test images are
always evaluated whole, at native resolution, so test numbers reflect
performance on entire real scenes rather than on patches that were seen
(overlapping) during training.

SECOND FLAW CAUGHT WHEN SCALING TO THE FULL DATASET:
The original implementation eagerly extracted every stride-32 patch from
every scene into a Python list at dataset-construction time. That is fine
for 10 example scenes (~300 patches) but is a correctness AND memory bug at
full scale: 4,300 scenes x ~49 overlapping 64x64 patches each x 3 arrays
(A, B, GT) x float32 works out to roughly 10+ GB of RAM before training
even starts, and takes minutes of up-front CPU time to extract. The dataset
below now auto-switches to an on-the-fly random-crop sampling mode once the
scene count crosses a threshold, so scaling from 10 scenes to 5,360 scenes
does not require the person to change anything.
"""

import glob
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# Above this many scenes in a split, switch from "extract every patch up
# front" to "sample random patches on the fly", to keep memory and startup
# time bounded regardless of dataset size.
EXHAUSTIVE_SCENE_LIMIT = 60


def _load_gray(path):
    """Load an image as single channel float32 in [0, 1]."""
    im = Image.open(path).convert("L")
    arr = np.array(im, dtype=np.float32) / 255.0
    return arr


def _load_mask(path):
    im = Image.open(path).convert("L")
    arr = np.array(im, dtype=np.float32)
    return (arr > 127).astype(np.float32)


def list_scene_ids(root, split):
    """Scene ids are filenames, taken from the A/ folder and matched by name
    against B/ and GT/. Any of the accepted image extensions is scanned."""
    a_dir = os.path.join(root, split, "A")
    files = []
    for ext in IMAGE_EXTS:
        files.extend(glob.glob(os.path.join(a_dir, f"*{ext}")))
    ids = sorted(os.path.basename(f) for f in files)

    # Sanity check: make sure B and GT actually have matching files, since a
    # silent mismatch here (e.g. someone re-exported only half a split)
    # would corrupt training without any visible error otherwise.
    b_dir = os.path.join(root, split, "B")
    gt_dir = os.path.join(root, split, "GT")
    missing_b = [sid for sid in ids if not os.path.exists(os.path.join(b_dir, sid))]
    missing_gt = [sid for sid in ids if not os.path.exists(os.path.join(gt_dir, sid))]
    if missing_b or missing_gt:
        raise FileNotFoundError(
            f"{split}: {len(missing_b)} scene(s) missing from B/, "
            f"{len(missing_gt)} scene(s) missing from GT/. "
            f"First few missing: B={missing_b[:3]} GT={missing_gt[:3]}"
        )
    return ids


def load_scene(root, split, scene_id):
    a = _load_gray(os.path.join(root, split, "A", scene_id))
    b = _load_gray(os.path.join(root, split, "B", scene_id))
    gt = _load_mask(os.path.join(root, split, "GT", scene_id))
    return a, b, gt


def per_image_standardize(img, eps=1e-6):
    """
    Normalize each SAR image by its own mean/std.
    SAR backscatter intensity is scene- and sensor-condition dependent, so a
    single global normalization constant across very different scenes (some
    urban, some rural, different incidence angles) would bias the model.
    Per-image standardization was added after noticing the raw [0,1] pixel
    scale gave very unstable training loss across scenes of very different
    average brightness.
    """
    mu = img.mean()
    sigma = img.std() + eps
    return (img - mu) / sigma


def extract_patches(a, b, gt, patch_size=64, stride=32):
    """Extract aligned overlapping patches from one scene."""
    h, w = a.shape
    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            pa = a[y:y + patch_size, x:x + patch_size]
            pb = b[y:y + patch_size, x:x + patch_size]
            pg = gt[y:y + patch_size, x:x + patch_size]
            patches.append((pa, pb, pg))
    return patches


def _random_crop(a, b, gt, patch_size, rng):
    h, w = a.shape
    if h < patch_size or w < patch_size:
        raise ValueError(
            f"Scene of size {(h, w)} is smaller than patch_size={patch_size}."
        )
    y = rng.randint(0, h - patch_size)
    x = rng.randint(0, w - patch_size)
    return (a[y:y + patch_size, x:x + patch_size],
            b[y:y + patch_size, x:x + patch_size],
            gt[y:y + patch_size, x:x + patch_size])


class _SceneCache:
    """Small LRU-ish cache so random-crop sampling does not re-decode the
    same PNG from disk on every single patch request. Each DataLoader worker
    process gets its own cache instance."""

    def __init__(self, root, split, max_scenes=256):
        self.root = root
        self.split = split
        self.max_scenes = max_scenes
        self._cache = {}
        self._order = []

    def get(self, scene_id):
        if scene_id in self._cache:
            return self._cache[scene_id]
        a, b, gt = load_scene(self.root, self.split, scene_id)
        a = per_image_standardize(a)
        b = per_image_standardize(b)
        self._cache[scene_id] = (a, b, gt)
        self._order.append(scene_id)
        if len(self._order) > self.max_scenes:
            oldest = self._order.pop(0)
            del self._cache[oldest]
        return a, b, gt


class SARChangeDataset(Dataset):
    """
    Patch-level dataset for train/val splits. Test scenes are handled
    separately (see FullSceneDataset) because we always evaluate on full,
    unseen scenes rather than on patches.

    Automatically picks between two modes:
      - "exhaustive": every stride-`stride` patch from every scene is
        extracted once at construction time and stored in memory. Used when
        the split is small (<= EXHAUSTIVE_SCENE_LIMIT scenes), matching the
        original behaviour used for the 10-scene example subset.
      - "random": scenes are loaded lazily (with a small LRU cache) and a
        fresh random patch location is drawn on every __getitem__ call.
        Used automatically once the split has more scenes than that, which
        is what the full ~5,360-pair benchmark needs to stay within memory
        and to start training in seconds rather than minutes.

    `epoch_size` only matters in "random" mode: it sets how many samples
    make up one nominal epoch (defaults to roughly one random 64x64 crop
    per scene per epoch).
    """

    def __init__(self, root, split, patch_size=64, stride=32, augment=False,
                 seed=0, mode="auto", epoch_size=None, cache_scenes=256):
        assert split in ("train", "val")
        self.root = root
        self.split = split
        self.augment = augment
        self.patch_size = patch_size
        self.rng = random.Random(seed)

        scene_ids = list_scene_ids(root, split)
        if len(scene_ids) == 0:
            raise FileNotFoundError(
                f"No scenes found under {os.path.join(root, split, 'A')}. "
                f"Check DATA_ROOT points at a folder containing "
                f"train/val/test subfolders with A/B/GT images."
            )
        self.scene_ids = scene_ids

        if mode == "auto":
            mode = "exhaustive" if len(scene_ids) <= EXHAUSTIVE_SCENE_LIMIT else "random"
        self.mode = mode

        if self.mode == "exhaustive":
            all_patches = []
            for sid in scene_ids:
                a, b, gt = load_scene(root, split, sid)
                a = per_image_standardize(a)
                b = per_image_standardize(b)
                all_patches.extend(extract_patches(a, b, gt, patch_size, stride))
            self.rng.shuffle(all_patches)
            self.patches = all_patches
        else:
            self.cache = _SceneCache(root, split, max_scenes=cache_scenes)
            self.epoch_size = epoch_size or len(scene_ids)

    def __len__(self):
        if self.mode == "exhaustive":
            return len(self.patches)
        return self.epoch_size

    def _augment(self, a, b, gt):
        if random.random() < 0.5:
            a, b, gt = a[:, ::-1].copy(), b[:, ::-1].copy(), gt[:, ::-1].copy()
        if random.random() < 0.5:
            a, b, gt = a[::-1, :].copy(), b[::-1, :].copy(), gt[::-1, :].copy()
        return a, b, gt

    def __getitem__(self, idx):
        if self.mode == "exhaustive":
            a, b, gt = self.patches[idx]
        else:
            sid = self.scene_ids[idx % len(self.scene_ids)]
            full_a, full_b, full_gt = self.cache.get(sid)
            a, b, gt = _random_crop(full_a, full_b, full_gt, self.patch_size, self.rng)

        if self.augment:
            # NOTE: we deliberately do NOT use random 90-degree rotation
            # augmentation for either model in the main comparison, since the
            # whole point of the experiment is to see which architecture
            # generalizes to unseen orientations *without* being shown
            # rotated examples during training.
            a, b, gt = self._augment(a, b, gt)

        a_t = torch.from_numpy(a).float().unsqueeze(0)
        b_t = torch.from_numpy(b).float().unsqueeze(0)
        gt_t = torch.from_numpy(gt).float().unsqueeze(0)
        return a_t, b_t, gt_t


class FullSceneDataset(Dataset):
    """
    Whole, un-patchified scenes for held-out evaluation (test split).

    `max_scenes`, if set, evaluates only a random subset of that many test
    scenes rather than every scene in the split. Useful for a full-scale
    dataset (e.g. 530 test scenes) where an 11-angle rotation sweep over
    every single one would be slow on CPU; leave it as None to evaluate the
    complete test set.
    """

    def __init__(self, root, split="test", max_scenes=None, seed=0):
        self.root = root
        self.split = split
        scene_ids = list_scene_ids(root, split)
        if max_scenes is not None and len(scene_ids) > max_scenes:
            rng = random.Random(seed)
            scene_ids = sorted(rng.sample(scene_ids, max_scenes))
        self.scene_ids = scene_ids

    def __len__(self):
        return len(self.scene_ids)

    def __getitem__(self, idx):
        sid = self.scene_ids[idx]
        a, b, gt = load_scene(self.root, self.split, sid)
        a = per_image_standardize(a)
        b = per_image_standardize(b)
        a_t = torch.from_numpy(a).float().unsqueeze(0)
        b_t = torch.from_numpy(b).float().unsqueeze(0)
        gt_t = torch.from_numpy(gt).float().unsqueeze(0)
        return a_t, b_t, gt_t, sid


def rotate_batch(a, b, gt, k):
    """Rotate a batch of (N,1,H,W) tensors by k*90 degrees (exact, no
    interpolation), used to test exact C4 equivariance and to build the
    rotated-test-set robustness benchmark."""
    a = torch.rot90(a, k, dims=(-2, -1))
    b = torch.rot90(b, k, dims=(-2, -1))
    gt = torch.rot90(gt, k, dims=(-2, -1))
    return a, b, gt


def rotate_arbitrary(a, b, gt, angle_deg):
    """Rotate by an arbitrary angle (bilinear interpolation) to test
    approximate equivariance beyond the exact 90-degree multiples that C4
    guarantees analytically."""
    import torchvision.transforms.functional as TF
    a = TF.rotate(a, angle_deg, interpolation=TF.InterpolationMode.BILINEAR)
    b = TF.rotate(b, angle_deg, interpolation=TF.InterpolationMode.BILINEAR)
    gt = TF.rotate(gt, angle_deg, interpolation=TF.InterpolationMode.NEAREST)
    return a, b, gt