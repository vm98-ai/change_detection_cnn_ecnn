"""
Two Siamese change-detection networks with matched capacity:

1. BaselineCNN     - ordinary convolutions, no built-in rotation symmetry.
2. EquivariantCNN  - e2cnn steerable convolutions built on the C4 group
                     (exact 90/180/270-degree rotation equivariance).

Both take a pre-flood patch A and a post-flood patch B and predict a
per-pixel flood probability map at the same resolution (no pooling, so we
avoid needing a decoder / skip connections, which keeps the two
architectures directly comparable).
"""

import torch
import torch.nn as nn

from e2cnn import gspaces
from e2cnn import nn as enn


class BaselineCNN(nn.Module):
    def __init__(self, base_ch=16):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        # Siamese encoder, shared weights between A and B branches.
        self.encoder = nn.Sequential(
            block(1, base_ch),
            block(base_ch, base_ch * 2),
            block(base_ch * 2, base_ch * 2),
        )

        # Fusion head sees concatenated features from both branches.
        self.head = nn.Sequential(
            block(base_ch * 4, base_ch * 2),
            nn.Conv2d(base_ch * 2, 1, kernel_size=1),
        )

    def forward(self, a, b):
        fa = self.encoder(a)
        fb = self.encoder(b)
        f = torch.cat([fa, fb], dim=1)
        return self.head(f)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class EquivariantCNN(nn.Module):
    """
    C4-steerable Siamese network. Field types use the regular representation
    of C4 internally, so a 90-degree rotation of the two input patches
    produces an exactly 90-degree-rotated feature map at every layer, and
    therefore an exactly 90-degree-rotated output mask.

    `fields` controls channel width in units of "fields"; each regular field
    of C4 corresponds to 4 scalar channels, so fields=(4, 8, 8) gives
    (16, 32, 32) raw channels per branch, matching BaselineCNN's
    (16, 32, 32) channel widths for a fair comparison.
    """

    def __init__(self, fields=(4, 8, 8)):
        super().__init__()
        self.gspace = gspaces.Rot2dOnR2(N=4)

        self.in_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr])

        t0 = self.in_type
        t1 = enn.FieldType(self.gspace, fields[0] * [self.gspace.regular_repr])
        t2 = enn.FieldType(self.gspace, fields[1] * [self.gspace.regular_repr])
        t3 = enn.FieldType(self.gspace, fields[2] * [self.gspace.regular_repr])
        self.out_encoder_type = t3

        def eq_block(tin, tout, kernel_size=3):
            return enn.SequentialModule(
                enn.R2Conv(tin, tout, kernel_size=kernel_size, padding=kernel_size // 2),
                enn.InnerBatchNorm(tout),
                enn.ReLU(tout, inplace=True),
            )

        self.encoder = enn.SequentialModule(
            eq_block(t0, t1),
            eq_block(t1, t2),
            eq_block(t2, t3),
        )

        fused_in_type = enn.FieldType(self.gspace, 2 * fields[2] * [self.gspace.regular_repr])
        fused_mid_type = enn.FieldType(self.gspace, fields[1] * [self.gspace.regular_repr])
        out_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr])

        self.head = enn.SequentialModule(
            eq_block(fused_in_type, fused_mid_type, kernel_size=3),
            enn.R2Conv(fused_mid_type, out_type, kernel_size=1),
        )

    def _encode(self, x):
        x = enn.GeometricTensor(x, self.in_type)
        f = self.encoder(x)
        return f

    def forward(self, a, b):
        fa = self._encode(a)
        fb = self._encode(b)
        fused_tensor = torch.cat([fa.tensor, fb.tensor], dim=1)
        fused_type = enn.FieldType(self.gspace, self.out_encoder_type.representations * 2)
        fused = enn.GeometricTensor(fused_tensor, fused_type)
        out = self.head(fused)
        return out.tensor

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def build_models():
    baseline = BaselineCNN(base_ch=16)
    equiv = EquivariantCNN(fields=(13, 19, 19))
    return baseline, equiv


if __name__ == "__main__":
    baseline, equiv = build_models()
    print("BaselineCNN params:", baseline.n_params())
    print("EquivariantCNN params:", equiv.n_params())

    a = torch.randn(2, 1, 64, 64)
    b = torch.randn(2, 1, 64, 64)
    out_base = baseline(a, b)
    out_eq = equiv(a, b)
    print("baseline out:", out_base.shape)
    print("equivariant out:", out_eq.shape)
