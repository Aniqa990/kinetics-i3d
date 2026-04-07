"""
PyTorch I3D model — Inception-v1 Inflated 3D ConvNet.

Ported from the TensorFlow implementation by DeepMind:
  https://github.com/deepmind/kinetics-i3d

This version is compatible with the pre-converted PyTorch weights from:
  https://github.com/piergiaj/pytorch-i3d  (rgb_imagenet.pt / flow_imagenet.pt)

Architecture paper:
  "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset"
  Carreira & Zisserman, CVPR 2017  https://arxiv.org/abs/1705.07750
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────── building blocks ─────────────────────────────

class MaxPool3dSamePad(nn.Module):
    """3-D max-pool that replicates TensorFlow's 'SAME' padding."""
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else [kernel_size]*3
        self.stride      = stride      if isinstance(stride,      (list, tuple)) else [stride]*3

    def forward(self, x):
        # compute how much padding is needed on each axis
        pad = []
        for i in range(3):          # T, H, W
            in_dim  = x.shape[2 + i]
            out_dim = (in_dim + self.stride[i] - 1) // self.stride[i]
            p       = max((out_dim - 1) * self.stride[i] + self.kernel_size[i] - in_dim, 0)
            pad = [p // 2, p - p // 2] + pad   # prepend because F.pad reads right-to-left
        x = F.pad(x, pad)
        return F.max_pool3d(x, self.kernel_size, self.stride)


class Unit3D(nn.Module):
    """Conv3D + optional BatchNorm + optional activation."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):
        super().__init__()
        self.name           = name
        self.activation_fn  = activation_fn
        self.use_batch_norm = use_batch_norm

        self.conv3d = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_shape,
            stride=stride,
            padding=padding,
            bias=use_bias)

        if use_batch_norm:
            self.bn = nn.BatchNorm3d(out_channels, eps=0.001, momentum=0.01)

    def forward(self, x):
        x = self.conv3d(x)
        if self.use_batch_norm:
            x = self.bn(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x


# ─────────────────────────────── inception block ─────────────────────────────

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels_list, name):
        """
        out_channels_list: [b0, b1_r, b1, b2_r, b2, b3]
            b0:   branch0 1×1 conv output channels
            b1_r: branch1 1×1 reduction channels
            b1:   branch1 3×3 conv output channels
            b2_r: branch2 1×1 reduction channels
            b2:   branch2 3×3 conv output channels
            b3:   branch3 pool + 1×1 output channels
        """
        super().__init__()
        b0, b1_r, b1, b2_r, b2, b3 = out_channels_list

        # Branch 0: 1×1
        self.b0 = Unit3D(in_channels, b0,   kernel_shape=[1,1,1], padding=0, name=name+'/b0')

        # Branch 1: 1×1 → 3×3
        self.b1_r = Unit3D(in_channels, b1_r, kernel_shape=[1,1,1], padding=0, name=name+'/b1_r')
        self.b1   = Unit3D(b1_r,        b1,   kernel_shape=[3,3,3], padding=1, name=name+'/b1')

        # Branch 2: 1×1 → 3×3
        self.b2_r = Unit3D(in_channels, b2_r, kernel_shape=[1,1,1], padding=0, name=name+'/b2_r')
        self.b2   = Unit3D(b2_r,        b2,   kernel_shape=[3,3,3], padding=1, name=name+'/b2')

        # Branch 3: MaxPool → 1×1
        self.b3_pool = MaxPool3dSamePad(kernel_size=[3,3,3], stride=[1,1,1])
        self.b3      = Unit3D(in_channels, b3, kernel_shape=[1,1,1], padding=0, name=name+'/b3')

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1(self.b1_r(x))
        b2 = self.b2(self.b2_r(x))
        b3 = self.b3(self.b3_pool(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


# ─────────────────────────────── full I3D model ──────────────────────────────

class InceptionI3d(nn.Module):
    """
    Inception-v1 I3D.

    Input shape: (B, C, T, H, W)  — PyTorch channel-first convention.
    """

    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7', 'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1', 'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b', 'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e', 'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b', 'Mixed_5c',
        'Logits', 'Predictions',
    )

    def __init__(self,
                 num_classes=400,
                 spatial_squeeze=True,
                 final_endpoint='Logits',
                 in_channels=3,
                 dropout_keep_prob=1.0):
        super().__init__()
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError(f'Unknown final endpoint: {final_endpoint}')

        self.num_classes       = num_classes
        self.spatial_squeeze   = spatial_squeeze
        self.final_endpoint    = final_endpoint
        self.dropout_keep_prob = dropout_keep_prob

        self.end_points = {}

        # ── stem ────────────────────────────────────────────────────────────
        self.Conv3d_1a_7x7 = Unit3D(in_channels, 64,
                                    kernel_shape=[7,7,7], stride=[2,2,2],
                                    padding=3, name='Conv3d_1a_7x7')
        self.MaxPool3d_2a_3x3 = MaxPool3dSamePad([1,3,3], [1,2,2])
        self.Conv3d_2b_1x1    = Unit3D(64,  64,  kernel_shape=[1,1,1], padding=0)
        self.Conv3d_2c_3x3    = Unit3D(64,  192, kernel_shape=[3,3,3], padding=1)
        self.MaxPool3d_3a_3x3 = MaxPool3dSamePad([1,3,3], [1,2,2])

        # ── inception blocks ─────────────────────────────────────────────────
        #                        in   b0  b1r  b1  b2r  b2  b3
        self.Mixed_3b = InceptionBlock(192,  [ 64,  96, 128,  16,  32,  32], 'Mixed_3b')
        self.Mixed_3c = InceptionBlock(256,  [128, 128, 192,  32,  96,  64], 'Mixed_3c')

        self.MaxPool3d_4a_3x3 = MaxPool3dSamePad([3,3,3], [2,2,2])

        self.Mixed_4b = InceptionBlock(480,  [192,  96, 208,  16,  48,  64], 'Mixed_4b')
        self.Mixed_4c = InceptionBlock(512,  [160, 112, 224,  24,  64,  64], 'Mixed_4c')
        self.Mixed_4d = InceptionBlock(512,  [128, 128, 256,  24,  64,  64], 'Mixed_4d')
        self.Mixed_4e = InceptionBlock(512,  [112, 144, 288,  32,  64,  64], 'Mixed_4e')
        self.Mixed_4f = InceptionBlock(528,  [256, 160, 320,  32, 128, 128], 'Mixed_4f')

        self.MaxPool3d_5a_2x2 = MaxPool3dSamePad([2,2,2], [2,2,2])

        self.Mixed_5b = InceptionBlock(832,  [256, 160, 320,  32, 128, 128], 'Mixed_5b')
        self.Mixed_5c = InceptionBlock(832,  [384, 192, 384,  48, 128, 128], 'Mixed_5c')

        # ── logits head ──────────────────────────────────────────────────────
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=[1, 1, 1])
        self.dropout  = nn.Dropout(p=1 - dropout_keep_prob)
        self.logits   = Unit3D(1024, num_classes,
                               kernel_shape=[1,1,1], padding=0,
                               activation_fn=None,
                               use_batch_norm=False,
                               use_bias=True,
                               name='logits')

    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W) float tensor, values in [-1, 1].
        Returns:
            Averaged logits of shape (B, num_classes).
        """
        # stem
        x = self.Conv3d_1a_7x7(x)
        x = self.MaxPool3d_2a_3x3(x)
        x = self.Conv3d_2b_1x1(x)
        x = self.Conv3d_2c_3x3(x)
        x = self.MaxPool3d_3a_3x3(x)

        # inception stack
        x = self.Mixed_3b(x)
        x = self.Mixed_3c(x)
        x = self.MaxPool3d_4a_3x3(x)
        x = self.Mixed_4b(x)
        x = self.Mixed_4c(x)
        x = self.Mixed_4d(x)
        x = self.Mixed_4e(x)
        x = self.Mixed_4f(x)
        x = self.MaxPool3d_5a_2x2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)

        # logit head
        x = self.avg_pool(x)              # (B, 1024, T', 1, 1)
        x = self.dropout(x)
        x = self.logits(x)                # (B, num_classes, T', H', W')

        # Collapse all spatial and temporal dims that may remain.
        # Works whether self.logits is the original Unit3D or a replaced head.
        if x.dim() == 5:
            # (B, C, T, H, W) → average over T, H, W
            x = x.mean(dim=[2, 3, 4])     # → (B, C)
        elif x.dim() == 4:
            x = x.mean(dim=[2, 3])        # → (B, C)
        elif x.dim() == 3:
            x = x.mean(dim=2)             # → (B, C)
        # if already (B, C) — nothing to do

        return x                          # (B, num_classes)

    def extract_features(self, x):
        """Return 1024-d feature vector before the logit head."""
        x = self.Conv3d_1a_7x7(x)
        x = self.MaxPool3d_2a_3x3(x)
        x = self.Conv3d_2b_1x1(x)
        x = self.Conv3d_2c_3x3(x)
        x = self.MaxPool3d_3a_3x3(x)
        x = self.Mixed_3b(x)
        x = self.Mixed_3c(x)
        x = self.MaxPool3d_4a_3x3(x)
        x = self.Mixed_4b(x)
        x = self.Mixed_4c(x)
        x = self.Mixed_4d(x)
        x = self.Mixed_4e(x)
        x = self.Mixed_4f(x)
        x = self.MaxPool3d_5a_2x2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.avg_pool(x)
        return x                          # (B, 1024, T', 1, 1)