"""Depth estimation for aicam, based on MiDaS."""

import sys

import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DepthEstimator:
    """MiDaS_small, with temporal smoothing.

    The model is not recurrent the way RVM is, so two nearly identical frames
    can produce slightly different depth maps. Without smoothing that shows up
    as defocus pulsing in the background. An exponential moving average on the
    low-resolution map removes it at no cost.

    `size=256` is the model's training resolution. Giving it more does not help;
    what matters here are the large surfaces, not the detail.
    """

    def __init__(self, device, dtype, size=256, smoothing=0.6):
        self.device = device
        self.dtype = dtype
        self.size = size
        self.smoothing = smoothing
        self._ema = None

        self.model = torch.hub.load(
            "intel-isl/MiDaS", "MiDaS_small", trust_repo=True
        ).eval().to(device, dtype)

        self.mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1)

    def reset(self):
        self._ema = None

    def __call__(self, src):
        """src: [1,3,H,W] RGB in 0..1. Returns [1,1,H,W] where 1 = near."""
        h, w = src.shape[-2:]
        x = F.interpolate(src, size=(self.size, self.size), mode="bicubic",
                          align_corners=False)
        d = self.model((x - self.mean) / self.std)      # [1, size, size]
        d = d[None]                                     # [1,1,size,size]

        # MiDaS gives relative inverse depth with no fixed scale, so normalise per frame.
        lo = d.amin()
        hi = d.amax()
        d = (d - lo) / (hi - lo).clamp(min=1e-6)

        if self._ema is None:
            self._ema = d
        else:
            self._ema = self.smoothing * self._ema + (1.0 - self.smoothing) * d

        return F.interpolate(self._ema, size=(h, w), mode="bilinear", align_corners=False)
