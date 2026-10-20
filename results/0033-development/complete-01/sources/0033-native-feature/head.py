"""Fixed supervised control on detached, already-computed native observations.

No presence/empty-mask decision is made here. The runner preserves those native
decisions. Statistics must be estimated on TRAIN only and supplied by the caller.
The caller also owns seed 0 and identical initial state/data order across arms.
"""

import torch
from torch import nn
from torch.nn import functional as F


CHANNELS = 32
LOW_SIZE = (256, 256)
STD_FLOOR = 1e-6
CORRECTION_BOUND = 4.0
PARAMETER_COUNT = 1377


def _statistic(value, name):
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().clone()
    if tensor.shape != (CHANNELS,):
        raise ValueError(f"{name} must have shape (32,)")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")
    return tensor


def _low_tensor(value, channels, name):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if (
        value.ndim != 4
        or value.shape[0] < 1
        or value.shape[1] != channels
        or tuple(value.shape[2:]) != LOW_SIZE
    ):
        raise ValueError(f"{name} must have shape [B,{channels},256,256], B>=1")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype float32")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


class NativeFeatureHead(nn.Module):
    """33→32 pointwise convolution, GELU, 32→1 padded 3×3 convolution.

    ``zero_features=True`` zeros the normalized feature channels. The native
    sigmoid-logit channel and the trainable architecture remain identical.
    ``train_std`` is population standard deviation, not variance.
    """

    def __init__(self, train_mean, train_std, *, zero_features=False):
        super().__init__()
        if type(zero_features) is not bool:
            raise TypeError("zero_features must be bool")
        mean = _statistic(train_mean, "train_mean")
        std = _statistic(train_std, "train_std")
        if bool((std < 0).any()):
            raise ValueError("train_std must be nonnegative")
        if mean.device != std.device:
            raise ValueError("TRAIN statistics must share a device")
        self.register_buffer("train_mean", mean.reshape(1, CHANNELS, 1, 1))
        self.register_buffer(
            "train_std", std.clamp_min(STD_FLOOR).reshape(1, CHANNELS, 1, 1)
        )
        self.zero_features = zero_features
        self.conv1 = nn.Conv2d(CHANNELS + 1, 32, kernel_size=1, dtype=torch.float32)
        self.activation = nn.GELU(approximate="none")
        self.conv2 = nn.Conv2d(32, 1, kernel_size=3, padding=1, dtype=torch.float32)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, features, native_logits):
        _low_tensor(features, CHANNELS, "features")
        _low_tensor(native_logits, 1, "native_logits")
        if features.shape[0] != native_logits.shape[0]:
            raise ValueError("features and native_logits batch sizes differ")
        if not (
            features.device
            == native_logits.device
            == self.train_mean.device
            == self.train_std.device
            == self.conv1.weight.device
            == self.conv2.weight.device
        ):
            raise ValueError("head, observations, and statistics must share a device")
        if any(p.dtype != torch.float32 for p in self.parameters()):
            raise TypeError("head parameters must remain float32")
        if self.train_mean.dtype != torch.float32 or self.train_std.dtype != torch.float32:
            raise TypeError("TRAIN statistics must remain float32")

        native = native_logits.detach()
        normalized = (features.detach() - self.train_mean) / self.train_std
        if not bool(torch.isfinite(normalized).all()):
            raise FloatingPointError("feature normalization produced nonfinite values")
        if self.zero_features:
            normalized = torch.zeros_like(normalized)
        inputs = torch.cat((normalized, torch.sigmoid(native)), dim=1)
        residual = self.conv2(self.activation(self.conv1(inputs)))
        if residual.dtype != torch.float32:
            raise TypeError("autocast is unsupported: head residual must be float32")
        output = native + CORRECTION_BOUND * torch.tanh(residual)
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError("head produced nonfinite logits")
        return output


def original_logits(low_logits, size_hw):
    """Resize low logits to original H,W; no threshold/presence decision."""
    _low_tensor(low_logits, 1, "low_logits")
    if (
        not isinstance(size_hw, (tuple, list))
        or len(size_hw) != 2
        or any(type(x) is not int or x <= 0 for x in size_hw)
    ):
        raise ValueError("size_hw must contain two positive Python integers")
    return F.interpolate(
        low_logits, size=tuple(size_hw), mode="bilinear", align_corners=False
    )
