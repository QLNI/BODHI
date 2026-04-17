#!/usr/bin/env python3
"""
Minimal LoRA for SmallGPT — parameter-efficient fine-tuning.

Base weights stay frozen. Only the low-rank adapter matrices (A, B) are trained.
This preserves BODHI's learned voice while letting each user's BODHI adapt to
their actual conversations overnight.

Typical adapter size for rank=8 on SmallGPT 50M: ~3-5 MB. Training 100-200
steps on recent conversations takes a few minutes on the 3080 or ~20 minutes
on CPU.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear and adds a trainable low-rank adapter."""

    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: int = 16,
                 dropout: float = 0.0):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        d_in = base_linear.in_features
        d_out = base_linear.out_features
        self.A = nn.Parameter(torch.zeros(rank, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, rank))
        # Kaiming-ish init for A, zero for B -> zero initial contribution
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta = self.drop(x) @ self.A.t() @ self.B.t()
        return base_out + delta * self.scale


def _replace_linear(parent: nn.Module, name: str, rank: int, alpha: int, dropout: float):
    layer = getattr(parent, name, None)
    if isinstance(layer, nn.Linear):
        setattr(parent, name, LoRALinear(layer, rank=rank, alpha=alpha, dropout=dropout))


def loraify(model, rank: int = 8, alpha: int = 16, dropout: float = 0.0,
            attention: bool = True, mlp: bool = True, head: bool = False):
    """Wrap targeted linear layers in a SmallGPT model with LoRA adapters.

    By default, adapts attention qkv/proj and MLP fc1/fc2 in every block.
    Head (vocab projection) is left alone unless head=True.
    Base weights are frozen; only A/B params remain trainable.
    """
    for p in model.parameters():
        p.requires_grad = False

    if hasattr(model, "blocks"):
        for block in model.blocks:
            if attention and hasattr(block, "attn"):
                _replace_linear(block.attn, "qkv", rank, alpha, dropout)
                _replace_linear(block.attn, "proj", rank, alpha, dropout)
            if mlp and hasattr(block, "mlp"):
                _replace_linear(block.mlp, "fc1", rank, alpha, dropout)
                _replace_linear(block.mlp, "fc2", rank, alpha, dropout)
    if head and hasattr(model, "head"):
        _replace_linear(model, "head", rank, alpha, dropout)

    return model


def lora_parameters(model):
    """Yield only the trainable (LoRA A/B) parameters."""
    for name, p in model.named_parameters():
        if p.requires_grad:
            yield p


def lora_state_dict(model):
    """Return just the adapter state (A/B tensors keyed by module path)."""
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[name + ".A"] = module.A.detach().cpu()
            state[name + ".B"] = module.B.detach().cpu()
    return state


def load_lora_state(model, state: dict, strict: bool = True):
    """Load adapter state back into a loraified model."""
    loaded = 0
    missing = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            ak = name + ".A"
            bk = name + ".B"
            if ak in state and bk in state:
                module.A.data.copy_(state[ak].to(module.A.device))
                module.B.data.copy_(state[bk].to(module.B.device))
                loaded += 1
            else:
                missing.append(name)
    if strict and missing:
        raise RuntimeError("LoRA state missing entries for: %s" % missing[:3])
    return loaded


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
