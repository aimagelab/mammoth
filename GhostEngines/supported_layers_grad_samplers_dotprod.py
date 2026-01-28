import torch
import transformers.pytorch_utils
from torch import nn
import torch.nn.functional as F
import time
from typing import Dict


def _compute_linear_dot_product(
    layer: nn.Linear,
    A_train: torch.Tensor,
    # A_val: torch.Tensor,
    B_train: torch.Tensor,
    val_grads_dict: Dict[str, torch.Tensor]
):
    """Computes the gradient dot-product for an nn.Linear layer."""
    if A_train is None or B_train is None:
        raise ValueError("Cannot compute without activations or backprops")

    with torch.no_grad():
        g_val_weight  = val_grads_dict.get(f"{layer.layer_name}.weight")
        g_val_bias = val_grads_dict.get(f"{layer.layer_name}.bias")

        # Keep device/dtype; avoid unnecessary casts
        A_train = A_train.detach()
        B_train = B_train.detach()
        dtype = A_train.dtype
        if g_val_weight is not None:
            g_val_weight = g_val_weight.detach().to(dtype=dtype)
            # Optimization: Use F.linear to compute A @ W.T instead of creating large (b, p, d) tensor
            # This computes the dot product between the gradient of the loss w.r.t output (B_train)
            # and the change in output caused by the validation weight direction (A_train @ W_val.T).
            prod = B_train * F.linear(A_train, g_val_weight)
            layer.weight.grad_dot_prod = prod.sum(dim=tuple(range(1, prod.dim())))

        if layer.bias is not None and g_val_bias is not None:
            g_val_bias = g_val_bias.detach().to(dtype=dtype)
            sum_dims_train = list(range(1, B_train.dim() - 1))
            g_train_i_bias = B_train.sum(dim=sum_dims_train) if B_train.dim() > 2 else B_train
            layer.bias.grad_dot_prod = (g_train_i_bias * g_val_bias).sum(dim=1)

# todo: 检查各层的计算逻辑
def _compute_conv2d_dot_product(
    layer: nn.Conv2d,
    A_train: torch.Tensor,
    B_train: torch.Tensor,
    val_grads_dict: Dict[str, torch.Tensor]
) -> None:
    """
    Computes dot-product for Conv2d using pure train activations/backprops and pre-computed validation gradients.
    """
    if A_train is None or B_train is None:
        return

    with torch.no_grad():
        A_train = A_train.detach()
        B_train = B_train.detach()
        dtype = A_train.dtype

        # --- Weight dot product ---
        g_val_weight = val_grads_dict.get(f"{layer.layer_name}.weight")
        if g_val_weight is not None:
            g_val_weight = g_val_weight.detach().to(dtype=dtype)
            
            # Optimization: Use F.conv2d instead of unfold + matmul
            # We want <dL/dW_train, dL/dW_val>
            # This is equivalent to <B_train, A_train * dL/dW_val> where * is convolution.
            Y_val = F.conv2d(
                A_train, 
                g_val_weight, 
                bias=None, 
                stride=layer.stride, 
                padding=layer.padding, 
                dilation=layer.dilation, 
                groups=layer.groups
            )
            prod = B_train * Y_val
            layer.weight.grad_dot_prod = prod.sum(dim=tuple(range(1, prod.dim())))

        # --- Bias dot product ---
        if layer.bias is not None:
            g_val_bias = val_grads_dict.get(f"{layer.layer_name}.bias")
            if g_val_bias is not None:
                g_val_bias = g_val_bias.detach().to(dtype=dtype)
                # Per-sample training bias grads
                g_train_i_bias = B_train.sum(dim=(2, 3)) # (b, p)
                layer.bias.grad_dot_prod = (g_train_i_bias * g_val_bias).sum(dim=1)

def _compute_batchnorm2d_dot_product(
    layer: nn.BatchNorm2d,
    A_train: torch.Tensor,
    B_train: torch.Tensor,
    val_grads_dict: Dict[str, torch.Tensor]
) -> None:
    # note: test w/o bn grad dot: failed
    """
    Computes dot-product for BatchNorm2d using pure train activations/backprops and pre-computed validation gradients.
    IMPORTANT: This must be called when the model is in train() mode to get the correct behavior.
    """
    if A_train is None or B_train is None:
        return

    with torch.no_grad():
        A_train = A_train.detach()
        B_train = B_train.detach()
        dtype = A_train.dtype

        # To get the correct per-sample grads, we must use the train-only statistics
        mean = A_train.mean(dim=(0, 2, 3), keepdim=True)
        var = A_train.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
        eps = layer.eps
        std = torch.sqrt(var + eps)

        # --- Weight dot product ---
        g_val_weight = val_grads_dict.get(f"{layer.layer_name}.weight")
        if g_val_weight is not None:
            g_val_weight = g_val_weight.detach().to(dtype=dtype)
            
            # Optimization: Avoid creating normalized_A_train (b, c, h, w)
            sum_BA = (B_train * A_train).sum(dim=(2, 3))
            sum_B = B_train.sum(dim=(2, 3))
            
            mean_s = mean.view(1, -1)
            std_s = std.view(1, -1)
            
            g_train_i_weight = (sum_BA - sum_B * mean_s) / std_s

            # Dot product
            layer.weight.grad_dot_prod = (g_train_i_weight * g_val_weight).sum(dim=1)

        # --- Bias dot product ---
        if layer.bias is not None:
            g_val_bias = val_grads_dict.get(f"{layer.layer_name}.bias")
            if g_val_bias is not None:
                g_val_bias = g_val_bias.detach().to(dtype=dtype)
                # Per-sample training grad for bias (beta)
                if 'sum_B' not in locals():
                     sum_B = B_train.sum(dim=(2, 3))
                g_train_i_bias = sum_B
                # Dot product
                layer.bias.grad_dot_prod = (g_train_i_bias * g_val_bias).sum(dim=1)

# The list of supported layers now maps to the new dot product functions
_supported_layers_dotprod = {
    nn.Linear: _compute_linear_dot_product,
    nn.Conv2d: _compute_conv2d_dot_product,
    nn.BatchNorm2d: _compute_batchnorm2d_dot_product,
}

