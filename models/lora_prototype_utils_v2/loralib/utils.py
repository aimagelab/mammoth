import torch

def _set_grad_to_zero(my_var):
    if my_var.grad is None:
        my_var.grad = torch.zeros_like(my_var)
    else:
        torch.nn.init.zeros_(my_var.grad)