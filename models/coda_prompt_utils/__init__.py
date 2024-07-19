"""
This package contains utility functions for the CoDA Prompt model. Implements a custom version of ViT to add prompt parameters.
"""

import copy

import torch


def gram_schmidt(vv, start_c, end_c, return_in_parameter=True):
    """
    Code for this function is modified from:
    https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py

    Perform Gram-Schmidt orthogonalization on the input matrix vv.
    """

    def projection(u, v):
        denominator = (u * u).sum()

        if denominator < 1e-8:
            return None
        else:
            return (v * u).sum() / denominator * u

    # check if the tensor is 3D and flatten the last two dimensions if necessary
    is_3d = len(vv.shape) == 3
    if is_3d:
        shape_2d = copy.deepcopy(vv.shape)
        vv = vv.view(vv.shape[0], -1)

    # swap rows and columns
    vv = vv.T

    # process matrix size
    uu = torch.zeros_like(vv, device=vv.device)

    if start_c > 0:
        uu[:, 0:start_c] = vv[:, 0:start_c].clone()

    for k in range(start_c, end_c):
        redo = True
        while redo:
            redo = False
            vk = torch.randn_like(vv[:, k]).to(vv.device)
            uk = 0
            for j in range(0, k):
                if not redo:
                    uj = uu[:, j].clone()
                    proj = projection(uj, vk)
                    if proj is None:
                        redo = True
                        print('restarting!!!')
                    else:
                        uk = uk + proj
            if not redo:
                uu[:, k] = vk - uk
    for k in range(start_c, end_c):
        uk = uu[:, k].clone()
        uu[:, k] = uk / (uk.norm())

    # undo swapping of rows and columns
    uu = uu.T

    # return from 2D
    if is_3d:
        uu = uu.view(shape_2d)

    if return_in_parameter:
        return torch.nn.Parameter(uu)

    return uu
