import torch
from torch import nn
from torch.nn import functional as F


class SPKDLoss(nn.Module):
    """
    "Similarity-Preserving Knowledge Distillation"
    """

    def __init__(self, reduction):
        super().__init__()
        self.reduction = reduction

    def matmul_and_normalize(self, z):
        z = torch.flatten(z, 1)
        return F.normalize(torch.matmul(z, torch.t(z)), 1)

    def compute_spkd_loss(self, teacher_outputs, student_outputs):
        g_t = self.matmul_and_normalize(teacher_outputs)
        g_s = self.matmul_and_normalize(student_outputs)
        return torch.norm(g_t - g_s) ** 2

    def forward(self, teacher_outputs, student_outputs):
        batch_size = teacher_outputs.shape[0]
        spkd_losses = self.compute_spkd_loss(teacher_outputs, student_outputs)
        spkd_loss = spkd_losses.sum()
        return spkd_loss / (batch_size ** 2) if self.reduction == 'batchmean' else spkd_loss
