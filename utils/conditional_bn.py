from torch import nn
from torch.autograd import Function


class SelectClearGrad(Function):
    @staticmethod
    def forward(ctx, x, indices):
        ctx.indices = indices
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output[ctx.indices] = 0
        return grad_output, None


class ConditionalBatchNorm1d(nn.Module):

    def __init__(self, num_features, num_tasks):
        super(ConditionalBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm1d(num_features,
                                 affine=False)
        self.embed = nn.Embedding(num_tasks, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()

    def forward(self, x, task_id, flag_stop_grad=None):
        out = self.bn(x)
        gamma, beta = self.embed(task_id).chunk(2, 1)
        if flag_stop_grad is not None:
            gamma = SelectClearGrad.apply(gamma, flag_stop_grad)
            beta = SelectClearGrad.apply(beta, flag_stop_grad)
        out = gamma.view(-1, self.num_features) * out + \
            beta.view(-1, self.num_features)
        return out


class ConditionalBatchNorm2d(nn.Module):

    def __init__(self, num_features, num_tasks):
        super(ConditionalBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features,
                                 affine=False)
        self.embed = nn.Embedding(num_tasks, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()

    def forward(self, x, task_id, flag_stop_grad=None):
        out = self.bn(x)
        gamma, beta = self.embed(task_id).chunk(2, 1)
        if flag_stop_grad is not None:
            gamma = SelectClearGrad.apply(gamma, flag_stop_grad)
            beta = SelectClearGrad.apply(beta, flag_stop_grad)
        out = gamma.view(-1, self.num_features, 1, 1) * out + \
            beta.view(-1, self.num_features, 1, 1)
        return out
