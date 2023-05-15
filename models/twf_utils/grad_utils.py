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
