from torch import nn
from torch.autograd import Function


class SelectClearGrad(Function):
    """
    A custom autograd function that clears gradients for selected indices.
    """

    @staticmethod
    def forward(ctx, x, indices):
        """
        Forward pass of the SelectClearGrad function.

        Args:
            x (torch.Tensor): The input tensor.
            indices (torch.Tensor): The indices to clear gradients for.

        Returns:
            torch.Tensor: The input tensor, reshaped to match the shape of x.
        """
        ctx.indices = indices
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the SelectClearGrad function.

        Args:
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            Tuple[torch.Tensor, None]: The gradient of the input tensor and None.
        """
        grad_output[ctx.indices] = 0
        return grad_output, None


class ConditionalBatchNorm1d(nn.Module):
    """
    Conditional Batch Normalization for 1D inputs.
    """

    def __init__(self, num_features, num_conditions):
        """
        Initializes the ConditionalBatchNorm1d module.

        Args:
            num_features (int): The number of input features.
            num_conditions (int): The number of conditioning variables.
        """
        super(ConditionalBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.embed = nn.Embedding(num_conditions, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()

    def forward(self, x, cond_id, flag_stop_grad=None):
        """
        Compute Conditional Batch Normalization for 1D inputs.

        The input tensor `x` is applied with a specific 1D batch normalization layer, specified by the `cond_id`.

        Args:
            x (torch.Tensor): The input tensor.
            cond_id (torch.Tensor): The index of the conditioning.
            flag_stop_grad (torch.Tensor, optional): The flag to stop gradients for gamma and beta.

        Returns:
            torch.Tensor: The output tensor.
        """
        out = self.bn(x)
        gamma, beta = self.embed(cond_id).chunk(2, 1)
        if flag_stop_grad is not None:
            gamma = SelectClearGrad.apply(gamma, flag_stop_grad)
            beta = SelectClearGrad.apply(beta, flag_stop_grad)
        out = gamma.view(-1, self.num_features) * out + beta.view(-1, self.num_features)
        return out


class ConditionalBatchNorm2d(nn.Module):
    """
    Conditional Batch Normalization for 2D inputs.
    """

    def __init__(self, num_features, num_conditions):
        """
        Initializes the ConditionalBatchNorm2d module.

        Args:
            num_features (int): The number of input features.
            num_conditions (int): The number of conditioning variables.
        """
        super(ConditionalBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_conditions, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()

    def forward(self, x, cond_id, flag_stop_grad=None):
        """
        Compute Conditional Batch Normalization for 2D inputs.

        The input tensor `x` is applied with a specific 2D batch normalization layer, specified by the `cond_id`.

        Args:
            x (torch.Tensor): The input tensor.
            cond_id (torch.Tensor): The index of the conditioning.
            flag_stop_grad (torch.Tensor, optional): The flag to stop gradients for gamma and beta.

        Returns:
            torch.Tensor: The output tensor.
        """
        out = self.bn(x)
        gamma, beta = self.embed(cond_id).chunk(2, 1)
        if flag_stop_grad is not None:
            gamma = SelectClearGrad.apply(gamma, flag_stop_grad)
            beta = SelectClearGrad.apply(beta, flag_stop_grad)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out
