import torch


class UnbiasedFisherModule(torch.nn.Module):

    def __init__(self, p, ewc_lambda: float, ewc_prior: float = 0.0):

        super().__init__()
        self.register_buffer(f'unnormalized_fisher', torch.zeros(p.shape))
        self.num_elems = self.unnormalized_fisher.numel()
        self.num_examples = 0
        self.current_task = -1

        self.register_buffer('ewc_lambda', torch.ones(1) * ewc_lambda)
        self.register_buffer('ewc_prior', torch.ones(1) * ewc_prior)

        self.register_buffer('ratio', torch.ones(1))

        self.register_buffer('coeff1', torch.ones(1))
        self.register_buffer('coeff2', torch.ones(1))
        self.register_buffer('norm', torch.ones(1))

        self.shape_sum = tuple((i + 1 for i in range(len(p.shape))))

    @torch.no_grad()
    def update(self, f, num_examples: int):
        assert num_examples > 0

        self.num_examples += num_examples
        self.unnormalized_fisher.add_(f)

        self.current_task += 1

        ratio = 0.5 if self.current_task == 0 else 1. / (self.current_task + 1)
        self.ratio.fill_(ratio)

        norm = 1. / self.num_examples
        self.norm.fill_(norm)

        coeff1 = self.ewc_lambda * (1. - self.ratio)
        self.coeff1.copy_(coeff1)

        coeff2 = self.ewc_lambda * self.ratio
        self.coeff2.copy_(coeff2)

    def dot_is_subbed(self):
        return True

    def get_fisher_matrix(self, scaled: bool = False):
        if scaled:
            return self.coeff1 * (self.ewc_prior + self.norm * self.unnormalized_fisher)
        return self.ewc_prior + self.norm * self.unnormalized_fisher

    def get_fisher_matrix_dot(self, scaled: bool = False):
        if scaled:
            return self.coeff2 * (self.ewc_prior + self.norm * self.unnormalized_fisher)
        return self.ewc_prior + self.norm * self.unnormalized_fisher

    def get_num_elems(self):
        return self.num_elems

    def trace(self):
        return self.get_fisher_matrix().sum()

    def dist(self, delta):
        return (self.get_fisher_matrix().unsqueeze(0) *
                delta.pow(2)).sum(dim=self.shape_sum)

    def forward(self, delta):
        return self.dist(delta)

    def dot_prod_no_grad(self, delta, delta_past):
        with torch.no_grad():
            a = (self.get_fisher_matrix_dot().unsqueeze(0) * delta_past)
        return (a * delta).sum((1, 2))

    def full_dot_prod_no_grad(self, delta, delta_past):
        with torch.no_grad():
            a = (self.get_fisher_matrix_dot().unsqueeze(0) * delta_past).unsqueeze(1)
        return (delta.unsqueeze(0) * a).sum((2, 3))

    def dot_prod(self, delta, idxs_delta_2):
        return ((delta * delta[idxs_delta_2]) * self.get_fisher_matrix_dot().unsqueeze(0)).sum((1, 2))

    def full_dot_prod(self, delta):
        return ((delta.unsqueeze(0) * self.get_fisher_matrix_dot().unsqueeze(0))
                * delta.unsqueeze(1)).sum((2, 3))


class CombinedFisherModule(torch.nn.Module):

    def __init__(self, p, ewc_lambda: float,
                 ewc_alpha: float, ewc_prior: float):

        super().__init__()

        self.register_buffer(f'unnormalized_fisher_pretrain',
                             torch.zeros(p.shape))
        self.register_buffer(f'unnormalized_fisher_prevtask',
                             ewc_prior * torch.ones(p.shape))

        self.num_elems = self.unnormalized_fisher_pretrain.numel()
        self.num_examples = 0
        self.current_task = -1

        self.register_buffer('ewc_lambda', torch.ones(1) * ewc_lambda)
        self.register_buffer('ewc_alpha', torch.ones(1) * ewc_alpha)
        self.register_buffer('ratio', torch.ones(1))

        self.register_buffer('coeff1', torch.ones(1))
        self.register_buffer('coeff2', torch.ones(1))
        self.register_buffer('coeff3', torch.ones(1))
        self.register_buffer('coeff4', torch.ones(1))

        self.register_buffer('norm', torch.ones(1))

        self.shape_sum = tuple((i + 1 for i in range(len(p.shape))))

    @torch.no_grad()
    def update(self, pretrain_fisher, num_examples: int):
        assert num_examples > 0

        self.num_examples += num_examples
        self.unnormalized_fisher_pretrain.add_(pretrain_fisher)

        self.current_task += 1

        T = 2 if self.current_task == 0 else self.current_task + 1

        self.ratio.fill_(1. / T)
        self.norm.fill_(1. / self.num_examples)

        coeff1 = self.ewc_alpha * self.ratio
        self.coeff1.copy_(coeff1).mul_(self.norm)

        coeff2 = self.ewc_lambda * (1. - self.ratio)
        self.coeff2.copy_(coeff2).mul_(self.norm)

        coeff3 = self.ratio * self.ewc_alpha * (1. / (T - 1))
        self.coeff3.copy_(coeff3).mul_(self.norm)

        coeff4 = self.ratio * self.ewc_lambda
        self.coeff4.copy_(coeff4).mul_(self.norm)

    @torch.no_grad()
    def update_fisher_prev_task(self, prevtask_fisher):
        self.unnormalized_fisher_prevtask.add_(prevtask_fisher)

    def dot_is_subbed(self):
        return True

    def get_fisher_matrix(self, scaled: bool = False):
        if scaled:
            return self.coeff1 * self.unnormalized_fisher_prevtask + \
                self.coeff2 * self.unnormalized_fisher_pretrain
        return self.norm * self.unnormalized_fisher_pretrain

    def get_fisher_matrix_dot(self, scaled: bool = False):
        if scaled:
            return self.coeff3 * self.unnormalized_fisher_prevtask + \
                self.coeff4 * self.unnormalized_fisher_pretrain
        return self.norm * self.unnormalized_fisher_pretrain

    def get_num_elems(self):
        return self.num_elems

    def trace(self):
        return self.get_fisher_matrix().sum()

    def dist(self, delta):
        return (self.get_fisher_matrix().unsqueeze(0) *
                delta.pow(2)).sum(dim=self.shape_sum)

    def forward(self, delta):
        return self.dist(delta)

    def dot_prod_no_grad(self, delta, delta_past):
        with torch.no_grad():
            a = (self.get_fisher_matrix_dot().unsqueeze(0) * delta_past)
        return (a * delta).sum((1, 2))

    def full_dot_prod_no_grad(self, delta, delta_past):
        with torch.no_grad():
            a = (self.get_fisher_matrix_dot().unsqueeze(0) * delta_past).unsqueeze(1)
        return (delta.unsqueeze(0) * a).sum((2, 3))

    def dot_prod(self, delta, idxs_delta_2):
        return ((delta * delta[idxs_delta_2]) * self.get_fisher_matrix_dot().unsqueeze(0)).sum((1, 2))

    def full_dot_prod(self, delta):
        return ((delta.unsqueeze(0) * self.get_fisher_matrix_dot().unsqueeze(0))
                * delta.unsqueeze(1)).sum((2, 3))


class AugmentedFisherModule(torch.nn.Module):

    def __init__(self, p, ewc_lambda: float,
                 ewc_alpha: float, ewc_prior: float):

        super().__init__()
        assert ewc_prior >= 0

        self.register_buffer(f'unnormalized_fisher', ewc_prior * torch.ones(p.shape))
        self.register_buffer(f'unnormalized_fisher_current', torch.zeros(p.shape))

        self.num_elems = self.unnormalized_fisher.numel()

        self.num_examples = 0
        self.num_current_examples = 0

        self.current_task = -1

        self.register_buffer('ewc_lambda', torch.ones(1) * ewc_lambda)
        self.register_buffer('ewc_alpha', torch.ones(1) * ewc_alpha)

        self.register_buffer('norm', torch.ones(1))
        self.register_buffer('ratio', torch.ones(1))
        self.register_buffer('gamma', torch.ones(1))
        self.register_buffer('coeff', torch.ones(1))
        self.register_buffer('coeff2', torch.ones(1))

        self.shape_sum = tuple((i + 1 for i in range(len(p.shape))))

    @torch.no_grad()
    def update(self, f, num_examples: int):
        assert num_examples > 0

        self.current_task += 1

        if self.current_task > 0:
            self.num_examples += self.num_current_examples
            self.unnormalized_fisher.add_(self.unnormalized_fisher_current)
        else:
            self.num_examples = num_examples

        num_prev_examples = self.num_examples - self.num_current_examples

        self.num_current_examples = num_examples
        self.unnormalized_fisher_current.copy_(f)

        T = 2 if self.current_task == 0 else self.current_task + 1
        self.ratio.fill_(1. / T)

        norm = 1. / self.num_examples
        self.norm.fill_(norm)

        gamma = self.ewc_lambda * (1. - self.ratio)
        self.gamma.copy_(gamma).mul_(self.norm)

        coeff = (self.ewc_alpha * T) - self.gamma * num_prev_examples
        self.coeff.copy_(coeff)

        if self.current_task > 0:
            self.coeff.mul_(1. / num_prev_examples)

        coeff2 = self.ewc_lambda * self.ratio
        self.coeff2.copy_(coeff2).mul_(self.norm)

    def dot_is_subbed(self):
        return False

    def get_fisher_matrix(self, scaled: bool = False):
        if scaled:
            return self.coeff * self.unnormalized_fisher - \
                self.gamma * self.unnormalized_fisher_current
        return self.norm * self.unnormalized_fisher

    def get_fisher_matrix_dot(self, scaled: bool = False):
        if scaled:
            return self.coeff2 * self.unnormalized_fisher
        return self.norm * self.unnormalized_fisher

    def get_num_elems(self):
        return self.num_elems

    def trace(self):
        return self.get_fisher_matrix().sum()

    def dist(self, delta):
        return (self.get_fisher_matrix().unsqueeze(0) *
                delta.pow(2)).sum(dim=self.shape_sum)

    def forward(self, delta):
        return self.dist(delta)

    def dot_prod_no_grad(self, delta, delta_past):
        with torch.no_grad():
            a = (self.get_fisher_matrix_dot().unsqueeze(0) * delta_past)
        return (a * delta).sum((1, 2))

    def full_dot_prod_no_grad(self, delta, delta_past):
        with torch.no_grad():
            a = (self.get_fisher_matrix_dot().unsqueeze(0) * delta_past).unsqueeze(1)
        return (delta.unsqueeze(0) * a).sum((2, 3))

    def dot_prod(self, delta, idxs_delta_2):
        return ((delta * delta[idxs_delta_2]) * self.get_fisher_matrix_dot().unsqueeze(0)).sum((1, 2))

    def full_dot_prod(self, delta):
        return ((delta.unsqueeze(0) * self.get_fisher_matrix_dot().unsqueeze(0))
                * delta.unsqueeze(1)).sum((2, 3))
