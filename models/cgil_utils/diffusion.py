import math
from dataclasses import dataclass

import torch
import tqdm


@dataclass
class NoiseSchedule:
    num_steps: int
    beta_t: torch.Tensor
    alpha_t: torch.Tensor
    alpha_bar_t: torch.Tensor
    img_weight: torch.Tensor
    noise_weight: torch.Tensor

    def init(
        self,
        num_steps: int,
        beta_t: torch.Tensor,
        alpha_t: torch.Tensor,
        alpha_bar_t: torch.Tensor,
        img_weight: torch.Tensor,
        noise_weight: torch.Tensor,
    ) -> None:
        self.num_steps = num_steps
        self.beta_t = beta_t
        self.alpha_t = alpha_t
        self.alpha_bar_t = alpha_bar_t
        self.img_weight = img_weight
        self.noise_weight = noise_weight

    def to(self, device: torch.device):
        self.beta_t = self.beta_t.to(device)
        self.alpha_t = self.alpha_t.to(device)
        self.alpha_bar_t = self.alpha_bar_t.to(device)
        self.img_weight = self.img_weight.to(device)
        self.noise_weight = self.noise_weight.to(device)
        return self


def get_cosine_schedule(num_steps: int, s: float = 0, exp: int = 2):
    alpha_bar_t = torch.cos(
        (torch.linspace(1, num_steps, steps=num_steps) / num_steps + s)
        / (1 + s)
        * math.pi
        / 2
    ).pow(exp)
    beta_t = 1 - alpha_bar_t / (alpha_bar_t.roll(1))
    beta_t[0] = max(2 * beta_t[1] - beta_t[2], 0)
    alpha_t = 1 - beta_t
    img_weight = torch.sqrt(alpha_bar_t)
    noise_weight = torch.sqrt(1 - alpha_bar_t)
    return NoiseSchedule(
        num_steps, beta_t, alpha_t, alpha_bar_t, img_weight, noise_weight
    )


def sinusoidal_embedding(
    index: torch.Tensor,
    embedding_dim: int,
    num_training_steps: int,
    device: torch.device,
) -> torch.Tensor:
    assert len(index.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(num_training_steps) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
    emb = index.unsqueeze(1) * emb.unsqueeze(0).to(device)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class MLPDiffusion(torch.nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_hidden, num_steps, device) -> None:
        super().__init__()
        self.silu = torch.nn.SiLU()

        self.fc_input = torch.nn.Linear(embed_dim, hidden_dim)
        self.hidden_layers = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden)]
        )
        self.fc_output = torch.nn.Linear(hidden_dim, embed_dim)
        self.fc_emb1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_emb2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.num_steps = num_steps
        self.device = device
        self.hidden_dim = hidden_dim

    def forward(self, x, timestep):
        t_emb = sinusoidal_embedding(timestep, self.hidden_dim, self.num_steps, self.device)
        t_emb = self.silu(self.fc_emb1(t_emb))
        t_emb = self.fc_emb2(t_emb)

        x = self.silu(self.fc_input(x))
        for layer in self.hidden_layers:
            x = self.silu(layer(x) + t_emb)
        return self.fc_output(x)


class DiffusionCA(torch.nn.Module):

    @staticmethod
    def q_function(x, schedule, timestep=None):
        if timestep is None:
            timestep = torch.randint(low=0, high=schedule.num_steps, size=(x.shape[0],))

        noise = torch.randn_like(x) * torch.randn_like(x)

        noise_weight = schedule.noise_weight[timestep].reshape(-1, 1)
        img_weight = schedule.img_weight[timestep].reshape(-1, 1)

        return x * img_weight + noise * noise_weight, noise, timestep

    def __init__(self, embed_dim, device, hidden_dim=256, num_hidden=1, diffusion_steps=32, greediness=2e-3, target="img", lr=1e-3, n_iters=1000, class_idx=0):
        super().__init__()
        self.n_iters = n_iters
        self.lr = lr
        self.class_idx = class_idx
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.schedule = get_cosine_schedule(diffusion_steps).to(device)
        self.net = MLPDiffusion(embed_dim, hidden_dim, num_hidden, self.schedule.num_steps, device)
        self.net = self.net.to(device)
        self.signal_to_noise_ratio = torch.log(self.schedule.img_weight / self.schedule.noise_weight).clamp(1)
        self.pred_weight = torch.linspace(1, 1 + greediness, steps=self.schedule.num_steps).to(device)
        self.target = target
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-2)

    @torch.enable_grad()
    def fit(self, x):
        self.min = torch.min(x, dim=0).values
        self.max = torch.max(x, dim=0).values
        x = (x - self.min) / (self.max - self.min)

        self.mu = torch.mean(x, dim=0)
        self.std = torch.std(x, dim=0)
        x = (x - self.mu) / self.std

        ds = torch.utils.data.TensorDataset(x)
        self.train()
        loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
        iters = 0
        with tqdm.trange(self.n_iters, desc=f"Training Diffusion [{self.class_idx}]") as pbar:
            for epoch in pbar:
                for data in loader:
                    self.optimizer.zero_grad()
                    inputs = data[0]
                    noisy_inputs, noise, timestep = self.q_function(inputs, self.schedule)
                    outputs = self.net(noisy_inputs, timestep.to(self.device) + 1)
                    if self.target == "noise":
                        loss = torch.mean((outputs - noise) ** 2)
                    else:
                        loss = (self.signal_to_noise_ratio[timestep] * torch.mean((outputs - inputs) ** 2, -1)).mean()

                    loss.backward()
                    self.optimizer.step()
                    pbar.set_postfix(loss=loss.item(), refresh=False)
                    iters += 1
                    if iters >= self.n_iters:
                        break
                if iters >= self.n_iters:
                    break
            pbar.update(1)
        self.eval()

    def sample(self, n_sample, resample_period=1):
        x = torch.randn(n_sample, self.embed_dim).to(self.device)
        noise = torch.randn_like(x)
        if self.target == "noise":
            for i in reversed(range(self.schedule.num_steps)):
                out = self.net(x, torch.tensor([i + 1]).to(self.device))
                x -= self.schedule.beta_t[i] / torch.sqrt(1 - self.schedule.alpha_bar_t[i]) * out
                if i > 0:
                    x += noise * torch.sqrt(self.schedule.beta_t[i])
        else:
            for i in reversed(range(self.schedule.num_steps)):
                if i % resample_period == 0:
                    noise = torch.randn_like(x)
                x = self.net(x, torch.tensor([i + 1]).to(self.device))
                if i > 0:
                    x = x * self.schedule.img_weight[i - 1] + noise * self.schedule.noise_weight[i - 1]

        x = x * self.std + self.mu
        x = x * (self.max - self.min) + self.min
        return x

    def forward(self, n_sample):
        return self.sample(n_sample)
