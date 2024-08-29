from typing import Tuple

import torch
import torch.nn as nn
import tqdm

try:
    import wandb
except ImportError:
    wandb = None


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        '''
        Args:
            input_dim: A integer indicating the size of input dimension.
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent dimension.
        '''
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )

    def forward(self, x: torch.Tensor):
        hidden = self.encoder(x)
        z_mu, z_logvar = hidden[:, :self.latent_dim], hidden[:, self.latent_dim:]
        return z_mu, z_logvar


class Decoder(nn.Module):
    def __init__(self, output_dim: int, hidden_dim: int, latent_dim: int):
        '''
        Args:
            latent_dim: A integer indicating the latent size.
            hidden_dim: A integer indicating the size of hidden dimension.
            output_dim: A integer indicating the output dimension.
        '''
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor):
        return self.decoder(x)


class VariationalAutoEncoderModel(torch.nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int,
                 lr: float, class_idx: int, n_iters: int = 100) -> None:
        super().__init__()
        self.n_iters = n_iters
        self.lr = lr
        self.class_idx = class_idx
        self.vae = VariationalAutoEncoder(input_dim, hidden_dim, latent_dim, self.n_iters)

    def fit(self, x: torch.Tensor) -> None:
        self.vae.fit(x, n_iters=self.n_iters, lr=self.lr, class_idx=self.class_idx)

    def sample(self, n_sample: int) -> torch.Tensor:
        return self.vae.sample(n_sample)[0]

    def forward(self, n_sample: int) -> torch.Tensor:
        return self.sample(n_sample)

    def reconstruct(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.vae.normalize(x)
        x_rec, z_mu, z_logvar = self.vae(x)
        x_rec = self.vae.denormalize(x_rec)
        return x_rec, z_mu, z_logvar


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, n_iters: int) -> None:
        super().__init__()
        self.n_iters = n_iters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.elbo = ELBO()
        self.register_buffer("mu", torch.zeros(input_dim))
        self.register_buffer("std", torch.ones(input_dim))
        self.register_buffer("min", torch.zeros(input_dim))
        self.register_buffer("max", torch.ones(input_dim))
        self.enc = Encoder(input_dim, hidden_dim, latent_dim)
        self.dec = Decoder(input_dim, hidden_dim, latent_dim)

    def reparameterization_trick(self, z_mu: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(z_mu) * torch.exp(z_logvar) + z_mu

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mu, z_logvar = self.enc(x)
        z_post = self.reparameterization_trick(z_mu, z_logvar)

        return self.dec(z_post), z_mu, z_logvar

    def sample(self, num_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        z = torch.randn(num_samples, self.enc.latent_dim, device=device)

        x = self.dec(z)
        x = self.denormalize(x)
        return x, z

    @torch.enable_grad()
    def fit(self, x: torch.Tensor, n_iters: int, lr: float, class_idx: int) -> None:
        self.min = torch.min(x, dim=0).values
        self.max = torch.max(x, dim=0).values
        x = (x - self.min) / (self.max - self.min)

        self.mu = torch.mean(x, dim=0)
        self.std = torch.std(x, dim=0)
        x = (x - self.mu) / self.std
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=n_iters // 10, T_mult=2)
        self.train()
        loader = torch.utils.data.DataLoader(x, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
        with tqdm.trange(n_iters, desc=f"Training VAE [{class_idx}]") as t:
            for _ in t:
                for batch in loader:
                    if len(batch) == 1:
                        continue
                    optimizer.zero_grad()
                    predicted, z_mu, z_logvar = self.forward(batch)
                    loss = self.elbo(batch, predicted, z_mu, z_logvar) / len(batch)
                    loss.backward()
                    optimizer.step()
                    t.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'], refresh=False)
                sched.step()
        self.eval()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.min) / (self.max - self.min)
        return (x - self.mu) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.std + self.mu
        return x * (self.max - self.min) + self.min


def gaussian_nll(mu: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gaussian_nll_loss(x, mu, torch.ones_like(mu), full=True, reduction='sum')


class ELBO(nn.Module):
    def __init__(self):
        super().__init__()

    def compute_rec_error(self, x: torch.Tensor, x_rec: torch.Tensor):
        return gaussian_nll(x_rec, x)

    def compute_kl(self, z_mu: torch.Tensor, z_logvar: torch.Tensor):
        return 0.5 * torch.sum(torch.exp(z_logvar) + z_mu**2 - 1.0 - z_logvar)

    def forward(self, x: torch.Tensor, x_rec: torch.Tensor,
                z_mu: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:

        recon_loss = self.compute_rec_error(x, x_rec)

        kl_loss = self.compute_kl(z_mu, z_logvar)

        if wandb.run:
            wandb.log({"reconstruction loss": recon_loss, "kl loss": kl_loss})

        return recon_loss + kl_loss
