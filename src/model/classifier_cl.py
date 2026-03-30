import torch
import torch.nn as nn
import torch.nn.functional as F
from neural_ode import NeuralODE


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.layer1(x)
        out = F.gelu(out)
        out = self.layer2(out)
        return out


class Classifier(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 32,
        mlp_hidden: int = 128,
        num_classes: int = 2,
        device: str = "cpu",
        ode_steps: int = 100,
    ):
        super().__init__()
        self.device = device
        self.ode_steps = ode_steps

        # Both ODEs are trainable — no freeze
        self.real_ode = NeuralODE(input_dim=latent_dim, device=device, hidden=hidden_dim)
        self.fake_ode = NeuralODE(input_dim=latent_dim, device=device, hidden=hidden_dim)

        self.mlp = MLP(
            input_dim=latent_dim * 2,
            hidden_dim=mlp_hidden,
            output_dim=num_classes,
        )

    def differentiable_inv_sample(self, ode, z, N):
        """
        Differentiable version of inv_sample — mirrors the logic in NeuralODE
        but WITHOUT @torch.no_grad(), so gradients flow back through ode weights.

        Runs reverse Euler: z_{t-dt} = z_t - f(z_t, 1-t) * dt
        which is the exact inverse of the forward sample direction.
        """
        dt = 1.0 / N
        for i in range(N):
            t = torch.ones((z.shape[0], 1), device=self.device) * i / N
            pred = ode(z, 1 - t)   # note: 1-t mirrors inv_sample in NeuralODE
            z = z - pred * dt
        return z

    def forward(self, z):
        # z is already in uniform space, shape (B, latent_dim)

        real_proj = self.differentiable_inv_sample(self.real_ode, z, N=self.ode_steps)  # (B, latent_dim)
        fake_proj = self.differentiable_inv_sample(self.fake_ode, z, N=self.ode_steps)  # (B, latent_dim)

        combined = torch.cat([real_proj, fake_proj], dim=1)   # (B, latent_dim * 2)
        logits = self.mlp(combined)                           # (B, num_classes)
        return logits