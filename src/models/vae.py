"""
β-VAE and FactorVAE implementations for protein representation disentanglement.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class VAE(nn.Module):
    """Variational Autoencoder for protein embeddings."""
    
    def __init__(self, input_dim: int, z_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class BetaVAE(pl.LightningModule):
    """β-VAE Lightning module for disentangled representation learning."""
    
    def __init__(
        self,
        input_dim: int,
        z_dim: int = 32,
        hidden_dim: int = 256,
        beta: float = 4.0,
        learning_rate: float = 1e-3,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = VAE(input_dim, z_dim, hidden_dim)
        self.beta = beta
        self.learning_rate = learning_rate
        
        # For logging
        self.train_losses = []
        self.val_losses = []

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(x)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute β-VAE loss components."""
        x = batch["emb"]
        recon, mu, logvar = self.forward(x)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        losses = self.compute_loss(batch)
        
        # Log losses
        for name, loss in losses.items():
            self.log(f"train/{name}", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        self.train_losses.append(losses["loss"].item())
        return losses["loss"]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        losses = self.compute_loss(batch)
        
        # Log losses
        for name, loss in losses.items():
            self.log(f"val/{name}", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self.val_losses.append(losses["loss"].item())
        return losses["loss"]

    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def encode_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch to latent space (for analysis)."""
        self.eval()
        with torch.no_grad():
            mu, _ = self.model.encode(x)
        return mu

    def sample(self, num_samples: int = 10) -> torch.Tensor:
        """Sample from the prior."""
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.model.z_dim).to(self.device)
            samples = self.model.decode(z)
        return samples


class FactorVAE(BetaVAE):
    """FactorVAE implementation with discriminator for improved disentanglement."""
    
    def __init__(
        self,
        input_dim: int,
        z_dim: int = 32,
        hidden_dim: int = 256,
        beta: float = 4.0,
        gamma: float = 6.4,
        learning_rate: float = 1e-3,
        **kwargs
    ):
        super().__init__(input_dim, z_dim, hidden_dim, beta, learning_rate, **kwargs)
        self.gamma = gamma
        
        # Discriminator for FactorVAE
        self.discriminator = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 2),  # Binary classification
        )
        
        # Separate optimizers
        self.automatic_optimization = False

    def discriminator_loss(self, z_true: torch.Tensor, z_perm: torch.Tensor) -> torch.Tensor:
        """Compute discriminator loss."""
        # Real (permuted) samples should be classified as 1
        # Fake (true) samples should be classified as 0
        d_real = self.discriminator(z_perm)
        d_fake = self.discriminator(z_true)
        
        d_loss_real = F.cross_entropy(d_real, torch.ones(z_perm.size(0), dtype=torch.long, device=self.device))
        d_loss_fake = F.cross_entropy(d_fake, torch.zeros(z_true.size(0), dtype=torch.long, device=self.device))
        
        return (d_loss_real + d_loss_fake) / 2

    def generator_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Compute generator (VAE) adversarial loss."""
        d_fake = self.discriminator(z)
        return F.cross_entropy(d_fake, torch.ones(z.size(0), dtype=torch.long, device=self.device))

    def permute_dims(self, z: torch.Tensor) -> torch.Tensor:
        """Permute dimensions of latent codes."""
        z_perm = z.clone()
        for i in range(z.size(1)):
            perm = torch.randperm(z.size(0))
            z_perm[:, i] = z_perm[perm, i]
        return z_perm

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Training step for FactorVAE."""
        vae_opt, disc_opt = self.optimizers()
        
        x = batch["emb"]
        
        # ============
        # Train VAE
        # ============
        recon, mu, logvar = self.forward(x)
        z = self.model.reparameterize(mu, logvar)
        
        # Standard VAE losses
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Adversarial loss
        tc_loss = self.generator_loss(z)
        
        vae_loss = recon_loss + self.beta * kl_loss + self.gamma * tc_loss
        
        vae_opt.zero_grad()
        self.manual_backward(vae_loss)
        vae_opt.step()
        
        # ==================
        # Train Discriminator
        # ==================
        z_perm = self.permute_dims(z.detach())
        disc_loss = self.discriminator_loss(z.detach(), z_perm)
        
        disc_opt.zero_grad()
        self.manual_backward(disc_loss)
        disc_opt.step()
        
        # Logging
        self.log("train/vae_loss", vae_loss, on_step=True, on_epoch=True)
        self.log("train/recon_loss", recon_loss, on_step=True, on_epoch=True)
        self.log("train/kl_loss", kl_loss, on_step=True, on_epoch=True)
        self.log("train/tc_loss", tc_loss, on_step=True, on_epoch=True)
        self.log("train/disc_loss", disc_loss, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        """Configure optimizers for VAE and discriminator."""
        vae_optimizer = torch.optim.Adam(
            list(self.model.parameters()), lr=self.learning_rate
        )
        disc_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.learning_rate
        )
        
        return [vae_optimizer, disc_optimizer]
