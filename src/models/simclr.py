"""
SimCLR implementation for contrastive learning on protein embeddings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class SimCLRHead(nn.Module):
    """Projection head for SimCLR."""
    
    def __init__(self, emb_dim: int, proj_dim: int = 64, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = emb_dim
            
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    Normalized Temperature-scaled Cross Entropy Loss for SimCLR.
    
    Args:
        z_i: Projections of first augmented view (B, D)
        z_j: Projections of second augmented view (B, D)
        temperature: Temperature parameter
        
    Returns:
        NT-Xent loss
    """
    batch_size = z_i.size(0)
    
    # Normalize projections
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    
    # Concatenate projections
    z = torch.cat([z_i, z_j], dim=0)  # (2B, D)
    
    # Compute similarity matrix
    sim_matrix = torch.mm(z, z.t()) / temperature  # (2B, 2B)
    
    # Create mask to exclude self-similarity
    mask = torch.eye(2 * batch_size, dtype=bool, device=z.device)
    sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
    
    # Positive pairs: (i, i+B) and (i+B, i)
    pos_indices = torch.arange(batch_size, device=z.device)
    pos_sim_i = sim_matrix[pos_indices, pos_indices + batch_size]  # z_i with z_j
    pos_sim_j = sim_matrix[pos_indices + batch_size, pos_indices]  # z_j with z_i
    
    # Compute loss for both directions
    exp_sim_matrix = torch.exp(sim_matrix)
    
    # For z_i
    sum_exp_i = exp_sim_matrix[pos_indices].sum(dim=1) - torch.exp(sim_matrix[pos_indices, pos_indices])
    loss_i = -pos_sim_i + torch.log(sum_exp_i)
    
    # For z_j
    sum_exp_j = exp_sim_matrix[pos_indices + batch_size].sum(dim=1) - torch.exp(sim_matrix[pos_indices + batch_size, pos_indices + batch_size])
    loss_j = -pos_sim_j + torch.log(sum_exp_j)
    
    return (loss_i + loss_j).mean()


class SimCLRTrainer(pl.LightningModule):
    """SimCLR trainer for contrastive learning on protein embeddings."""
    
    def __init__(
        self,
        emb_dim: int,
        proj_dim: int = 64,
        temperature: float = 0.5,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.encoder = nn.Identity()  # Embeddings are precomputed
        self.projection_head = SimCLRHead(emb_dim, proj_dim)
        
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and projection head."""
        h = self.encoder(x)
        z = self.projection_head(h)
        return z

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step for SimCLR."""
        # Expect batch to contain two augmented views
        if 'emb1' in batch and 'emb2' in batch:
            z1 = self.forward(batch['emb1'])
            z2 = self.forward(batch['emb2'])
        else:
            # If only one view, create augmented version (e.g., with noise)
            emb = batch['emb']
            z1 = self.forward(emb)
            # Simple augmentation: add Gaussian noise
            emb_aug = emb + 0.1 * torch.randn_like(emb)
            z2 = self.forward(emb_aug)
        
        loss = nt_xent_loss(z1, z2, self.temperature)
        
        self.log('train/simclr_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        # Similar to training step
        if 'emb1' in batch and 'emb2' in batch:
            z1 = self.forward(batch['emb1'])
            z2 = self.forward(batch['emb2'])
        else:
            emb = batch['emb']
            z1 = self.forward(emb)
            emb_aug = emb + 0.1 * torch.randn_like(emb)
            z2 = self.forward(emb_aug)
        
        loss = nt_xent_loss(z1, z2, self.temperature)
        
        self.log('val/simclr_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.learning_rate * 0.01
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def get_representations(self, dataloader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract representations and projections for analysis."""
        self.eval()
        representations = []
        projections = []
        
        with torch.no_grad():
            for batch in dataloader:
                emb = batch['emb'].to(self.device)
                h = self.encoder(emb)
                z = self.projection_head(h)
                
                representations.append(h.cpu())
                projections.append(z.cpu())
        
        return torch.cat(representations), torch.cat(projections)