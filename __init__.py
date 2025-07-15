# src/__init__.py
"""Protein disentanglement research framework."""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@university.edu"

# src/data/__init__.py
"""Data utilities for protein sequence processing."""

from .utils import (
    ProteinEmbeddingDataset,
    EmbeddingWrapper,
    get_dataloader,
    create_sample_fasta
)

__all__ = [
    "ProteinEmbeddingDataset",
    "EmbeddingWrapper", 
    "get_dataloader",
    "create_sample_fasta"
]

# src/models/__init__.py
"""Model implementations for disentangled representation learning."""

from .vae import BetaVAE, FactorVAE, VAE
from .simclr import SimCLRTrainer, SimCLRHead, nt_xent_loss

__all__ = [
    "BetaVAE",
    "FactorVAE", 
    "VAE",
    "SimCLRTrainer",
    "SimCLRHead",
    "nt_xent_loss"
]

# src/evaluation/__init__.py
"""Evaluation metrics for disentanglement analysis."""

from .metrics import (
    DisentanglementEvaluator,
    beta_vae_metric,
    mig_score,
    sap_score,
    dci_disentanglement,
    evaluate_protein_representations
)

__all__ = [
    "DisentanglementEvaluator",
    "beta_vae_metric",
    "mig_score", 
    "sap_score",
    "dci_disentanglement",
    "evaluate_protein_representations"
]

# src/experiment/__init__.py
"""Experiment orchestration and management."""

from .runner import ExperimentRunner

__all__ = ["ExperimentRunner"]
