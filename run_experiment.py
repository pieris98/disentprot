#!/usr/bin/env python3
"""
Minimal test script that skips evaluation for debugging.
"""
import os
import sys
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
import multiprocessing as mp

# Fix CUDA multiprocessing issues
if torch.cuda.is_available():
    try:
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method('spawn', force=True)
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.set_float32_matmul_precision('medium')
    except RuntimeError:
        pass

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.experiment.runner import ExperimentRunner
from src.data.utils import create_sample_fasta

def main():
    """Minimal test without evaluation."""
    print("Running minimal test (training only, no evaluation)...")
    
    # Load config
    config = OmegaConf.load("config.yaml")
    
    # Debug settings
    config.training.max_epochs = 2
    config.data.batch_size = 4
    config.vae.z_dim = 8
    config.vae.hidden_dim = 64
    
    # Create sample data
    if not os.path.exists(config.data.fasta_path):
        create_sample_fasta(config.data.fasta_path, num_samples=10)
    
    # Initialize runner
    runner = ExperimentRunner(config)
    
    # Just run training part
    pl.seed_everything(config.experiment.seed, workers=True)
    
    train_loader, val_loader, emb_dim = runner.prepare_data()
    model = runner.create_model("beta_vae", emb_dim)
    trainer = runner.setup_trainer("beta_vae")
    
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    print("✅ Training completed successfully!")
    print(f"📁 Checkpoints saved in: experiments/checkpoints/")
    print("🎯 Skipping evaluation for this test")

if __name__ == "__main__":
    main()