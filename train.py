#!/usr/bin/env python3
"""
Training script for protein disentanglement experiments.
"""
import argparse
import os
import sys
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.experiment.runner import ExperimentRunner
from src.data.utils import create_sample_fasta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train protein disentanglement models")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Path to config file"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["beta_vae", "factor_vae", "simclr", "all"],
        default="all",
        help="Model type to train"
    )
    
    parser.add_argument(
        "--fasta", 
        type=str,
        help="Path to FASTA file (overrides config)"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int,
        help="Number of epochs (overrides config)"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int,
        help="Batch size (overrides config)"
    )
    
    parser.add_argument(
        "--gpu", 
        action="store_true",
        help="Use GPU if available"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Run in debug mode with small sample"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load config
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    config = OmegaConf.load(args.config)
    
    # Override config with command line args
    if args.fasta:
        config.data.fasta_path = args.fasta
    if args.epochs:
        config.training.max_epochs = args.epochs
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.gpu and torch.cuda.is_available():
        config.training.accelerator = "gpu"
        config.embedding.device = "cuda"
    
    # Debug mode
    if args.debug:
        config.training.max_epochs = 2
        config.data.batch_size = 4
        logger.info("Running in debug mode")
    
    # Create sample data if needed
    if not os.path.exists(config.data.fasta_path):
        logger.info(f"Creating sample FASTA at {config.data.fasta_path}")
        num_samples = 20 if args.debug else 100
        create_sample_fasta(config.data.fasta_path, num_samples=num_samples)
    
    # Initialize experiment runner
    runner = ExperimentRunner(config)
    
    # Run experiments
    if args.model == "all":
        logger.info("Running all experiments...")
        results = runner.run_all_experiments()
    else:
        logger.info(f"Running {args.model} experiment...")
        results = {args.model: runner.run_experiment(args.model)}
    
    # Print results summary
    print("\n" + "="*60)
    print("TRAINING COMPLETED - RESULTS SUMMARY")
    print("="*60)
    
    for model_type, result in results.items():
        print(f"\n{model_type.upper()}:")
        if "error" in result:
            print(f"  ❌ ERROR: {result['error']}")
        else:
            print(f"  ✅ Training completed successfully")
            if "evaluation" in result:
                eval_metrics = result["evaluation"]
                print(f"  📊 Evaluation metrics:")
                for metric, value in eval_metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"    {metric}: {value:.4f}")
            
            if "best_model_path" in result:
                print(f"  💾 Best model: {result['best_model_path']}")
    
    print(f"\n🎯 Results saved in: {config.experiment.save_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
