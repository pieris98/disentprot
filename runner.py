"""
Experiment runner for protein disentanglement experiments.
"""
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import hydra
from omegaconf import DictConfig
import logging
from typing import Dict, Any, Optional

from ..data.utils import ProteinEmbeddingDataset, EmbeddingWrapper, get_dataloader, create_sample_fasta
from ..models.vae import BetaVAE, FactorVAE
from ..models.simclr import SimCLRTrainer
from ..evaluation.metrics import evaluate_protein_representations

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Main experiment runner for protein disentanglement studies."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.setup_logging()
        self.setup_directories()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def setup_directories(self):
        """Create necessary directories."""
        os.makedirs(self.config.experiment.save_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.config.data.fasta_path), exist_ok=True)
        
    def prepare_data(self) -> tuple:
        """Prepare datasets and dataloaders."""
        logger.info("Preparing data...")
        
        # Create sample data if it doesn't exist
        if not os.path.exists(self.config.data.fasta_path):
            logger.info(f"Creating sample FASTA at {self.config.data.fasta_path}")
            create_sample_fasta(self.config.data.fasta_path, num_samples=100)
        
        # Create embedding dataset
        embed_dataset = ProteinEmbeddingDataset(
            fasta_path=self.config.data.fasta_path,
            model_name=self.config.embedding.model_name,
            aggregation=self.config.embedding.aggregation,
            max_len=self.config.data.max_len,
            device=self.config.embedding.device
        )
        
        # Wrap for Lightning
        wrapped_dataset = EmbeddingWrapper(embed_dataset)
        
        # Get embedding dimension
        sample_emb = wrapped_dataset[0]["emb"]
        emb_dim = sample_emb.shape[0]
        logger.info(f"Embedding dimension: {emb_dim}")
        
        # Create train/val split
        train_size = int(0.8 * len(wrapped_dataset))
        val_size = len(wrapped_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            wrapped_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.experiment.seed)
        )
        
        # Create dataloaders
        train_loader = get_dataloader(
            train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers
        )
        
        val_loader = get_dataloader(
            val_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers
        )
        
        return train_loader, val_loader, emb_dim
    
    def create_model(self, model_type: str, emb_dim: int) -> pl.LightningModule:
        """Create model based on configuration."""
        if model_type == "beta_vae":
            model = BetaVAE(
                input_dim=emb_dim,
                z_dim=self.config.vae.z_dim,
                hidden_dim=self.config.vae.hidden_dim,
                beta=self.config.vae.beta,
                learning_rate=self.config.vae.learning_rate
            )
        elif model_type == "factor_vae":
            model = FactorVAE(
                input_dim=emb_dim,
                z_dim=self.config.vae.z_dim,
                hidden_dim=self.config.vae.hidden_dim,
                beta=self.config.vae.beta,
                learning_rate=self.config.vae.learning_rate
            )
        elif model_type == "simclr":
            model = SimCLRTrainer(
                emb_dim=emb_dim,
                proj_dim=self.config.simclr.proj_dim,
                temperature=self.config.simclr.temperature,
                learning_rate=self.config.simclr.learning_rate
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        logger.info(f"Created {model_type} model")
        return model
    
    def setup_trainer(self, model_name: str) -> pl.Trainer:
        """Setup PyTorch Lightning trainer."""
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                dirpath=os.path.join(self.config.experiment.save_dir, "checkpoints"),
                filename=f"{model_name}-{{epoch:02d}}-{{val_loss:.2f}}",
                monitor="val/loss" if "vae" in model_name else "val/simclr_loss",
                mode="min",
                save_top_k=3,
                save_last=True
            ),
            EarlyStopping(
                monitor="val/loss" if "vae" in model_name else "val/simclr_loss",
                patience=15,
                mode="min",
                verbose=True
            )
        ]
        
        # Logger
        wandb_logger = WandbLogger(
            name=f"{self.config.experiment.name}_{model_name}",
            project="protein-disentanglement",
            save_dir=self.config.experiment.save_dir
        )
        
        # Trainer
        trainer = pl.Trainer(
            max_epochs=self.config.training.max_epochs,
            accelerator=self.config.training.accelerator,
            devices=1,
            callbacks=callbacks,
            logger=wandb_logger,
            log_every_n_steps=self.config.training.log_every_n_steps,
            check_val_every_n_epoch=self.config.training.check_val_every_n_epoch,
            deterministic=True
        )
        
        return trainer
    
    def run_experiment(self, model_type: str) -> Dict[str, Any]:
        """Run a single experiment."""
        logger.info(f"Starting {model_type} experiment...")
        
        # Set seed
        pl.seed_everything(self.config.experiment.seed)
        
        # Prepare data
        train_loader, val_loader, emb_dim = self.prepare_data()
        
        # Create model
        model = self.create_model(model_type, emb_dim)
        
        # Setup trainer
        trainer = self.setup_trainer(model_type)
        
        # Train model
        logger.info("Starting training...")
        trainer.fit(model, train_loader, val_loader)
        
        # Evaluate model
        logger.info("Evaluating model...")
        eval_results = evaluate_protein_representations(
            model, val_loader, device=self.config.embedding.device
        )
        
        # Save results
        results = {
            "model_type": model_type,
            "config": self.config,
            "evaluation": eval_results,
            "best_model_path": trainer.checkpoint_callback.best_model_path
        }
        
        results_path = os.path.join(
            self.config.experiment.save_dir, f"{model_type}_results.pt"
        )
        torch.save(results, results_path)
        
        logger.info(f"Experiment completed. Results saved to {results_path}")
        return results
    
    def run_all_experiments(self) -> Dict[str, Dict[str, Any]]:
        """Run all configured experiments."""
        model_types = ["beta_vae", "factor_vae", "simclr"]
        all_results = {}
        
        for model_type in model_types:
            try:
                results = self.run_experiment(model_type)
                all_results[model_type] = results
            except Exception as e:
                logger.error(f"Error in {model_type} experiment: {e}")
                all_results[model_type] = {"error": str(e)}
        
        # Save combined results
        combined_path = os.path.join(
            self.config.experiment.save_dir, "all_results.pt"
        )
        torch.save(all_results, combined_path)
        
        return all_results


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for experiments."""
    runner = ExperimentRunner(cfg)
    results = runner.run_all_experiments()
    
    # Print summary
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    
    for model_type, result in results.items():
        print(f"\n{model_type.upper()}:")
        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            eval_metrics = result.get("evaluation", {})
            for metric, value in eval_metrics.items():
                print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
