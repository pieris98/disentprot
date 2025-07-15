#!/usr/bin/env python3
"""
Evaluation script for trained protein disentanglement models.
"""
import argparse
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.utils import ProteinEmbeddingDataset, EmbeddingWrapper, get_dataloader
from src.models.vae import BetaVAE, FactorVAE
from src.models.simclr import SimCLRTrainer
from src.evaluation.metrics import evaluate_protein_representations, DisentanglementEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate protein disentanglement models")
    
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Path to config file"
    )
    
    parser.add_argument(
        "--fasta", 
        type=str,
        help="Path to FASTA file for evaluation"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="evaluation_results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--plot", 
        action="store_true",
        help="Generate visualization plots"
    )
    
    parser.add_argument(
        "--model_type", 
        type=str, 
        choices=["beta_vae", "factor_vae", "simclr"],
        help="Model type (auto-detected if not specified)"
    )
    
    return parser.parse_args()


def load_model(checkpoint_path: str, model_type: str = None):
    """Load model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Auto-detect model type from checkpoint if not specified
    if model_type is None:
        state_dict_keys = checkpoint.get("state_dict", {}).keys()
        if any("discriminator" in key for key in state_dict_keys):
            model_type = "factor_vae"
        elif any("projection_head" in key for key in state_dict_keys):
            model_type = "simclr"
        else:
            model_type = "beta_vae"
        
        logger.info(f"Auto-detected model type: {model_type}")
    
    # Get hyperparameters
    hparams = checkpoint.get("hyper_parameters", {})
    
    # Create model
    if model_type == "beta_vae":
        model = BetaVAE.load_from_checkpoint(checkpoint_path)
    elif model_type == "factor_vae":
        model = FactorVAE.load_from_checkpoint(checkpoint_path)
    elif model_type == "simclr":
        model = SimCLRTrainer.load_from_checkpoint(checkpoint_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.eval()
    logger.info(f"Loaded {model_type} model from {checkpoint_path}")
    
    return model, model_type


def extract_representations(model, dataloader, model_type: str, device: str = "cuda"):
    """Extract representations from model."""
    model.to(device)
    model.eval()
    
    representations = []
    original_embeddings = []
    
    with torch.no_grad():
        for batch in dataloader:
            emb = batch["emb"].to(device)
            original_embeddings.append(emb.cpu())
            
            if model_type in ["beta_vae", "factor_vae"]:
                # Extract latent representations
                mu, _ = model.model.encode(emb)
                representations.append(mu.cpu())
            elif model_type == "simclr":
                # Extract projections
                proj = model.projection_head(emb)
                representations.append(proj.cpu())
    
    return torch.cat(representations), torch.cat(original_embeddings)


def plot_latent_space(representations: torch.Tensor, save_path: str):
    """Plot latent space using t-SNE and PCA."""
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    repr_np = representations.cpu().numpy()
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(repr_np)
    
    axes[0].scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6)
    axes[0].set_title(f'PCA of Latent Space\nExplained Variance: {pca.explained_variance_ratio_.sum():.3f}')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    
    # t-SNE
    if len(repr_np) > 1:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(repr_np)-1))
        tsne_result = tsne.fit_transform(repr_np)
        
        axes[1].scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.6)
        axes[1].set_title('t-SNE of Latent Space')
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Latent space plot saved to {save_path}")


def plot_dimension_analysis(representations: torch.Tensor, save_path: str):
    """Analyze and plot individual latent dimensions."""
    repr_np = representations.cpu().numpy()
    n_dims = repr_np.shape[1]
    
    # Calculate statistics for each dimension
    dim_stats = {
        'mean': np.mean(repr_np, axis=0),
        'std': np.std(repr_np, axis=0),
        'min': np.min(repr_np, axis=0),
        'max': np.max(repr_np, axis=0)
    }
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Dimension statistics
    dims = range(n_dims)
    
    axes[0, 0].bar(dims, dim_stats['mean'])
    axes[0, 0].set_title('Mean Activation per Dimension')
    axes[0, 0].set_xlabel('Latent Dimension')
    axes[0, 0].set_ylabel('Mean')
    
    axes[0, 1].bar(dims, dim_stats['std'])
    axes[0, 1].set_title('Standard Deviation per Dimension')
    axes[0, 1].set_xlabel('Latent Dimension')
    axes[0, 1].set_ylabel('Std Dev')
    
    # Distribution of first few dimensions
    n_show = min(5, n_dims)
    for i in range(n_show):
        axes[1, 0].hist(repr_np[:, i], alpha=0.6, label=f'Dim {i}', bins=20)
    axes[1, 0].set_title(f'Distribution of First {n_show} Dimensions')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Correlation heatmap (for smaller dimensions)
    if n_dims <= 20:
        corr_matrix = np.corrcoef(repr_np.T)
        im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 1].set_title('Dimension Correlation Matrix')
        axes[1, 1].set_xlabel('Latent Dimension')
        axes[1, 1].set_ylabel('Latent Dimension')
        plt.colorbar(im, ax=axes[1, 1])
    else:
        axes[1, 1].text(0.5, 0.5, f'Too many dimensions ({n_dims})\nto show correlation matrix', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Dimension Analysis')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Dimension analysis plot saved to {save_path}")
    
    return dim_stats


def generate_evaluation_report(
    model_type: str, 
    metrics: dict, 
    dim_stats: dict, 
    save_path: str
):
    """Generate a comprehensive evaluation report."""
    
    report = f"""
# Protein Disentanglement Model Evaluation Report

## Model Information
- **Model Type**: {model_type.upper()}
- **Evaluation Date**: {torch.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Disentanglement Metrics

"""
    
    # Add metrics
    for metric_name, value in metrics.items():
        if isinstance(value, (int, float)) and not np.isnan(value):
            report += f"- **{metric_name.replace('_', ' ').title()}**: {value:.4f}\n"
    
    report += f"""

## Latent Space Analysis

### Dimension Statistics
- **Number of Dimensions**: {len(dim_stats['mean'])}
- **Mean Activation Range**: [{np.min(dim_stats['mean']):.3f}, {np.max(dim_stats['mean']):.3f}]
- **Std Dev Range**: [{np.min(dim_stats['std']):.3f}, {np.max(dim_stats['std']):.3f}]
- **Active Dimensions** (std > 0.1): {np.sum(dim_stats['std'] > 0.1)}/{len(dim_stats['std'])}

### Interpretation Guidelines

#### β-VAE Metrics
- **Beta VAE Score**: Measures how well individual factors can be predicted from latent dimensions
  - Range: [0, 1], Higher is better
  - Good: > 0.7, Fair: 0.5-0.7, Poor: < 0.5

#### MIG (Mutual Information Gap)
- **MIG Score**: Measures mutual information gap between most and second-most informative latent variable
  - Range: [0, 1], Higher is better
  - Good: > 0.3, Fair: 0.1-0.3, Poor: < 0.1

#### SAP (Separated Attribute Predictability)
- **SAP Score**: Measures how well separated attributes can be predicted
  - Range: [0, 1], Higher is better
  - Good: > 0.5, Fair: 0.2-0.5, Poor: < 0.2

#### DCI Metrics
- **Disentanglement**: How concentrated each latent dimension is on a single factor
- **Completeness**: How well each factor is captured by a single latent dimension
- **Informativeness**: How well factors can be predicted from latent variables

### Recommendations

"""
    
    # Add recommendations based on metrics
    if metrics.get('beta_vae', 0) > 0.7:
        report += "✅ **Good disentanglement** - Model successfully separates factors\n"
    elif metrics.get('beta_vae', 0) > 0.5:
        report += "⚠️ **Moderate disentanglement** - Consider tuning β parameter or architecture\n"
    else:
        report += "❌ **Poor disentanglement** - Model struggles to separate factors\n"
    
    active_ratio = np.sum(dim_stats['std'] > 0.1) / len(dim_stats['std'])
    if active_ratio < 0.5:
        report += "⚠️ **Many inactive dimensions** - Consider reducing latent dimensionality\n"
    elif active_ratio > 0.9:
        report += "⚠️ **All dimensions active** - May need stronger regularization\n"
    else:
        report += "✅ **Good dimension utilization**\n"
    
    # Save report
    with open(save_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Evaluation report saved to {save_path}")


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config
    config = OmegaConf.load(args.config) if os.path.exists(args.config) else OmegaConf.create({})
    
    # Load model
    try:
        model, model_type = load_model(args.checkpoint, args.model_type)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Prepare data
    fasta_path = args.fasta or config.get('data', {}).get('fasta_path', 'data/my_proteins.fasta')
    
    if not os.path.exists(fasta_path):
        logger.error(f"FASTA file not found: {fasta_path}")
        sys.exit(1)
    
    # Create dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_dataset = ProteinEmbeddingDataset(
        fasta_path=fasta_path,
        model_name=config.get('embedding', {}).get('model_name', 'esm2_t33_650M_UR50D'),
        aggregation=config.get('embedding', {}).get('aggregation', 'mean'),
        device=device
    )
    
    wrapped_dataset = EmbeddingWrapper(embed_dataset)
    dataloader = get_dataloader(wrapped_dataset, batch_size=32, shuffle=False)
    
    # Extract representations
    logger.info("Extracting representations...")
    representations, original_embeddings = extract_representations(
        model, dataloader, model_type, device
    )
    
    # Evaluate disentanglement
    logger.info("Computing disentanglement metrics...")
    evaluator = DisentanglementEvaluator()
    metrics = evaluator.evaluate(representations)
    
    # Generate plots if requested
    dim_stats = {}
    if args.plot:
        logger.info("Generating plots...")
        
        # Latent space visualization
        plot_latent_space(
            representations, 
            os.path.join(args.output_dir, f"{model_type}_latent_space.png")
        )
        
        # Dimension analysis
        dim_stats = plot_dimension_analysis(
            representations,
            os.path.join(args.output_dir, f"{model_type}_dimensions.png")
        )
    
    # Generate report
    generate_evaluation_report(
        model_type, 
        metrics, 
        dim_stats,
        os.path.join(args.output_dir, f"{model_type}_report.md")
    )
    
    # Save detailed results
    results = {
        'model_type': model_type,
        'metrics': metrics,
        'representations': representations,
        'dimension_stats': dim_stats,
        'checkpoint_path': args.checkpoint
    }
    
    torch.save(results, os.path.join(args.output_dir, f"{model_type}_evaluation.pt"))
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETED")
    print("="*60)
    print(f"Model Type: {model_type.upper()}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output Directory: {args.output_dir}")
    
    print("\nDisentanglement Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, (int, float)) and not np.isnan(value):
            print(f"  {metric}: {value:.4f}")
    
    if dim_stats:
        print(f"\nLatent Space Analysis:")
        print(f"  Dimensions: {len(dim_stats['mean'])}")
        print(f"  Active dimensions: {np.sum(dim_stats['std'] > 0.1)}")
        print(f"  Mean activation range: [{np.min(dim_stats['mean']):.3f}, {np.max(dim_stats['mean']):.3f}]")
    
    print("="*60)


if __name__ == "__main__":
    main()