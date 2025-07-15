#!/usr/bin/env python3
"""
Monitor training progress and provide real-time updates.
"""
import os
import time
import glob
import torch
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

def monitor_checkpoints(checkpoint_dir="experiments_full/checkpoints"):
    """Monitor checkpoint directory for new models."""
    print(f"📁 Monitoring checkpoints in: {checkpoint_dir}")
    
    seen_files = set()
    
    while True:
        try:
            # Check for new checkpoints
            checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
            
            for ckpt in checkpoints:
                if ckpt not in seen_files:
                    seen_files.add(ckpt)
                    
                    # Get file stats
                    stat = os.stat(ckpt)
                    size_mb = stat.st_size / (1024 * 1024)
                    modified = datetime.fromtimestamp(stat.st_mtime)
                    
                    print(f"✅ New checkpoint: {os.path.basename(ckpt)}")
                    print(f"   Size: {size_mb:.1f} MB")
                    print(f"   Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}")
                    print()
            
            # Check WandB logs
            wandb_dir = "experiments_full/wandb"
            if os.path.exists(wandb_dir):
                runs = glob.glob(os.path.join(wandb_dir, "run-*"))
                if runs:
                    latest_run = max(runs, key=os.path.getctime)
                    print(f"📊 Latest WandB run: {os.path.basename(latest_run)}")
            
            time.sleep(30)  # Check every 30 seconds
            
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped.")
            break
        except Exception as e:
            print(f"❌ Error monitoring: {e}")
            time.sleep(30)


def show_training_progress(results_dir="experiments_full"):
    """Show training progress from saved results."""
    print(f"📊 Training Progress Summary")
    print("=" * 50)
    
    # Look for result files
    result_files = glob.glob(os.path.join(results_dir, "*_results.pt"))
    
    if not result_files:
        print("No results found yet. Training may still be in progress.")
        return
    
    for result_file in result_files:
        try:
            results = torch.load(result_file, map_location="cpu")
            model_type = results.get("model_type", "unknown")
            
            print(f"\n🔬 {model_type.upper()} Results:")
            print("-" * 30)
            
            # Show evaluation metrics
            if "evaluation" in results:
                eval_results = results["evaluation"]
                for metric, value in eval_results.items():
                    if isinstance(value, (int, float)):
                        print(f"  {metric}: {value:.4f}")
            
            # Show best model path
            if "best_model_path" in results:
                print(f"  Best model: {results['best_model_path']}")
                
        except Exception as e:
            print(f"❌ Error reading {result_file}: {e}")


def plot_training_curves(checkpoint_dir="experiments_full/checkpoints"):
    """Plot training curves from TensorBoard logs."""
    print("📈 Generating training curves...")
    
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        # Find TensorBoard log files
        tb_logs = glob.glob(os.path.join(checkpoint_dir, "**", "events.out.tfevents.*"), recursive=True)
        
        if not tb_logs:
            print("No TensorBoard logs found.")
            return
        
        # Plot training curves
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        for log_file in tb_logs:
            try:
                event_acc = EventAccumulator(log_file)
                event_acc.Reload()
                
                # Get available scalars
                scalars = event_acc.Tags()['scalars']
                
                # Plot training loss
                if 'train/loss' in scalars:
                    train_loss = event_acc.Scalars('train/loss')
                    steps = [s.step for s in train_loss]
                    values = [s.value for s in train_loss]
                    axes[0, 0].plot(steps, values, label='Train Loss')
                
                # Plot validation loss
                if 'val/loss' in scalars:
                    val_loss = event_acc.Scalars('val/loss')
                    steps = [s.step for s in val_loss]
                    values = [s.value for s in val_loss]
                    axes[0, 1].plot(steps, values, label='Val Loss')
                
                # Plot KL loss
                if 'train/kl_loss' in scalars:
                    kl_loss = event_acc.Scalars('train/kl_loss')
                    steps = [s.step for s in kl_loss]
                    values = [s.value for s in kl_loss]
                    axes[1, 0].plot(steps, values, label='KL Loss')
                
                # Plot reconstruction loss
                if 'train/recon_loss' in scalars:
                    recon_loss = event_acc.Scalars('train/recon_loss')
                    steps = [s.step for s in recon_loss]
                    values = [s.value for s in recon_loss]
                    axes[1, 1].plot(steps, values, label='Recon Loss')
                
            except Exception as e:
                print(f"Error processing {log_file}: {e}")
        
        # Customize plots
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        axes[1, 0].set_title('KL Divergence')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('KL Loss')
        axes[1, 0].legend()
        
        axes[1, 1].set_title('Reconstruction Loss')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Recon Loss')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Training curves saved as training_curves.png")
        
    except ImportError:
        print("❌ TensorBoard not installed. Install with: pip install tensorboard")
    except Exception as e:
        print(f"❌ Error plotting curves: {e}")


def main():
    """Main monitoring function."""
    parser = argparse.ArgumentParser(description="Monitor training progress")
    parser.add_argument("--action", choices=["monitor", "progress", "plot"], 
                       default="progress", help="Action to perform")
    parser.add_argument("--checkpoint_dir", default="experiments_full/checkpoints",
                       help="Checkpoint directory")
    parser.add_argument("--results_dir", default="experiments_full",
                       help="Results directory")
    
    args = parser.parse_args()
    
    if args.action == "monitor":
        monitor_checkpoints(args.checkpoint_dir)
    elif args.action == "progress":
        show_training_progress(args.results_dir)
    elif args.action == "plot":
        plot_training_curves(args.checkpoint_dir)


if __name__ == "__main__":
    main()