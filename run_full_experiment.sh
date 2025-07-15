#!/bin/bash
# Run full-scale protein disentanglement experiment

set -e

echo "🧬 PROTEIN DISENTANGLEMENT EXPERIMENT"
echo "====================================="

# Set environment variables
export CUDA_LAUNCH_BLOCKING=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_USE_CUDA_DSA=1

# Parse arguments
DATASET=${1:-"sample"}
MODEL=${2:-"all"}
EPOCHS=${3:-100}

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Model: $MODEL"
echo "  Epochs: $EPOCHS"
echo ""

# Create data directory
mkdir -p data experiments_full

# Download appropriate dataset
echo "📥 Downloading dataset..."
case $DATASET in
    "sample")
        python download_protein_data.py --dataset sample --num_sequences 5000 --output data/real_proteins.fasta
        ;;
    "pfam")
        python download_protein_data.py --dataset pfam --output data/pfam_proteins.fasta
        ;;
    "covid")
        python download_protein_data.py --dataset covid --output data/covid_proteins.fasta
        ;;
    "uniref")
        python download_protein_data.py --dataset uniref --num_sequences 10000 --output data/uniref_proteins.fasta
        ;;
    *)
        echo "❌ Unknown dataset: $DATASET"
        echo "Available datasets: sample, pfam, covid, uniref"
        exit 1
        ;;
esac

# Set FASTA path based on dataset
case $DATASET in
    "sample")
        FASTA_PATH="data/real_proteins.fasta"
        ;;
    "pfam")
        FASTA_PATH="data/pfam_proteins.fasta"
        ;;
    "covid")
        FASTA_PATH="data/covid_proteins.fasta"
        ;;
    "uniref")
        FASTA_PATH="data/uniref_proteins.fasta"
        ;;
esac

echo "✅ Dataset ready: $FASTA_PATH"

# Run the experiment
echo ""
echo "🚀 Starting training..."
echo "This may take several hours depending on your hardware."
echo ""

python train.py \
    --config config_full.yaml \
    --model $MODEL \
    --fasta $FASTA_PATH \
    --epochs $EPOCHS \
    --gpu

echo ""
echo "✅ EXPERIMENT COMPLETED!"
echo "📁 Results saved in: experiments_full/"
echo "📊 View logs at: https://wandb.ai/pompos002/protein-disentanglement"
echo ""
echo "🔬 Next steps:"
echo "  1. Evaluate models: python evaluate.py --checkpoint experiments_full/checkpoints/best.ckpt"
echo "  2. Compare results: ls experiments_full/*_results.pt"
echo "  3. Generate plots: python evaluate.py --checkpoint experiments_full/checkpoints/best.ckpt --plot"