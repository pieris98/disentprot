#!/bin/bash
# Quick fix for CUDA multiprocessing and deterministic algorithm issues

echo "Applying comprehensive fix for CUDA issues..."

# Update config file to set num_workers to 0
sed -i 's/num_workers: 4/num_workers: 0  # Set to 0 to avoid CUDA multiprocessing issues/' config.yaml

# Set environment variables for CUDA deterministic operations
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "Environment variables set:"
echo "  CUDA_LAUNCH_BLOCKING=1"
echo "  TORCH_USE_CUDA_DSA=1" 
echo "  CUBLAS_WORKSPACE_CONFIG=:4096:8"

echo ""
echo "Fix applied! Now you can run:"
echo "  make test"
echo "  or"
echo "  CUBLAS_WORKSPACE_CONFIG=:4096:8 python train.py --debug --model beta_vae"