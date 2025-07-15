#!/usr/bin/env python3
"""
Fix CUDA multiprocessing issues by setting appropriate start method.
Run this before importing any CUDA-related modules.
"""
import multiprocessing as mp
import torch
import os
import sys

def fix_cuda_multiprocessing():
    """Fix CUDA multiprocessing issues."""
    try:
        # Set multiprocessing start method to 'spawn' for CUDA compatibility
        if torch.cuda.is_available():
            # Only set if not already set
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method('spawn', force=True)
                print("Set multiprocessing start method to 'spawn' for CUDA compatibility")
            
            # Set environment variables for better CUDA multiprocessing
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            os.environ['TORCH_USE_CUDA_DSA'] = '1'
            
            # Recommended PyTorch settings for Tensor Cores
            torch.set_float32_matmul_precision('medium')
            
    except RuntimeError as e:
        print(f"Warning: Could not set multiprocessing start method: {e}")
        print("This may cause issues with CUDA and multiprocessing.")

if __name__ == "__main__":
    fix_cuda_multiprocessing()
    print("CUDA multiprocessing fix applied!")