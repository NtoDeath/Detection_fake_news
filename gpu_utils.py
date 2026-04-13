"""
GPU/Device Management Utilities for All Training Scripts

Provides robust device detection and setup for PyTorch training.
Ensures GPU (CUDA) is used when available, with automatic fallback to CPU.

Usage:
------
from gpu_utils import setup_training_device, get_device_info

device = setup_training_device()  # Returns torch.device
info = get_device_info()  # Returns device info dict
"""

import torch
from typing import Dict, Tuple


def setup_training_device(verbose: bool = True) -> torch.device:
    """
    Detect and configure the best available device for training.
    
    Priority order:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon)
    3. CPU (fallback)
    
    Parameters
    ----------
    verbose : bool
        If True, print device information and test results.
    
    Returns
    -------
    torch.device
        The device to use for training.
    
    Raises
    ------
    RuntimeError
        If no device is available (should not happen as CPU is always available).
    """
    
    # Try CUDA first
    if torch.cuda.is_available():
        try:
            # Test CUDA accessibility
            _ = torch.randn(1, device='cuda')
            device = torch.device('cuda')
            if verbose:
                print("✅ CUDA (GPU) Selected")
            return device
        except Exception as e:
            if verbose:
                print(f"⚠️ CUDA available but error occurred: {str(e)}")
                print("   Falling back to CPU...")
    
    # Try MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        try:
            _ = torch.randn(1, device='mps')
            device = torch.device('mps')
            if verbose:
                print("✅ MPS (Apple Silicon GPU) Selected")
            return device
        except Exception as e:
            if verbose:
                print(f"⚠️ MPS available but error occurred: {str(e)}")
                print("   Falling back to CPU...")
    
    # Fallback to CPU
    device = torch.device('cpu')
    if verbose:
        print("⚠️ CPU Selected (CUDA and MPS not available)")
    return device


def get_device_info() -> Dict[str, str]:
    """
    Get detailed information about the current device setup.
    
    Returns
    -------
    dict
        Dictionary with device information keys:
        - 'device': Device type (cuda, mps, cpu)
        - 'gpu_name': GPU model name if available
        - 'cuda_version': CUDA version if available
        - 'gpu_memory': Total GPU memory in GB if available
        - 'compute_capability': GPU compute capability if available
    """
    info = {}
    
    device = setup_training_device(verbose=False)
    info['device'] = str(device)
    
    if device.type == 'cuda':
        gpu_idx = 0
        info['gpu_name'] = torch.cuda.get_device_name(gpu_idx)
        info['cuda_version'] = torch.version.cuda
        total_memory_gb = torch.cuda.get_device_properties(gpu_idx).total_memory / 1e9
        info['gpu_memory'] = f"{total_memory_gb:.1f} GB"
        cap = torch.cuda.get_device_capability(gpu_idx)
        info['compute_capability'] = f"{cap[0]}.{cap[1]}"
    elif device.type == 'mps':
        info['gpu_name'] = "Apple Silicon"
    else:
        info['gpu_name'] = "N/A"
    
    return info


def print_device_info(title: str = "🎯 Device Configuration") -> None:
    """
    Pretty-print device information to console.
    
    Parameters
    ----------
    title : str
        Title for the output section.
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    device = setup_training_device(verbose=False)
    info = get_device_info()
    
    print(f"Device: {info['device'].upper()}")
    
    if device.type == 'cuda':
        print(f"GPU: {info['gpu_name']}")
        print(f"CUDA Version: {info['cuda_version']}")
        print(f"Memory: {info['gpu_memory']}")
        print(f"Compute Capability: {info['compute_capability']}")
        
        # Show memory usage
        free_memory_gb = torch.cuda.mem_get_info()[0] / 1e9
        print(f"Memory Available: {free_memory_gb:.1f} GB")
    elif device.type == 'mps':
        print(f"GPU: {info['gpu_name']}")
    else:
        print("GPU: Not available (using CPU)")
    
    print(f"{'='*60}\n")


def move_model_to_device(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """
    Safely move a model to the specified device.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to move.
    device : torch.device
        Target device.
    
    Returns
    -------
    torch.nn.Module
        The model on the specified device.
    """
    return model.to(device)


# Convenience functions for common operations
def tensor_to_device(tensor: torch.Tensor, device: torch.device = None) -> torch.Tensor:
    """
    Move a tensor to the specified device (or detected device if None).
    
    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to move.
    device : torch.device, optional
        Target device. If None, uses detected device.
    
    Returns
    -------
    torch.Tensor
        The tensor on the specified device.
    """
    if device is None:
        device = setup_training_device(verbose=False)
    return tensor.to(device)


if __name__ == "__main__":
    # Demo when run as a script
    print("GPU Utils Demo")
    print_device_info("🎯 Your Device Configuration")
    
    info = get_device_info()
    print("\nDevice Info Dictionary:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test GPU performance
    print("\n⚡ Quick GPU Performance Test:")
    device = setup_training_device(verbose=False)
    
    # Create a large tensor and perform operations
    size = 10000
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    import time
    start = time.time()
    c = torch.matmul(a, b)
    elapsed = time.time() - start
    
    print(f"  Matrix multiplication ({size}x{size}): {elapsed*1000:.2f}ms")
    print(f"  Status: {'✅ GPU Working' if device.type == 'cuda' else '✅ Device Ready'}")
