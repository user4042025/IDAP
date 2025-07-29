"""Utility functions for training and evaluating the model.

This module implements a collection of utility functions for deep learning model training,
evaluation, and analysis. It provides functionality for:
1. Model evaluation metrics (top-k accuracy)
2. Model architecture analysis and statistics
3. Layer pruning utilities
4. Attention mechanism analysis
5. Model parameter management (freezing/unfreezing)
"""

import time
import torch
import torch.nn as nn
from torchinfo import summary


def evaluate_topk(model, test_loader, device, k=5):
    """
    Evaluate the top-k accuracy of the model on the given test dataset.

    Args:
        model: Model to evaluate.
        test_loader: DataLoader for the test set.
        device: Device to use (CPU/GPU).
        k: Number of top predictions to consider. Default is 5.

    Returns:
        float: Top-k accuracy as a ratio between 0 and 1.
    """

    model.eval()

    correct = 0  # Counter for correct predictions
    total = 0    # Counter for total samples
    start_time = time.time()

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.topk(k, 1, True, True)
            # Count correct predictions (true label in top-k)
            correct += (predicted == y.view(-1, 1)).sum().item()
            total += y.size(0)

    elapsed_time = time.time() - start_time
    print(f"[evaluate_topk] Top-{k} accuracy computed in {elapsed_time:.2f} seconds.")

    return correct / total


def is_prunable_layer(module, num_classes=10):
    """
    Check whether the layer is prunable.

    Args:
        module: Layer to check.
        num_classes: Output size of final classifier, to exclude last layer.

    Returns:
        bool: True if the layer can be pruned, False otherwise.
    """

    if isinstance(module, nn.Conv2d) and module.kernel_size != (1, 1):
        # Exclude 1x1 conv layers
        return True

    elif isinstance(module, nn.Linear) and module.out_features != num_classes:
        # Exclude last linear layer to preserve output structure
        return True

    elif isinstance(module, nn.MultiheadAttention):
        return True

    return False


def get_module_by_name(model, name):
    """
    Retrieve a submodule from a model using its name.

    Args:
        model: Model to search within.
        name: Dot-separated name of the module.

    Returns:
        torch.nn.Module: The requested submodule.

    Raises:
        ValueError: If the module name path is invalid or module not found.
    """

    parts = name.split(".")
    module = model

    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = module._modules.get(part)
            if module is None:
                raise ValueError(f"Module '{part}' not found in the model.")

    return module


def get_model_summary(model, device="cuda"):
    """
    Generate a detailed summary of the model's architecture and resource usage.
    
    This function uses torchinfo to analyze the model and provide statistics about:
    - Parameter counts (total and trainable)
    - Computational complexity (FLOPs)
    - Memory usage (forward/backward pass, parameters)
    - Model size

    Args:
        model: Model to analyze.
        device: Device for dummy input and summary generation. Default is 'cuda'.

    Returns:
        dict: Dictionary containing model statistics:
            - total_params: Total number of parameters
            - trainable_params: Number of trainable parameters
            - gflops: Computational complexity in GFLOPs
            - forward_backward_pass_size_MB: Memory for forward/backward pass
            - params_size_MB: Memory for parameters
            - estimated_total_memory_MB: Total estimated memory usage
    """

    # Generate summary with a dummy input of size (1, 3, 224, 224)
    cur_summary = summary(model, (1, 3, 224, 224), device=device, verbose=0)

    return {
        "total_params": cur_summary.total_params,
        "trainable_params": cur_summary.trainable_params,
        "gflops": cur_summary.total_mult_adds / 1e9,  # Convert to GFLOPs
        "forward_backward_pass_size_MB": cur_summary.total_output_bytes / 1024**2,
        "params_size_MB": cur_summary.total_param_bytes / 1024**2,
        "estimated_total_memory_MB": (
            cur_summary.total_input +
            cur_summary.total_output_bytes +
            cur_summary.total_param_bytes
        ) / 1024**2
    }


def count_attention_heads(model):
    """
    Count the total number of attention heads across all MultiheadAttention layers.

    Args:
        model: Model to inspect.

    Returns:
        int: Total number of attention heads in the model.
    """

    total_heads = 0

    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            total_heads += module.num_heads

    return total_heads


def get_named_attention_layers(model):
    """
    Retrieve all attention layers along with their names from the model.

    Args:
        model: Model to inspect.

    Returns:
        list: List of tuples (name, attention_layer) for all MultiheadAttention layers.
    """

    attention_layers = []

    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            attention_layers.append((name, module))

    return attention_layers


def freeze_model(model):
    """
    Freeze all parameters in the model (set requires_grad=False).

    Args:
        model: Model to freeze.

    Returns:
        None
    """
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model):
    """
    Unfreeze all parameters in the model (set requires_grad=True).

    Args:
        model: Model to unfreeze.

    Returns:
        None
    """
    for param in model.parameters():
        param.requires_grad = True
