"""Module for plotting training metrics.

This module implements functionality for parsing training log files and generating
visualizations of various training metrics. It provides tools to analyze and plot:
1. GFlops (Giga Floating Point Operations per Second)
2. Total model parameters
3. Parameter sizes in bytes
4. TOP-1 and TOP-5 accuracy metrics
"""

import matplotlib.pyplot as plt
import re
from datetime import datetime
import numpy as np
import json


def parse_log_file(log_file_path):
    """
    Parse training log file to extract various metrics and model information.
    
    Args:
        log_file_path: Path to the training log file
        
    Returns:
        tuple: Contains the following metrics:
            - timestamps: List of timestamps for each measurement
            - gflops_values: List of GFlops measurements
            - total_params_values: List of total parameter counts
            - params_size_values: List of parameter sizes in bytes
            - top1_accuracy_values: List of TOP-1 accuracy measurements
            - top5_accuracy_values: List of TOP-5 accuracy measurements
            - model_name: Name of the model
            - dataset_name: Name of the dataset
    """

    # Initialize lists to store different metrics
    timestamps = []
    gflops_values = []
    total_params_values = []
    params_size_values = []
    top1_accuracy_values = []
    top5_accuracy_values = []
    model_name = None
    dataset_name = None
    
    print(f"Reading log file: {log_file_path}")
    line_count = 0
    matches_found = 0
    
    with open(log_file_path, 'r') as f:
        for line in f:
            line_count += 1
            # Print progress every 10000 lines
            if line_count % 10000 == 0:
                print(f"Processed {line_count} lines...")
            
            # Extract model and dataset information from the first relevant line
            if model_name is None and 'resnet' in line.lower():
                model_name = 'ResNet50'
                dataset_name = 'CIFAR-10'
            
            # Look for lines containing model summary information
            if "'gflops':" in line or "'total_params':" in line:
                # Extract timestamp using regex pattern
                timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                if timestamp_match:
                    timestamp = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                    timestamps.append(timestamp)
                
                # Extract GFlops value and convert to actual GFlops
                gflops_match = re.search(r"'gflops':\s*(\d+)", line)
                if gflops_match:
                    matches_found += 1
                    gflops = float(gflops_match.group(1)) / 1e9  # Convert to actual GFlops
                    gflops_values.append(gflops)
                
                # Extract total number of parameters
                total_params_match = re.search(r"'total_params':\s*(\d+)", line)
                if total_params_match:
                    total_params = float(total_params_match.group(1))
                    total_params_values.append(total_params)
                
                # Extract parameter size in bytes
                params_size_match = re.search(r"'params_size':\s*(\d+)", line)
                if params_size_match:
                    params_size = float(params_size_match.group(1))
                    params_size_values.append(params_size)
            
            # Extract TOP-1 accuracy value and convert to percentage
            accuracy_match = re.search(r"TOP-1 Accuracy:\s*(\d+\.\d+)", line)
            if accuracy_match:
                accuracy = float(accuracy_match.group(1))
                accuracy = round(accuracy * 100, 2)
                # Skip if the value is the same as the previous one
                if len(top1_accuracy_values) != 0 and top1_accuracy_values[-1] == accuracy:
                    continue
                top1_accuracy_values.append(accuracy)
            
            # Extract TOP-5 accuracy value and convert to percentage
            top5_accuracy_match = re.search(r"TOP-5 Accuracy:\s*(\d+\.\d+)", line)
            if top5_accuracy_match:
                top5_accuracy = float(top5_accuracy_match.group(1))
                top5_accuracy = round(top5_accuracy * 100, 2)

                if len(top5_accuracy_values) != 0 and top5_accuracy_values[-1] == top5_accuracy:
                    continue
                top5_accuracy_values.append(top5_accuracy)
    
    # Print summary of extracted data
    print(f"Total lines processed: {line_count}")
    print(f"Total GFlops matches found: {matches_found}")
    print(f"GFlops values: {gflops_values}")
    print(f"Total params values: {total_params_values}")
    print(f"Params size values: {params_size_values}")
    print(f"TOP-1 Accuracy values: {top1_accuracy_values}")
    print(f"TOP-5 Accuracy values: {top5_accuracy_values}")
    
    # Debug information if no matches found
    if matches_found == 0:
        print("\nDebug: Here are some example lines from the log file:")
        with open(log_file_path, 'r') as f:
            for i, line in enumerate(f):
                if i < 5:
                    print(f"Line {i+1}: {line.strip()}")
    
    return timestamps, gflops_values, total_params_values, params_size_values, top1_accuracy_values, top5_accuracy_values, model_name, dataset_name


def plot_metrics(timestamps, gflops_values, total_params_values, model_name, dataset_name, output_prefix='training_metrics'):
    """
    Plot GFlops and total parameters over iterations.
    
    Args:
        timestamps: List of timestamps
        gflops_values: List of GFlops measurements
        total_params_values: List of total parameter counts
        model_name: Name of the model
        dataset_name: Name of the dataset
        output_prefix: Prefix for output file names
        
    Returns:
        tuple: Paths to the generated plot files (gflops_plot, params_plot)
    """

    # Create figure for GFlops plot
    plt.figure(figsize=(12, 6))
    
    # Use iteration numbers (indices) for x-axis
    iterations = range(1, len(gflops_values)+1)
    
    # Plot GFlops with line and markers
    plt.plot(iterations, gflops_values, '#1f77b4', linewidth=2, label='GFlops')
    plt.plot(iterations, gflops_values, 'o', color='#1f77b4', markersize=6)
    
    # Add text labels for each GFlops point
    for i, gflops in enumerate(gflops_values):
        plt.text(iterations[i], gflops, f'{gflops:.2f}', 
                ha='center', va='bottom', fontsize=9)
    
    # Configure GFlops plot appearance
    plt.title(f'Training GFlops Over Iterations\n{model_name} on {dataset_name}', fontsize=12, pad=15)
    plt.xlabel('Iteration', fontsize=10)
    plt.ylabel('GFlops', fontsize=10)
    plt.grid(True, which='major', linestyle='-', alpha=0.3)
    plt.grid(True, which='minor', linestyle=':', alpha=0.2)
    plt.ylim(bottom=0)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(frameon=False, loc='upper right')
    plt.xticks(iterations)
    plt.tight_layout()
    
    # Save GFlops plot
    gflops_output = f'{output_prefix}_gflops.png'
    plt.savefig(gflops_output, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create figure for total parameters plot
    plt.figure(figsize=(12, 6))
    
    # Convert total parameters to millions for better readability
    total_params_millions = [params / 1000000 for params in total_params_values]
    
    # Plot total parameters with line and markers
    plt.plot(iterations, total_params_millions, '#ff7f0e', linewidth=2, label='Total Parameters (M)')
    plt.plot(iterations, total_params_millions, 'o', color='#ff7f0e', markersize=6)
    
    # Add text labels for each parameter count point
    for i, params in enumerate(total_params_millions):
        plt.text(iterations[i], params, f'{params:.2f}', 
                ha='center', va='bottom', fontsize=9)
    
    # Configure total parameters plot appearance
    plt.title(f'Total Parameters Over Iterations\n{model_name} on {dataset_name}', fontsize=12, pad=15)
    plt.xlabel('Iteration', fontsize=10)
    plt.ylabel('Total Parameters (M)', fontsize=10)
    plt.grid(True, which='major', linestyle='-', alpha=0.3)
    plt.grid(True, which='minor', linestyle=':', alpha=0.2)
    plt.ylim(bottom=0)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(frameon=False, loc='upper right')
    plt.xticks(iterations)
    plt.tight_layout()
    
    # Save total parameters plot
    params_output = f'{output_prefix}_params.png'
    plt.savefig(params_output, dpi=300, bbox_inches='tight')
    plt.close()
    
    return gflops_output, params_output


def plot_top1_accuracy(accuracy_values, model_name, dataset_name, output_prefix='training_metrics'):
    """
    Plot TOP-1 accuracy over iterations.
    
    Args:
        accuracy_values: List of TOP-1 accuracy measurements
        model_name: Name of the model
        dataset_name: Name of the dataset
        output_prefix: Prefix for output file name
        
    Returns:
        str: Path to the generated plot file
    """
    plt.figure(figsize=(12, 6))
    
    # Use iteration numbers (indices) for x-axis
    iterations = range(1, len(accuracy_values)+1)
    
    # Plot accuracy with line and markers
    plt.plot(iterations, accuracy_values, '#2ca02c', linewidth=2, label='TOP-1 Accuracy')
    plt.plot(iterations, accuracy_values, 'o', color='#2ca02c', markersize=6)
    
    # Add text labels for each accuracy point
    for i, acc in enumerate(accuracy_values):
        acc = round(acc, 2)
        plt.text(iterations[i], acc, f'{acc:.2f}', 
                ha='center', va='bottom', fontsize=9)
    
    # Calculate min and max accuracy for y-axis limits
    min_acc = min(accuracy_values)
    max_acc = max(accuracy_values)

    # Configure accuracy plot appearance
    plt.title(f'TOP-1 Accuracy Over Iterations\n{model_name} on {dataset_name}', fontsize=12, pad=15)
    plt.xlabel('Iteration', fontsize=10)
    plt.ylabel('TOP-1 Accuracy (%)', fontsize=10)
    plt.grid(True, which='major', linestyle='-', alpha=0.3)
    plt.grid(True, which='minor', linestyle=':', alpha=0.2)
    y_margin = (max_acc - min_acc) * 0.1  # 10% margin
    plt.ylim(bottom=min_acc - y_margin, top=max_acc + y_margin)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(frameon=False, loc='upper right')
    plt.xticks(iterations)
    plt.tight_layout()
    
    # Save accuracy plot
    accuracy_output = f'{output_prefix}_top1_accuracy.png'
    plt.savefig(accuracy_output, dpi=300, bbox_inches='tight')
    plt.close()
    
    return accuracy_output

def plot_top5_accuracy(accuracy_values, model_name, dataset_name, output_prefix='training_metrics'):
    """
    Plot TOP-5 accuracy over iterations.
    
    Args:
        accuracy_values: List of TOP-5 accuracy measurements
        model_name: Name of the model
        dataset_name: Name of the dataset
        output_prefix: Prefix for output file name
        
    Returns:
        str: Path to the generated plot file, or None if not enough data points
    """

    # Check if there are enough data points for meaningful plotting
    if len(accuracy_values) < 2:
        print("Warning: Not enough TOP-5 accuracy data points to create a meaningful plot (need at least 2 points)")
        return None
        
    try:
        plt.figure(figsize=(12, 6))
        
        # Use iteration numbers (indices) for x-axis
        iterations = range(1, len(accuracy_values)+1)
        
        # Plot accuracy with line and markers
        plt.plot(iterations, accuracy_values, '#9467bd', linewidth=2, label='TOP-5 Accuracy')
        plt.plot(iterations, accuracy_values, 'o', color='#9467bd', markersize=6)
        
        # Add text labels for each accuracy point
        for i, acc in enumerate(accuracy_values):
            acc = round(acc, 2)
            plt.text(iterations[i], acc, f'{acc:.2f}', 
                    ha='center', va='bottom', fontsize=9)
        
        # Calculate min and max accuracy for y-axis limits
        min_acc = min(accuracy_values)
        max_acc = max(accuracy_values)
        
        # Configure accuracy plot appearance
        plt.title(f'TOP-5 Accuracy Over Iterations\n{model_name} on {dataset_name}', fontsize=12, pad=15)
        plt.xlabel('Iteration', fontsize=10)
        plt.ylabel('TOP-5 Accuracy (%)', fontsize=10)
        plt.grid(True, which='major', linestyle='-', alpha=0.3)
        plt.grid(True, which='minor', linestyle=':', alpha=0.2)
        
        # Set y-axis limits with margins
        y_margin = (max_acc - min_acc) * 0.1  # 10% margin
        plt.ylim(bottom=min_acc - y_margin, top=max_acc + y_margin)
        
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.legend(frameon=False, loc='upper right')
        plt.xticks(iterations)
        plt.tight_layout()
        
        # Save accuracy plot
        accuracy_output = f'{output_prefix}_top5_accuracy.png'
        plt.savefig(accuracy_output, dpi=300, bbox_inches='tight')
        plt.close()
        
        return accuracy_output
        
    except Exception as e:
        print(f"Error plotting TOP-5 accuracy: {str(e)}")
        plt.close('all')  # Ensure all figures are closed
        return None

def plot_params_size(params_size_values, model_name, dataset_name, output_prefix='training_metrics'):
    """
    Plot model parameters size over iterations.
    
    Args:
        params_size_values: List of parameter sizes in bytes
        model_name: Name of the model
        dataset_name: Name of the dataset
        output_prefix: Prefix for output file name
        
    Returns:
        str: Path to the generated plot file
    """

    plt.figure(figsize=(12, 6))
    
    # Use iteration numbers (indices) for x-axis
    iterations = range(1, len(params_size_values)+1)
    
    # Convert parameter sizes to MB for better readability
    params_size_mb = [size / (1024 * 1024) for size in params_size_values]
    
    # Plot parameter sizes with line and markers
    plt.plot(iterations, params_size_mb, '#17becf', linewidth=2, label='Params Size (MB)')
    plt.plot(iterations, params_size_mb, 'o', color='#17becf', markersize=6)
    
    # Add text labels for each size point
    for i, size in enumerate(params_size_mb):
        plt.text(iterations[i], size, f'{size:.2f}', 
                ha='center', va='bottom', fontsize=9)
    
    # Configure parameter size plot appearance
    plt.title(f'Model Parameters Size Over Iterations\n{model_name} on {dataset_name}', fontsize=12, pad=15)
    plt.xlabel('Iteration', fontsize=10)
    plt.ylabel('Parameters Size (MB)', fontsize=10)
    plt.grid(True, which='major', linestyle='-', alpha=0.3)
    plt.grid(True, which='minor', linestyle=':', alpha=0.2)
    
    # Set y-axis limits with proper margins
    min_size = min(params_size_mb)
    max_size = max(params_size_mb)
    y_margin = (max_size - min_size) * 0.1  # 10% margin
    plt.ylim(bottom=max(0, min_size - y_margin), top=max_size + y_margin)
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(frameon=False, loc='upper right')
    plt.xticks(iterations)
    plt.tight_layout()
    
    # Save parameter size plot
    params_size_output = f'{output_prefix}_params_size.png'
    plt.savefig(params_size_output, dpi=300, bbox_inches='tight')
    plt.close()
    
    return params_size_output

def main():
    """
    Main function to process log file and generate plots.
    Reads the training log file, extracts metrics, and generates various plots.
    """
    # Path to the training log file
    log_file_path = r'.\logs\training_log.log'
    
    try:
        # Parse log file to extract metrics
        timestamps, gflops_values, total_params_values, params_size_values, top1_accuracy_values, top5_accuracy_values, model_name, dataset_name = parse_log_file(log_file_path)
        
        # Check if we have the minimum required data
        if not gflops_values or not total_params_values:
            print("No data found in the log file.")
            return
        
        # Print GFlops statistics
        print(f"Found {len(gflops_values)} GFlops measurements")
        print(f"Average GFlops: {np.mean(gflops_values):.2f}")
        print(f"Max GFlops: {np.max(gflops_values):.2f}")
        print(f"Min GFlops: {np.min(gflops_values):.2f}")
        
        # Print total parameters statistics
        print(f"\nFound {len(total_params_values)} total_params measurements")
        print(f"Average total_params: {np.mean(total_params_values):.2f}")
        print(f"Max total_params: {np.max(total_params_values):.2f}")
        print(f"Min total_params: {np.min(total_params_values):.2f}")
        
        # Print parameter size statistics if available
        if params_size_values:
            print(f"\nFound {len(params_size_values)} params_size measurements")
            print(f"Average params_size: {np.mean(params_size_values):.2f} bytes")
            print(f"Max params_size: {np.max(params_size_values):.2f} bytes")
            print(f"Min params_size: {np.min(params_size_values):.2f} bytes")
        
        # Print TOP-1 accuracy statistics if available
        if top1_accuracy_values:
            print(f"\nFound {len(top1_accuracy_values)} TOP-1 accuracy measurements")
            print(f"Average TOP-1 accuracy: {np.mean(top1_accuracy_values):.2f}%")
            print(f"Max TOP-1 accuracy: {np.max(top1_accuracy_values):.2f}%")
            print(f"Min TOP-1 accuracy: {np.min(top1_accuracy_values):.2f}%")
        
        # Print TOP-5 accuracy statistics if available
        if top5_accuracy_values:
            print(f"\nFound {len(top5_accuracy_values)} TOP-5 accuracy measurements")
            print(f"Average TOP-5 accuracy: {np.mean(top5_accuracy_values):.2f}%")
            print(f"Max TOP-5 accuracy: {np.max(top5_accuracy_values):.2f}%")
            print(f"Min TOP-5 accuracy: {np.min(top5_accuracy_values):.2f}%")
        
        # Generate and save plots
        gflops_file, params_file = plot_metrics(timestamps, gflops_values, total_params_values, model_name, dataset_name)
        print(f"GFlops plot saved as '{gflops_file}'")
        print(f"Total parameters plot saved as '{params_file}'")
        
        # Generate parameter size plot if data is available
        if params_size_values:
            params_size_file = plot_params_size(params_size_values, model_name, dataset_name)
            print(f"Parameters size plot saved as '{params_size_file}'")
        
        # Generate TOP-1 accuracy plot if data is available
        if top1_accuracy_values:
            top1_accuracy_file = plot_top1_accuracy(top1_accuracy_values, model_name, dataset_name)
            print(f"TOP-1 accuracy plot saved as '{top1_accuracy_file}'")
        
        # Generate TOP-5 accuracy plot if data is available
        if top5_accuracy_values:
            top5_accuracy_file = plot_top5_accuracy(top5_accuracy_values, model_name, dataset_name)
            if top5_accuracy_file:
                print(f"TOP-5 accuracy plot saved as '{top5_accuracy_file}'")
        
    except FileNotFoundError:
        print(f"Error: Log file '{log_file_path}' not found.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
