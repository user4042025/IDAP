"""Main script for IDAP pruning."""

import datetime
import logging
import numpy as np
import random
import os
from torch.utils.tensorboard import SummaryWriter

import torch
from torch.utils.data import DataLoader

from src.idap import idap_pruning
from src.models import load_model_and_dataset

# Save the current date and time
current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# Logging to a file
logging.basicConfig(
    filename=current_date+".log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Configuration for models and datasets
MODEL_CONFIGS = [
    {
        "name": "resnet50",
        "paths": {
            "cifar10": r".\weights\resnet50__cifar10.pth",
            "cifar100": r".\weights\resnet50__cifar100.pth",
            "stanford_cars": r".\weights\resnet50__stanford_cars.pth",
            "flowers102": r".\weights\resnet50__flowers102.pth",
            "food101": r".\weights\resnet50__food101.pth",
            "oxford_iiit_pet": r".\weights\resnet50__oxford_iiit_pet.pth",
            "fashion_mnist": r".\weights\resnet50__fashion_mnist.pth",
            "fer2013": r".\weights\resnet50__fer2013.pth",
            "inaturalist": r".\weights\resnet50__inaturalist.pth",
            "imagenet": r".\weights\resnet50__imagenet.pth"
        }
    },
    {
        "name": "densenet121",
        "paths": {
            "cifar10": r".\weights\densenet121__cifar10.pth",
            "cifar100": r".\weights\densenet121__cifar100.pth",
            "stanford_cars": r".\weights\densenet121__stanford_cars.pth",
            "flowers102": r".\weights\densenet121__flowers102.pth",
            "food101": r".\weights\densenet121__food101.pth",
            "oxford_iiit_pet": r".\weights\densenet121__oxford_iiit_pet.pth",
            "fashion_mnist": r".\weights\densenet121__fashion_mnist.pth",
            "fer2013": r".\weights\densenet121__fer2013.pth",
            "inaturalist": r".\weights\densenet121__inaturalist.pth",
            "imagenet": r".\weights\densenet121__imagenet.pth"
        }
    },
    {
        "name": "efficientnet_b4",
        "paths": {
            "cifar10": r".\weights\efficientnet_b4__cifar10.pth",
            "cifar100": r".\weights\efficientnet_b4__cifar100.pth",
            "stanford_cars": r".\weights\efficientnet_b4__stanford_cars.pth",
            "flowers102": r".\weights\efficientnet_b4__flowers102.pth",
            "food101": r".\weights\efficientnet_b4__food101.pth",
            "oxford_iiit_pet": r".\weights\efficientnet_b4__oxford_iiit_pet.pth",
            "fashion_mnist": r".\weights\efficientnet_b4__fashion_mnist.pth",
            "fer2013": r".\weights\efficientnet_b4__fer2013.pth",
            "inaturalist": r".\weights\efficientnet_b4__inaturalist.pth",
            "imagenet": r".\weights\efficientnet_b4__imagenet.pth"
        }
    },
    {
        "name": "efficientnetv2_s",
        "paths": {
            "cifar10": r".\weights\efficientnetv2_s__cifar10.pth",
            "cifar100": r".\weights\efficientnetv2_s__cifar100.pth",
            "stanford_cars": r".\weights\efficientnetv2_s__stanford_cars.pth",
            "flowers102": r".\weights\efficientnetv2_s__flowers102.pth",
            "food101": r".\weights\efficientnetv2_s__food101.pth",
            "oxford_iiit_pet": r".\weights\efficientnetv2_s__oxford_iiit_pet.pth",
            "fashion_mnist": r".\weights\efficientnetv2_s__fashion_mnist.pth",
            "fer2013": r".\weights\efficientnetv2_s__fer2013.pth",
            "inaturalist": r".\weights\efficientnetv2_s__inaturalist.pth",
            "imagenet": r".\weights\efficientnetv2_s__imagenet.pth"
        }
    },
    {
        "name": "vit_base_patch16_224",
        "paths": {
            "cifar10": r".\weights\vit_base_patch16_224__cifar10.pth",
            "cifar100": r".\weights\vit_base_patch16_224__cifar100.pth",
            "stanford_cars": r".\weights\vit_base_patch16_224__stanford_cars.pth",
            "flowers102": r".\weights\vit_base_patch16_224__flowers102.pth",
            "food101": r".\weights\vit_base_patch16_224__food101.pth",
            "oxford_iiit_pet": r".\weights\vit_base_patch16_224__oxford_iiit_pet.pth",
            "fashion_mnist": r".\weights\vit_base_patch16_224__fashion_mnist.pth",
            "fer2013": r".\weights\vit_base_patch16_224__fer2013.pth",
            "inaturalist": r".\weights\vit_base_patch16_224__inaturalist.pth",
            "imagenet": r".\weights\vit_base_patch16_224__imagenet.pth"
        }
    },
    {
        "name": "mobilenetv3_large_100",
        "paths": {
            "cifar10": r".\weights\mobilenetv3_large_100__cifar10.pth",
            "cifar100": r".\weights\mobilenetv3_large_100__cifar100.pth",
            "stanford_cars": r".\weights\mobilenetv3_large_100__stanford_cars.pth",
            "flowers102": r".\weights\mobilenetv3_large_100__flowers102.pth",
            "food101": r".\weights\mobilenetv3_large_100__food101.pth",
            "oxford_iiit_pet": r".\weights\mobilenetv3_large_100__oxford_iiit_pet.pth",
            "fashion_mnist": r".\weights\mobilenetv3_large_100__fashion_mnist.pth",
            "fer2013": r".\weights\mobilenetv3_large_100__fer2013.pth",
            "inaturalist": r".\weights\mobilenetv3_large_100__inaturalist.pth",
            "imagenet": r".\weights\mobilenetv3_large_100__imagenet.pth"
        }
    },
    {
        "name": "convnext_small",
        "paths": {
            "cifar10": r".\weights\convnext_small__cifar10.pth",
            "cifar100": r".\weights\convnext_small__cifar100.pth",
            "stanford_cars": r".\weights\convnext_small__stanford_cars.pth",
            "flowers102": r".\weights\convnext_small__flowers102.pth",
            "food101": r".\weights\convnext_small__food101.pth",
            "oxford_iiit_pet": r".\weights\convnext_small__oxford_iiit_pet.pth",
            "fashion_mnist": r".\weights\convnext_small__fashion_mnist.pth",
            "fer2013": r".\weights\convnext_small__fer2013.pth",
            "inaturalist": r".\weights\convnext_small__inaturalist.pth",
            "imagenet": r".\weights\convnext_small__imagenet.pth"
        }
    },
    {
        "name": "inception_v3",
        "paths": {
            "cifar10": r".\weights\inception_v3__cifar10.pth",
            "cifar100": r".\weights\inception_v3__cifar100.pth",
            "stanford_cars": r".\weights\inception_v3__stanford_cars.pth",
            "flowers102": r".\weights\inception_v3__flowers102.pth",
            "food101": r".\weights\inception_v3__food101.pth",
            "oxford_iiit_pet": r".\weights\inception_v3__oxford_iiit_pet.pth",
            "fashion_mnist": r".\weights\inception_v3__fashion_mnist.pth",
            "fer2013": r".\weights\inception_v3__fer2013.pth",
            "inaturalist": r".\weights\inception_v3__inaturalist.pth",
            "imagenet": r".\weights\inception_v3__imagenet.pth"
        }
    },
    {
        "name": "vgg19_bn",
        "paths": {
            "cifar10": r".\weights\vgg19_bn__cifar10.pth",
            "cifar100": r".\weights\vgg19_bn__cifar100.pth",
            "stanford_cars": r".\weights\vgg19_bn__stanford_cars.pth",
            "flowers102": r".\weights\vgg19_bn__flowers102.pth",
            "food101": r".\weights\vgg19_bn__food101.pth",
            "oxford_iiit_pet": r".\weights\vgg19_bn__oxford_iiit_pet.pth",
            "fashion_mnist": r".\weights\vgg19_bn__fashion_mnist.pth",
            "fer2013": r".\weights\vgg19_bn__fer2013.pth",
            "inaturalist": r".\weights\vgg19_bn__inaturalist.pth",
            "imagenet": r".\weights\vgg19_bn__imagenet.pth"
        }
    },
    {
        "name": "shufflenet_v2_x2_0",
        "paths": {
            "cifar10": r".\weights\shufflenet_v2_x2_0__cifar10.pth",
            "cifar100": r".\weights\shufflenet_v2_x2_0__cifar100.pth",
            "stanford_cars": r".\weights\shufflenet_v2_x2_0__stanford_cars.pth",
            "flowers102": r".\weights\shufflenet_v2_x2_0__flowers102.pth",
            "food101": r".\weights\shufflenet_v2_x2_0__food101.pth",
            "oxford_iiit_pet": r".\weights\shufflenet_v2_x2_0__oxford_iiit_pet.pth",
            "fashion_mnist": r".\weights\shufflenet_v2_x2_0__fashion_mnist.pth",
            "fer2013": r".\weights\shufflenet_v2_x2_0__fer2013.pth",
            "inaturalist": r".\weights\shufflenet_v2_x2_0__inaturalist.pth",
            "imagenet": r".\weights\shufflenet_v2_x2_0__imagenet.pth"
        }
    }
]

# List of supported datasets for pruning
DATASETS = [
    "cifar10",          # 10 classes
    "cifar100",         # 100 classes
    "stanford_cars",    # 196 classes
    "flowers102",       # 102 classes
    "food101",          # 101 classes
    "oxford_iiit_pet",  # 37 classes
    "fashion_mnist",    # 10 classes
    "fer2013",          # 7 classes
    "inaturalist",      # 8142 classes
    "imagenet"          # 1000 classes
]

def set_seed(seed=42):
    """Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed value. Default is 42.
        
    This function ensures that all random operations (PyTorch, NumPy, Python)
    are deterministic and reproducible.
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_model_dataset(model_config, dataset_name, device, writer):
    """Process a single model-dataset combination for pruning.
    
    Args:
        model_config: Configuration dictionary containing model name and paths
        dataset_name: Name of the dataset to use
        device: Device to run the model on (CPU/GPU)
        writer: TensorBoard writer for logging metrics
    """

    # Get the path to the pre-trained weights for this model-dataset combination
    model_path = model_config["paths"][dataset_name]
    model_name = model_config["name"]
    
    logging.debug("Start pruning for model %s on dataset %s", model_name, dataset_name)
    
    # Load the model and dataset with appropriate transforms
    model, num_classes, train_dataset, test_dataset = load_model_and_dataset(
        model_path, model_name, dataset_name
    )

    # Log initial model statistics
    total_params = sum(p.numel() for p in model.parameters())
    writer.add_scalar(f'{model_name}_{dataset_name}/Initial_Parameters', total_params, 0)

    # Pruning configuration
    batch_size = 100
    max_performance_metric_degradation_th = 0.05  # Maximum allowed performance degradation
    base_pruning_percentage = 0.10  # Initial pruning percentage
    number_of_pruning_iterations = 20  # Number of pruning iterations

    # Create data loaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Perform IDAP pruning
    pruned_model = idap_pruning(
        model_path=model_path,
        pretrained_model=model,
        test_loader=test_loader,
        train_loader=train_loader,
        num_classes=num_classes,
        max_performance_metric_degradation_th=max_performance_metric_degradation_th,
        number_of_pruning_iterations=number_of_pruning_iterations,
        base_pruning_percentage=base_pruning_percentage,
        device=device,
        writer=writer
    )

    # Log final model statistics after pruning
    total_params_after = sum(p.numel() for p in pruned_model.parameters())
    writer.add_scalar(f'{model_name}_{dataset_name}/Final_Parameters', total_params_after, 1)
    writer.add_scalar(
        f'{model_name}_{dataset_name}/Parameters_Reduction',
        (total_params - total_params_after) / total_params * 100,
        1
    )

    # Save the pruned model
    pruned_model_path = model_path.replace(".pth", f"_pruned_{dataset_name}.pth")
    torch.save(pruned_model, pruned_model_path)
    logging.debug("Saved pruned model to %s", pruned_model_path)

def main():
    """Main function to run IDAP pruning on multiple models and datasets."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    # Create TensorBoard writer for logging metrics
    log_dir = os.path.join('runs', f'pruning_{current_date}')
    writer = SummaryWriter(log_dir=log_dir)

    # Process each model-dataset combination
    for model_config in MODEL_CONFIGS:
        for dataset_name in DATASETS:
            try:
                process_model_dataset(model_config, dataset_name, device, writer)
            except Exception as e:
                # Log error and continue with next combination
                logging.error(
                    "Error processing model %s with dataset %s: %s",
                    model_config["name"],
                    dataset_name,
                    str(e)
                )
                continue

    writer.close()


if __name__ == "__main__":
    main()
