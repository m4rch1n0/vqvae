import argparse
from pathlib import Path
from typing import Optional

import torch
import lpips
import yaml
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import torch.nn.functional as F
from torchvision.datasets import CIFAR10

from src.eval.metrics import psnr, ssim_simple


def load_images(
    path_or_name: str, 
    num_images: int, 
    size: int, 
    dataset_name: str, 
    is_real_data: bool = False,
    samples_per_class: Optional[int] = None
):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)
    ])

    if is_real_data:
        if dataset_name.lower() == "fashionmnist":
            dataset = FashionMNIST(root="data", train=False, download=True, transform=transform)
        elif dataset_name.lower() == "cifar10":
            dataset = CIFAR10(root="data", train=False, download=True, transform=transform)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Load samples by class to match generated samples structure
        if samples_per_class is not None:
            num_classes = num_images // samples_per_class
            images_by_class = []
            
            # Group dataset by class
            class_samples = {i: [] for i in range(num_classes)}
            for idx, (img, label) in enumerate(dataset):
                if len(class_samples[label]) < samples_per_class:
                    class_samples[label].append(img)
                # Stop when we have enough samples for all classes
                if all(len(samples) >= samples_per_class for samples in class_samples.values()):
                    break
            
            # Arrange samples in same order as generated samples (class by class)
            for class_id in range(num_classes):
                class_imgs = class_samples[class_id][:samples_per_class]
                images_by_class.extend(class_imgs)
            
            return torch.stack(images_by_class)
        else:
            # Fallback to sequential loading
            return torch.stack([dataset[i][0] for i in range(num_images)])
    else:
        # Load from image grid
        grid_img = Image.open(path_or_name).convert("RGB")
        grid_tensor = transforms.ToTensor()(grid_img)
        
        if samples_per_class is None:
            raise ValueError("`samples_per_class` must be provided for loading an image grid.")

        num_rows = num_images // samples_per_class  # 10 classes
        c, grid_h, grid_w = grid_tensor.shape
        
        # Calculate actual cell size from grid dimensions
        cell_h = grid_h // num_rows
        cell_w = grid_w // samples_per_class
        
        images = []
        for row in range(num_rows):
            for col in range(samples_per_class):
                # Calculate position based on actual grid layout
                top = row * cell_h
                left = col * cell_w
                img = grid_tensor[:, top:top+cell_h, left:left+cell_w]
                # Resize to target size
                img = transforms.functional.resize(img, (size, size))
                images.append(img)
        
        return torch.stack(images)


def preprocess_for_lpips(images: torch.Tensor, target_size: int = 64) -> torch.Tensor:
    """Prepares a batch of images for LPIPS evaluation."""
    # Ensure 3 channels
    if images.size(1) == 1:
        images = images.repeat(1, 3, 1, 1)
    
    # Resize to a safe size for AlexNet
    images = F.interpolate(images, size=(target_size, target_size), mode='bilinear', align_corners=False)
    
    # Normalize to [-1, 1] range, as expected by LPIPS
    return images * 2 - 1


def main(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_fn = lpips.LPIPS(net='alex').to(device)

    dataset_name = config.get("dataset_name", config.get("data", {}).get("dataset_name", "fashionmnist"))
    samples_per_class = config.get("samples_per_class")
    
    generated_images = load_images(
        config['generated_path'], 
        config['num_samples'], 
        config['image_size'], 
        dataset_name, 
        is_real_data=False,
        samples_per_class=samples_per_class
    )
    real_images = load_images(
        dataset_name, 
        config['num_samples'], 
        config['image_size'], 
        dataset_name, 
        is_real_data=True,
        samples_per_class=samples_per_class
    )

    generated_images = generated_images.to(device)
    real_images = real_images.to(device)

    # Preprocess for LPIPS
    generated_lpips = preprocess_for_lpips(generated_images)
    real_lpips = preprocess_for_lpips(real_images)

    # Compute metrics
    psnr_val = psnr(generated_images, real_images)
    ssim_val = ssim_simple(generated_images, real_images)
    lpips_val = lpips_fn(generated_lpips, real_lpips).mean()

    results = {
        "PSNR": f"{psnr_val:.4f}",
        "SSIM": f"{ssim_val:.4f}",
        "LPIPS": f"{lpips_val:.4f}"
    }
    
    print(f"PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}, LPIPS: {lpips_val:.4f}")

    # Save results to a file
    out_dir = Path(config['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.yaml", 'w') as f:
        yaml.dump(results, f)

    
    if samples_per_class is not None:
        num_classes = config['num_samples'] // samples_per_class
        comparison_samples = []
        
        for class_id in range(min(num_classes, 5)):  # Show first 5 classes to keep grid manageable
            # Take first 2 samples of each class
            start_idx = class_id * samples_per_class
            real_class_samples = real_images[start_idx:start_idx + 2]
            gen_class_samples = generated_images[start_idx:start_idx + 2]
            
            # Alternate real and generated for this class
            for i in range(2):
                comparison_samples.append(real_class_samples[i])
                comparison_samples.append(gen_class_samples[i])
        
        comparison_grid = torch.stack(comparison_samples)
        save_image(comparison_grid, out_dir / "comparison_grid.png", nrow=4, 
                  normalize=True, scale_each=True)
    else:
        # Fallback to simple comparison
        comparison_grid = torch.cat([real_images[:8], generated_images[:8]], 0)
        save_image(comparison_grid, out_dir / "comparison_grid.png", nrow=8)
    
    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the evaluation config file.")
    args = parser.parse_args()
    main(args.config)
