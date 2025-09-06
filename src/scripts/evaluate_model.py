import argparse
from pathlib import Path

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


def load_images(path: str, num_images: int, size: int, dataset_name: str):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)
    ])
    if dataset_name.lower() == "fashionmnist":
        dataset = FashionMNIST(root="data", train=False, download=True, transform=transform)
        return torch.stack([dataset[i][0] for i in range(num_images)])
    elif dataset_name.lower() == "cifar10":
        dataset = CIFAR10(root="data", train=False, download=True, transform=transform)
        return torch.stack([dataset[i][0] for i in range(num_images)])
    else:
        grid_img = Image.open(path).convert("RGB")
        grid_tensor = transforms.ToTensor()(grid_img)
        
        c, h, w = grid_tensor.shape
        grid_size = int(num_images**0.5)
        img_h, img_w = h // grid_size, w // grid_size
        
        images = []
        for i in range(grid_size):
            for j in range(grid_size):
                top, left = i * img_h, j * img_w
                img = grid_tensor[:, top:top+img_h, left:left+img_w]
                images.append(transforms.functional.resize(img, (size, size)))
        
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

    dataset_name = config.get("dataset_name", "fashionmnist") # Default to fashionmnist
    generated_images = load_images(config['generated_path'], config['num_samples'], config['image_size'], dataset_name)
    real_images = load_images(dataset_name, config['num_samples'], config['image_size'], dataset_name)

    generated_images = generated_images.to(device)
    real_images = real_images.to(device)

    # Preprocess for LPIPS
    generated_lpips = preprocess_for_lpips(generated_images)
    real_lpips = preprocess_for_lpips(real_images)

    print(f"Shape for LPIPS (generated): {generated_lpips.shape}")
    print(f"Shape for LPIPS (real): {real_lpips.shape}")

    # Metrics
    psnr_val = psnr(generated_images, real_images)
    ssim_val = ssim_simple(generated_images, real_images)
    lpips_val = lpips_fn(generated_lpips, real_lpips).mean()

    results = {
        "PSNR": f"{psnr_val:.4f}",
        "SSIM": f"{ssim_val:.4f}",
        "LPIPS": f"{lpips_val:.4f}"
    }
    print("Evaluation Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")

    # Save results to a file
    out_dir = Path(config['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.yaml", 'w') as f:
        yaml.dump(results, f)

    # Save comparison grid
    comparison_grid = torch.cat([real_images[:8], generated_images[:8]], 0)
    save_image(comparison_grid, out_dir / "comparison_grid.png", nrow=8)
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the evaluation config file.")
    args = parser.parse_args()
    main(args.config)
