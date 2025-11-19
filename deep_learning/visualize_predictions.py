"""
PulmoSeg DL - Visualização de Predições
"""

import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.config import cfg
from src.dataset import get_dataloaders
from src.augmentation import get_validation_augmentation
from src.models.unet import PulmoSegModel


def visualize_predictions(checkpoint_path, num_samples=10, save_dir=None):
    """Visualiza predições do modelo"""
    device = cfg.get_device()

    # Carregar modelo
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get('config', {})

    model = PulmoSegModel(
        architecture=model_config.get('MODEL_NAME', 'unet'),
        encoder_name=model_config.get('ENCODER_NAME', 'resnet34'),
        encoder_weights=None
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Carregar test set
    _, _, test_loader, _ = get_dataloaders(
        val_transform=get_validation_augmentation()
    )

    # Visualizar
    save_dir = save_dir or Path('results/dl_models/visualizations')
    save_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    with torch.no_grad():
        for images, masks, paths in test_loader:
            if count >= num_samples:
                break

            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5

            # Visualizar batch
            for i in range(min(images.shape[0], num_samples - count)):
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # Imagem original
                img = images[i].cpu().permute(1, 2, 0).numpy()
                axes[0].imshow(img)
                axes[0].set_title('Imagem Original')
                axes[0].axis('off')

                # Ground truth
                gt = masks[i, 0].cpu().numpy()
                axes[1].imshow(gt, cmap='gray')
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')

                # Predição
                pred = preds[i, 0].cpu().numpy()
                axes[2].imshow(pred, cmap='gray')
                axes[2].set_title('Predição (DL)')
                axes[2].axis('off')

                plt.tight_layout()
                plt.savefig(save_dir / f'prediction_{count}.png', dpi=150)
                plt.close()

                count += 1

    print(f"✓ {count} visualizações salvas em {save_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num-samples', type=int, default=10)
    args = parser.parse_args()

    visualize_predictions(args.checkpoint, args.num_samples)


if __name__ == '__main__':
    main()
