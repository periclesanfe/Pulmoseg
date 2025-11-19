"""
PulmoSeg DL - Avaliação de Modelos no Test Set
"""

import argparse
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from src.config import cfg
from src.dataset import get_dataloaders
from src.augmentation import get_validation_augmentation
from src.models.unet import PulmoSegModel
from src.metrics import MetricsTracker


def evaluate_model(checkpoint_path, test_loader, device):
    """Avalia modelo no test set"""
    # Carregar checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Recriar modelo
    model_config = checkpoint.get('config', {})
    model_name = model_config.get('MODEL_NAME', 'unet')
    encoder_name = model_config.get('ENCODER_NAME', 'resnet34')

    model = PulmoSegModel(
        architecture=model_name,
        encoder_name=encoder_name,
        encoder_weights=None
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Avaliar
    metrics = MetricsTracker(metrics_list=['dice', 'iou', 'precision', 'recall'])

    results = []

    with torch.no_grad():
        for images, masks, paths in tqdm(test_loader, desc='Avaliando'):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            metrics.update(outputs, masks)

            # Salvar resultados por amostra
            for i in range(len(paths)):
                results.append({'path': paths[i]})

    avg_metrics = metrics.get_averages()

    return avg_metrics, results


def main():
    parser = argparse.ArgumentParser(description='Avaliar modelo no test set')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Caminho para o checkpoint (.pth)')
    args = parser.parse_args()

    device = cfg.get_device()

    # Carregar test set
    _, _, test_loader, _ = get_dataloaders(
        val_transform=get_validation_augmentation()
    )

    # Avaliar
    metrics, results = evaluate_model(args.checkpoint, test_loader, device)

    # Print resultados
    print(f"\n{'='*60}")
    print("Resultados no Test Set")
    print(f"{'='*60}")
    for metric_name, value in metrics.items():
        print(f"{metric_name.upper()}: {value:.4f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
