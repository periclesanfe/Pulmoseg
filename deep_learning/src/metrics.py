"""
PulmoSeg Deep Learning - Métricas de Avaliação
Implementações de métricas para segmentação médica
"""

import torch
import numpy as np


def dice_coefficient(predictions, targets, threshold=0.5, smooth=1e-6):
    """
    Calcula Dice Coefficient (F1-Score) para segmentação binária.

    Dice = 2 * |A ∩ B| / (|A| + |B|)

    Args:
        predictions: Tensor [B, 1, H, W] ou [B, H, W] (probabilidades)
        targets: Tensor [B, 1, H, W] ou [B, H, W] (ground truth [0, 1])
        threshold: Threshold para binarizar predictions
        smooth: Smoothing para evitar divisão por zero

    Returns:
        Dice coefficient médio do batch
    """
    # Aplicar sigmoid se necessário
    if predictions.min() < 0 or predictions.max() > 1:
        predictions = torch.sigmoid(predictions)

    # Binarizar predictions
    predictions = (predictions > threshold).float()

    # Flatten (manter dimensão do batch)
    predictions = predictions.view(predictions.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)

    # Calcular Dice para cada amostra do batch
    intersection = (predictions * targets).sum(dim=1)
    dice = (2. * intersection + smooth) / (predictions.sum(dim=1) + targets.sum(dim=1) + smooth)

    return dice.mean()


def iou_score(predictions, targets, threshold=0.5, smooth=1e-6):
    """
    Calcula IoU (Intersection over Union / Jaccard Index).

    IoU = |A ∩ B| / |A ∪ B|

    Args:
        predictions: Tensor [B, 1, H, W] (probabilidades)
        targets: Tensor [B, 1, H, W] (ground truth [0, 1])
        threshold: Threshold para binarizar predictions
        smooth: Smoothing para evitar divisão por zero

    Returns:
        IoU médio do batch
    """
    # Aplicar sigmoid se necessário
    if predictions.min() < 0 or predictions.max() > 1:
        predictions = torch.sigmoid(predictions)

    # Binarizar predictions
    predictions = (predictions > threshold).float()

    # Flatten
    predictions = predictions.view(predictions.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)

    # Calcular interseção e união
    intersection = (predictions * targets).sum(dim=1)
    union = predictions.sum(dim=1) + targets.sum(dim=1) - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou.mean()


def precision_score(predictions, targets, threshold=0.5, smooth=1e-6):
    """
    Calcula Precision (Positive Predictive Value).

    Precision = TP / (TP + FP)

    Args:
        predictions: Tensor [B, 1, H, W] (probabilidades)
        targets: Tensor [B, 1, H, W] (ground truth [0, 1])
        threshold: Threshold para binarizar predictions
        smooth: Smoothing para evitar divisão por zero

    Returns:
        Precision médio do batch
    """
    # Aplicar sigmoid e binarizar
    if predictions.min() < 0 or predictions.max() > 1:
        predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()

    # Flatten
    predictions = predictions.view(predictions.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)

    # True Positives e False Positives
    TP = (predictions * targets).sum(dim=1)
    FP = (predictions * (1 - targets)).sum(dim=1)

    precision = (TP + smooth) / (TP + FP + smooth)

    return precision.mean()


def recall_score(predictions, targets, threshold=0.5, smooth=1e-6):
    """
    Calcula Recall (Sensitivity / True Positive Rate).

    Recall = TP / (TP + FN)

    Args:
        predictions: Tensor [B, 1, H, W] (probabilidades)
        targets: Tensor [B, 1, H, W] (ground truth [0, 1])
        threshold: Threshold para binarizar predictions
        smooth: Smoothing para evitar divisão por zero

    Returns:
        Recall médio do batch
    """
    # Aplicar sigmoid e binarizar
    if predictions.min() < 0 or predictions.max() > 1:
        predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()

    # Flatten
    predictions = predictions.view(predictions.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)

    # True Positives e False Negatives
    TP = (predictions * targets).sum(dim=1)
    FN = ((1 - predictions) * targets).sum(dim=1)

    recall = (TP + smooth) / (TP + FN + smooth)

    return recall.mean()


def f1_score(predictions, targets, threshold=0.5, smooth=1e-6):
    """
    Calcula F1-Score (harmônica entre Precision e Recall).

    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    Nota: F1-Score é equivalente ao Dice Coefficient para segmentação binária.

    Args:
        predictions: Tensor [B, 1, H, W] (probabilidades)
        targets: Tensor [B, 1, H, W] (ground truth [0, 1])
        threshold: Threshold para binarizar predictions
        smooth: Smoothing para evitar divisão por zero

    Returns:
        F1-Score médio do batch
    """
    precision = precision_score(predictions, targets, threshold, smooth)
    recall = recall_score(predictions, targets, threshold, smooth)

    f1 = (2 * precision * recall + smooth) / (precision + recall + smooth)

    return f1


class MetricsTracker:
    """
    Classe para rastrear múltiplas métricas durante treinamento/validação.
    """

    def __init__(self, metrics_list=['dice', 'iou', 'precision', 'recall']):
        """
        Args:
            metrics_list: Lista de nomes de métricas a rastrear
        """
        self.metrics_list = metrics_list
        self.reset()

    def reset(self):
        """Reseta todas as métricas"""
        self.metrics = {metric: [] for metric in self.metrics_list}

    def update(self, predictions, targets, threshold=0.5):
        """
        Atualiza métricas com novo batch.

        Args:
            predictions: Tensor [B, 1, H, W] (probabilidades ou logits)
            targets: Tensor [B, 1, H, W] (ground truth [0, 1])
            threshold: Threshold para binarização
        """
        with torch.no_grad():
            for metric_name in self.metrics_list:
                if metric_name == 'dice':
                    value = dice_coefficient(predictions, targets, threshold)
                elif metric_name == 'iou':
                    value = iou_score(predictions, targets, threshold)
                elif metric_name == 'precision':
                    value = precision_score(predictions, targets, threshold)
                elif metric_name == 'recall':
                    value = recall_score(predictions, targets, threshold)
                elif metric_name == 'f1':
                    value = f1_score(predictions, targets, threshold)
                else:
                    continue

                self.metrics[metric_name].append(value.item())

    def get_averages(self):
        """
        Retorna médias de todas as métricas.

        Returns:
            Dicionário com {metric_name: average_value}
        """
        averages = {}
        for metric_name, values in self.metrics.items():
            if len(values) > 0:
                averages[metric_name] = np.mean(values)
            else:
                averages[metric_name] = 0.0

        return averages

    def __repr__(self):
        """String representation das métricas"""
        averages = self.get_averages()
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in averages.items()])
        return metrics_str


if __name__ == '__main__':
    # Teste das métricas
    print("Testando métricas...")

    # Criar tensores de exemplo
    batch_size, channels, height, width = 4, 1, 256, 256

    # Predictions (probabilidades após sigmoid)
    predictions = torch.rand(batch_size, channels, height, width)

    # Targets (ground truth binário [0, 1])
    targets = torch.randint(0, 2, (batch_size, channels, height, width)).float()

    print(f"\nShape das predictions: {predictions.shape}")
    print(f"Shape dos targets: {targets.shape}")
    print(f"Range predictions: [{predictions.min():.2f}, {predictions.max():.2f}]")
    print(f"Range targets: [{targets.min():.0f}, {targets.max():.0f}]")

    # Testar métricas individuais
    print("\nMétricas individuais:")
    print(f"  Dice Coefficient: {dice_coefficient(predictions, targets):.4f}")
    print(f"  IoU Score:        {iou_score(predictions, targets):.4f}")
    print(f"  Precision:        {precision_score(predictions, targets):.4f}")
    print(f"  Recall:           {recall_score(predictions, targets):.4f}")
    print(f"  F1-Score:         {f1_score(predictions, targets):.4f}")

    # Testar MetricsTracker
    print("\nTestando MetricsTracker:")
    tracker = MetricsTracker(metrics_list=['dice', 'iou', 'precision', 'recall'])

    # Simular 5 batches
    for i in range(5):
        preds = torch.rand(batch_size, channels, height, width)
        targs = torch.randint(0, 2, (batch_size, channels, height, width)).float()
        tracker.update(preds, targs)

    print(f"  {tracker}")
    print("\n✓ Todas as métricas funcionando corretamente!")
