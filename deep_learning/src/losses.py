"""
PulmoSeg Deep Learning - Loss Functions
Implementações de loss functions para segmentação médica
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss para segmentação binária.

    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Dice Loss = 1 - Dice
    """

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Tensor [B, 1, H, W] (logits ou probabilidades)
            targets: Tensor [B, 1, H, W] (ground truth binário [0, 1])
        """
        # Aplicar sigmoid se necessário (para trabalhar com logits)
        predictions = torch.sigmoid(predictions)

        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Calcular interseção e soma
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)

        return 1 - dice


class BCEDiceLoss(nn.Module):
    """
    Combinação de Binary Cross-Entropy e Dice Loss.

    Loss = α * BCE + (1-α) * Dice
    """

    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Tensor [B, 1, H, W] (logits)
            targets: Tensor [B, 1, H, W] (ground truth [0, 1])
        """
        bce_loss = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss para lidar com desbalanceamento de classes.

    FL(p_t) = -α * (1 - p_t)^γ * log(p_t)

    γ > 0 reduz a perda relativa para exemplos bem classificados,
    focando mais nos exemplos difíceis.
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Tensor [B, 1, H, W] (logits)
            targets: Tensor [B, 1, H, W] (ground truth [0, 1])
        """
        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')

        # Probabilidades
        probs = torch.sigmoid(predictions)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma

        # Focal loss
        focal_loss = self.alpha * focal_term * bce_loss

        return focal_loss.mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalização do Dice Loss.

    Útil para lidar com desbalanceamento entre FP e FN.
    α controla penalização de FP, β controla FN.
    """

    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Tensor [B, 1, H, W] (logits)
            targets: Tensor [B, 1, H, W] (ground truth [0, 1])
        """
        predictions = torch.sigmoid(predictions)

        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives, False Negatives
        TP = (predictions * targets).sum()
        FP = ((1 - targets) * predictions).sum()
        FN = (targets * (1 - predictions)).sum()

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        return 1 - tversky


def get_loss_function(loss_name='combined', dice_weight=0.5, bce_weight=0.5):
    """
    Factory function para criar loss function baseada no nome.

    Args:
        loss_name: 'dice', 'bce', 'focal', 'combined', 'tversky'
        dice_weight: Peso do Dice Loss (se combined)
        bce_weight: Peso do BCE Loss (se combined)

    Returns:
        Loss function instance
    """
    loss_name = loss_name.lower()

    if loss_name == 'dice':
        return DiceLoss()
    elif loss_name == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_name == 'focal':
        return FocalLoss()
    elif loss_name == 'combined':
        return BCEDiceLoss(dice_weight=dice_weight, bce_weight=bce_weight)
    elif loss_name == 'tversky':
        return TverskyLoss()
    else:
        raise ValueError(f"Loss desconhecida: {loss_name}. "
                        f"Opções: 'dice', 'bce', 'focal', 'combined', 'tversky'")


if __name__ == '__main__':
    # Teste das loss functions
    print("Testando loss functions...")

    # Criar tensores de exemplo
    batch_size, channels, height, width = 4, 1, 256, 256

    # Predictions (logits)
    predictions = torch.randn(batch_size, channels, height, width)

    # Targets (ground truth binário [0, 1])
    targets = torch.randint(0, 2, (batch_size, channels, height, width)).float()

    # Testar cada loss
    losses = {
        'Dice': DiceLoss(),
        'BCE': nn.BCEWithLogitsLoss(),
        'Focal': FocalLoss(),
        'Combined (BCE+Dice)': BCEDiceLoss(),
        'Tversky': TverskyLoss()
    }

    print(f"\nShape das predictions: {predictions.shape}")
    print(f"Shape dos targets: {targets.shape}")
    print(f"Range predictions: [{predictions.min():.2f}, {predictions.max():.2f}]")
    print(f"Range targets: [{targets.min():.0f}, {targets.max():.0f}]")
    print("\nResultados:")

    for name, loss_fn in losses.items():
        loss_value = loss_fn(predictions, targets)
        print(f"  {name:25s}: {loss_value.item():.6f}")

    print("\n✓ Todas as loss functions funcionando corretamente!")
