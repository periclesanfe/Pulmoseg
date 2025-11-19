"""
PulmoSeg Deep Learning - Model Architectures
Wrappers para arquiteturas de segmentação usando segmentation_models_pytorch
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from src.config import cfg


def create_model(architecture='unet', encoder_name='resnet34', encoder_weights='imagenet',
                in_channels=3, out_channels=1):
    """
    Factory function para criar modelos de segmentação.

    Args:
        architecture: 'unet', 'unetplusplus', 'manet' (Attention U-Net), 'fpn', 'pspnet', etc
        encoder_name: Nome do encoder pré-treinado (resnet34, resnet50, efficientnet-b0, etc)
        encoder_weights: 'imagenet' para usar pesos pré-treinados, None para treinar do zero
        in_channels: Número de canais de entrada (3 para RGB)
        out_channels: Número de canais de saída (1 para segmentação binária)

    Returns:
        Modelo PyTorch
    """
    architecture = architecture.lower()

    # Dicionário de arquiteturas disponíveis
    model_classes = {
        'unet': smp.Unet,
        'unetplusplus': smp.UnetPlusPlus,
        'unet++': smp.UnetPlusPlus,  # Alias
        'manet': smp.MAnet,  # Attention U-Net
        'attention_unet': smp.MAnet,  # Alias
        'fpn': smp.FPN,
        'pspnet': smp.PSPNet,
        'deeplabv3': smp.DeepLabV3,
        'deeplabv3plus': smp.DeepLabV3Plus,
        'pan': smp.PAN,
        'linknet': smp.Linknet,
    }

    if architecture not in model_classes:
        raise ValueError(
            f"Arquitetura '{architecture}' não reconhecida. "
            f"Opções: {list(model_classes.keys())}"
        )

    ModelClass = model_classes[architecture]

    # Criar modelo
    model = ModelClass(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_channels,
        activation=None  # Usar sigmoid/softmax na loss function
    )

    return model


class PulmoSegModel(nn.Module):
    """
    Wrapper para modelos de segmentação com funcionalidades extras.

    Adiciona:
    - Contagem de parâmetros
    - Summary do modelo
    - Facilita inferência
    """

    def __init__(self, architecture='unet', encoder_name='resnet34',
                 encoder_weights='imagenet', in_channels=3, out_channels=1):
        super(PulmoSegModel, self).__init__()

        self.architecture = architecture
        self.encoder_name = encoder_name

        # Criar modelo base
        self.model = create_model(
            architecture=architecture,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            out_channels=out_channels
        )

    def forward(self, x):
        """Forward pass"""
        return self.model(x)

    def count_parameters(self):
        """Conta número total de parâmetros treináveis"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        """Retorna summary do modelo"""
        total_params = self.count_parameters()
        print(f"\n{'='*60}")
        print(f"Model Summary: {self.architecture.upper()}")
        print(f"{'='*60}")
        print(f"Architecture: {self.architecture}")
        print(f"Encoder: {self.encoder_name}")
        print(f"Total Parameters: {total_params:,}")
        print(f"{'='*60}\n")

        return {
            'architecture': self.architecture,
            'encoder': self.encoder_name,
            'total_params': total_params
        }

    @torch.no_grad()
    def predict(self, x, threshold=0.5):
        """
        Predição com threshold.

        Args:
            x: Input tensor [B, C, H, W]
            threshold: Threshold para binarização (padrão: 0.5)

        Returns:
            Máscara binária [B, 1, H, W]
        """
        self.eval()
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        masks = (probs > threshold).float()
        return masks


def get_model(model_name=None, encoder_name=None, encoder_weights=None):
    """
    Cria modelo usando configurações padrão de cfg.

    Args:
        model_name: Nome da arquitetura (padrão: cfg.MODEL_NAME)
        encoder_name: Nome do encoder (padrão: cfg.ENCODER_NAME)
        encoder_weights: Pesos do encoder (padrão: cfg.ENCODER_WEIGHTS)

    Returns:
        PulmoSegModel instance
    """
    model_name = model_name or cfg.MODEL_NAME
    encoder_name = encoder_name or cfg.ENCODER_NAME
    encoder_weights = encoder_weights or cfg.ENCODER_WEIGHTS

    model = PulmoSegModel(
        architecture=model_name,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=cfg.IN_CHANNELS,
        out_channels=cfg.OUT_CHANNELS
    )

    return model


if __name__ == '__main__':
    # Teste dos modelos
    print("Testando modelos de segmentação...\n")

    # Criar input de teste
    batch_size = 2
    channels = 3
    height = 256
    width = 256

    x = torch.randn(batch_size, channels, height, width)
    print(f"Input shape: {x.shape}\n")

    # Testar diferentes arquiteturas
    architectures = ['unet', 'unetplusplus', 'manet']

    for arch in architectures:
        print(f"\nTestando {arch.upper()}:")
        print("-" * 60)

        model = PulmoSegModel(
            architecture=arch,
            encoder_name='resnet34',
            encoder_weights=None,  # Sem pesos pré-treinados para teste rápido
            in_channels=3,
            out_channels=1
        )

        # Summary
        summary = model.summary()

        # Forward pass
        output = model(x)
        print(f"Output shape: {output.shape}")

        # Predição
        pred_mask = model.predict(x, threshold=0.5)
        print(f"Predicted mask shape: {pred_mask.shape}")
        print(f"Unique values in mask: {pred_mask.unique()}")

    print("\n✓ Todos os modelos funcionando corretamente!")
