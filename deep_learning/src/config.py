"""
PulmoSeg Deep Learning - Configurações Centralizadas
Hiperparâmetros, paths e configurações do sistema
"""

import torch
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Config:
    """Configurações centralizadas do projeto"""

    # ==================== PATHS ====================
    DATASET_PATH: Path = Path('LIDC-IDRI-slices')
    CHECKPOINTS_DIR: Path = Path('checkpoints')
    RESULTS_DIR: Path = Path('results/dl_models')
    TENSORBOARD_DIR: Path = Path('runs')

    # ==================== DATASET ====================
    TRAIN_SPLIT: float = 0.70
    VAL_SPLIT: float = 0.15
    TEST_SPLIT: float = 0.15
    RANDOM_SEED: int = 42
    MIN_MASKS_REQUIRED: int = 2  # Mínimo de máscaras válidas para consenso

    # ==================== MODEL ====================
    # Arquiteturas disponíveis: 'unet', 'attention_unet', 'unet_plus_plus'
    MODEL_NAME: str = 'unet'
    ENCODER_NAME: str = 'resnet34'  # Encoder pré-treinado (ImageNet)
    ENCODER_WEIGHTS: str = 'imagenet'  # ou None para treinar do zero
    IN_CHANNELS: int = 3  # RGB (converter grayscale para 3 canais)
    OUT_CHANNELS: int = 1  # Máscara binária

    # ==================== TRAINING ====================
    BATCH_SIZE: int = 8  # Ajustar baseado em RAM disponível
    NUM_EPOCHS: int = 50
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-5

    # Early Stopping
    EARLY_STOPPING_PATIENCE: int = 15

    # Learning Rate Scheduler
    LR_SCHEDULER: str = 'ReduceLROnPlateau'  # ou 'CosineAnnealingLR'
    LR_PATIENCE: int = 5
    LR_FACTOR: float = 0.5
    LR_MIN: float = 1e-7

    # ==================== DATA AUGMENTATION ====================
    AUGMENTATION_ENABLED: bool = True
    AUG_ROTATION_LIMIT: int = 15  # ±15 graus
    AUG_SCALE_LIMIT: float = 0.1  # 0.9-1.1
    AUG_BRIGHTNESS_LIMIT: float = 0.2
    AUG_CONTRAST_LIMIT: float = 0.2
    AUG_HORIZONTAL_FLIP: float = 0.5
    AUG_VERTICAL_FLIP: float = 0.5
    AUG_ELASTIC_TRANSFORM: bool = True

    # ==================== LOSS FUNCTION ====================
    LOSS_FUNCTION: str = 'combined'  # 'dice', 'bce', 'focal', 'combined'
    DICE_WEIGHT: float = 0.5
    BCE_WEIGHT: float = 0.5

    # ==================== DEVICE ====================
    @staticmethod
    def get_device():
        """Detecta automaticamente o melhor device disponível"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Apple Silicon (M1/M2/M3)
            return torch.device('mps')
        else:
            return torch.device('cpu')

    # ==================== OPTIMIZATION ====================
    # Mixed Precision Training (para economizar memória)
    USE_MIXED_PRECISION: bool = True  # Habilitado para Apple MPS

    # Gradient Clipping (evitar exploding gradients)
    GRADIENT_CLIP_VAL: float = 1.0

    # Número de workers para DataLoader
    NUM_WORKERS: int = 4  # Ajustar baseado em CPU cores

    # ==================== VISUALIZATION ====================
    VISUALIZE_EVERY_N_EPOCHS: int = 5
    NUM_SAMPLES_TO_VISUALIZE: int = 4

    # ==================== EVALUATION ====================
    METRICS: list = None  # Será definido como ['dice', 'iou', 'precision', 'recall']

    def __post_init__(self):
        """Inicialização pós-criação"""
        if self.METRICS is None:
            self.METRICS = ['dice', 'iou', 'precision', 'recall']

        # Criar diretórios se não existirem
        self.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)

    def to_dict(self):
        """Converte configuração para dicionário"""
        return {k: str(v) if isinstance(v, Path) else v
                for k, v in self.__dict__.items()
                if not k.startswith('_')}

    def __repr__(self):
        """Representação string formatada"""
        config_str = "="*60 + "\n"
        config_str += "PulmoSeg Deep Learning Configuration\n"
        config_str += "="*60 + "\n"

        for key, value in self.to_dict().items():
            if not callable(value):
                config_str += f"{key}: {value}\n"

        config_str += "="*60
        return config_str


# Instância global de configuração
cfg = Config()


if __name__ == '__main__':
    # Teste de configuração
    print(cfg)
    print(f"\nDevice detectado: {cfg.get_device()}")
