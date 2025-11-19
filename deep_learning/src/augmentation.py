"""
PulmoSeg Deep Learning - Data Augmentation
Pipeline de augmentation otimizado para imagens médicas usando Albumentations
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import cfg


def get_training_augmentation():
    """
    Retorna pipeline de augmentation para TREINO.

    Augmentations médicas conservadoras:
    - Rotação limitada (±15°)
    - Flip horizontal/vertical
    - Escala limitada
    - Ajustes de brilho/contraste leves
    - Elastic deformation (simula variações anatômicas)
    """
    if not cfg.AUGMENTATION_ENABLED:
        return A.Compose([
            A.NoOp()  # Sem augmentation
        ])

    transforms = [
        # Transformações geométricas
        A.ShiftScaleRotate(
            shift_limit=0.05,  # Shift leve
            scale_limit=cfg.AUG_SCALE_LIMIT,  # 0.9-1.1
            rotate_limit=cfg.AUG_ROTATION_LIMIT,  # ±15°
            p=0.7,
            border_mode=0  # Constant padding
        ),

        # Flips
        A.HorizontalFlip(p=cfg.AUG_HORIZONTAL_FLIP),
        A.VerticalFlip(p=cfg.AUG_VERTICAL_FLIP),

        # Transformações de intensidade (leves para imagens médicas)
        A.RandomBrightnessContrast(
            brightness_limit=cfg.AUG_BRIGHTNESS_LIMIT,  # ±20%
            contrast_limit=cfg.AUG_CONTRAST_LIMIT,  # ±20%
            p=0.5
        ),

        # Elastic deformation (simula variações anatômicas naturais)
        A.ElasticTransform(
            alpha=50,
            sigma=8,
            alpha_affine=8,
            p=0.3 if cfg.AUG_ELASTIC_TRANSFORM else 0.0
        ) if cfg.AUG_ELASTIC_TRANSFORM else A.NoOp(),

        # Grid distortion (deformação de grade)
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.1,
            p=0.2
        ),

        # Optical distortion
        A.OpticalDistortion(
            distort_limit=0.05,
            shift_limit=0.05,
            p=0.2
        ),

        # Random gamma (ajuste de gamma para variação de exposição)
        A.RandomGamma(
            gamma_limit=(80, 120),
            p=0.3
        ),
    ]

    # Remover NoOp se presente
    transforms = [t for t in transforms if not isinstance(t, A.NoOp)]

    return A.Compose(transforms)


def get_validation_augmentation():
    """
    Retorna pipeline de augmentation para VALIDAÇÃO/TESTE.

    Apenas normalização, sem augmentation.
    """
    return A.Compose([
        A.NoOp()  # Sem augmentation em validação/teste
    ])


def get_preprocessing():
    """
    Pré-processamento padrão (normalização, conversão para tensor).

    Nota: Normalização [0,1] já é feita no dataset.py
    Esta função fica disponível para futuras normalizações adicionais.
    """
    return A.Compose([
        A.NoOp()
    ])


def visualize_augmentation(image, mask, num_augmentations=5):
    """
    Visualiza exemplos de augmentation para debugging.

    Args:
        image: Imagem numpy [H, W] ou [H, W, 3]
        mask: Máscara numpy [H, W]
        num_augmentations: Número de variações a gerar
    """
    import matplotlib.pyplot as plt
    import numpy as np

    aug_pipeline = get_training_augmentation()

    fig, axes = plt.subplots(2, num_augmentations + 1, figsize=(20, 8))

    # Original
    axes[0, 0].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[1, 0].imshow(mask, cmap='gray')
    axes[1, 0].set_title('Original Mask')
    axes[1, 0].axis('off')

    # Augmented
    for i in range(num_augmentations):
        augmented = aug_pipeline(image=image, mask=mask)
        aug_img = augmented['image']
        aug_mask = augmented['mask']

        axes[0, i+1].imshow(aug_img, cmap='gray' if len(aug_img.shape) == 2 else None)
        axes[0, i+1].set_title(f'Augmented {i+1}')
        axes[0, i+1].axis('off')

        axes[1, i+1].imshow(aug_mask, cmap='gray')
        axes[1, i+1].set_title(f'Mask {i+1}')
        axes[1, i+1].axis('off')

    plt.tight_layout()
    plt.savefig('results/augmentation_examples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Exemplos de augmentation salvos em: results/augmentation_examples.png")


if __name__ == '__main__':
    # Teste do pipeline de augmentation
    import cv2
    import numpy as np
    from pathlib import Path

    print("Testando pipeline de augmentation...")

    # Carregar uma imagem de exemplo do dataset
    dataset_path = Path('LIDC-IDRI-slices')
    patient_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith('LIDC-IDRI-')]

    if patient_dirs:
        for patient_dir in patient_dirs[:5]:  # Tentar primeiros 5 pacientes
            nodule_dirs = [d for d in patient_dir.iterdir() if d.is_dir() and d.name.startswith('nodule-')]
            if nodule_dirs:
                nodule_dir = nodule_dirs[0]
                images_dir = nodule_dir / 'images'
                if images_dir.exists():
                    image_files = list(images_dir.glob('*.png'))
                    if image_files:
                        # Carregar imagem e máscara
                        image_path = image_files[0]
                        mask_path = nodule_dir / 'mask-0' / image_path.name

                        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else np.zeros_like(image)

                        if image is not None:
                            print(f"✓ Imagem carregada: {image_path}")
                            print(f"  Shape: {image.shape}")

                            # Visualizar augmentations
                            visualize_augmentation(image, mask, num_augmentations=5)
                            break
        else:
            print("❌ Não foi possível encontrar imagem válida para teste")
    else:
        print("❌ Dataset não encontrado")

    # Printar configurações de augmentation
    print("\nConfigurações de Augmentation:")
    print(f"  Augmentation habilitado: {cfg.AUGMENTATION_ENABLED}")
    print(f"  Rotação: ±{cfg.AUG_ROTATION_LIMIT}°")
    print(f"  Escala: {1 - cfg.AUG_SCALE_LIMIT} - {1 + cfg.AUG_SCALE_LIMIT}")
    print(f"  Brilho/Contraste: ±{cfg.AUG_BRIGHTNESS_LIMIT}")
    print(f"  Flip horizontal: {cfg.AUG_HORIZONTAL_FLIP}")
    print(f"  Flip vertical: {cfg.AUG_VERTICAL_FLIP}")
    print(f"  Elastic transform: {cfg.AUG_ELASTIC_TRANSFORM}")
