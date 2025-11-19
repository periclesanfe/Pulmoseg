"""
PulmoSeg Deep Learning - Dataset e DataLoaders
PyTorch Dataset customizado para LIDC-IDRI com split train/val/test
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import random
from sklearn.model_selection import train_test_split

from src.config import cfg


class PulmoSegDataset(Dataset):
    """
    Dataset customizado para segmentação de nódulos pulmonares.

    Carrega imagens e máscaras de consenso do LIDC-IDRI dataset.
    """

    def __init__(self, samples: List[Dict], transform=None, is_train: bool = True):
        """
        Args:
            samples: Lista de dicionários com 'image_path' e 'mask_paths'
            transform: Transformações/augmentations do albumentations
            is_train: Se True, aplica augmentation (se transform fornecido)
        """
        self.samples = samples
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Retorna:
            image: Tensor [3, H, W] (RGB)
            mask: Tensor [1, H, W] (máscara binária)
            image_path: String (para debugging)
        """
        sample = self.samples[idx]
        image_path = sample['image_path']
        mask_paths = sample['mask_paths']

        # Carregar imagem
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Não foi possível carregar: {image_path}")

        # Criar máscara de consenso
        mask = self._create_consensus_mask(mask_paths)

        # Aplicar transformações/augmentation
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Converter grayscale para RGB (3 canais) para encoders pré-treinados
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Normalizar imagem [0, 255] -> [0, 1]
        image = image.astype(np.float32) / 255.0

        # Normalizar máscara para binário [0, 1]
        mask = (mask > 127).astype(np.float32)

        # Converter para tensores PyTorch
        # Imagem: [H, W, 3] -> [3, H, W]
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        # Máscara: [H, W] -> [1, H, W]
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask, str(image_path)

    def _create_consensus_mask(self, mask_paths: List[Path], majority_threshold: int = 2) -> np.ndarray:
        """
        Cria máscara de consenso a partir de múltiplas anotações.

        Args:
            mask_paths: Lista de caminhos para mask-0 até mask-3
            majority_threshold: Número mínimo de radiologistas que devem concordar

        Returns:
            Máscara de consenso binária (0 ou 255)
        """
        masks = []

        for mask_path in mask_paths:
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None and cv2.countNonZero(mask) > 0:
                    # Normalizar para binário (0 ou 1)
                    _, binary_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
                    masks.append(binary_mask)

        if len(masks) == 0:
            # Retornar máscara vazia se não há anotações
            # (assumindo tamanho padrão 512x512 do LIDC-IDRI)
            return np.zeros((512, 512), dtype=np.uint8)

        # Somar todas as máscaras e aplicar threshold de maioria
        consensus = np.sum(masks, axis=0)
        consensus_mask = (consensus >= majority_threshold).astype(np.uint8) * 255

        return consensus_mask


def scan_dataset(dataset_path: Path, min_masks: int = 2) -> List[Dict]:
    """
    Escaneia o dataset LIDC-IDRI e retorna lista de amostras válidas.

    Args:
        dataset_path: Caminho para LIDC-IDRI-slices/
        min_masks: Número mínimo de máscaras válidas requeridas

    Returns:
        Lista de dicionários com {'image_path', 'mask_paths', 'patient_id', 'nodule_id', 'slice_id'}
    """
    samples = []

    patient_dirs = sorted([d for d in dataset_path.iterdir()
                          if d.is_dir() and d.name.startswith('LIDC-IDRI-')])

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name

        nodule_dirs = sorted([d for d in patient_dir.iterdir()
                             if d.is_dir() and d.name.startswith('nodule-')])

        for nodule_dir in nodule_dirs:
            nodule_id = nodule_dir.name

            # Verificar se há imagens
            images_dir = nodule_dir / 'images'
            if not images_dir.exists():
                continue

            # Processar cada slice
            slice_files = sorted(images_dir.glob('*.png'))

            for slice_file in slice_files:
                slice_id = slice_file.stem

                # Coletar caminhos das máscaras
                mask_paths = [nodule_dir / f'mask-{i}' / slice_file.name for i in range(4)]

                # Verificar se há máscaras válidas suficientes
                valid_masks = sum(1 for mp in mask_paths
                                 if mp.exists() and cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) is not None
                                 and cv2.countNonZero(cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)) > 0)

                if valid_masks >= min_masks:
                    samples.append({
                        'image_path': slice_file,
                        'mask_paths': mask_paths,
                        'patient_id': patient_id,
                        'nodule_id': nodule_id,
                        'slice_id': slice_id
                    })

    return samples


def split_dataset_by_patients(samples: List[Dict], train_ratio: float = 0.70,
                              val_ratio: float = 0.15, test_ratio: float = 0.15,
                              random_seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Divide o dataset em train/val/test baseado em pacientes (não em slices).

    Importante: Slices do mesmo paciente ficam no mesmo split para evitar data leakage.

    Args:
        samples: Lista de amostras do scan_dataset()
        train_ratio, val_ratio, test_ratio: Proporções de split
        random_seed: Seed para reprodutibilidade

    Returns:
        train_samples, val_samples, test_samples
    """
    # Agrupar samples por paciente
    patient_samples = {}
    for sample in samples:
        patient_id = sample['patient_id']
        if patient_id not in patient_samples:
            patient_samples[patient_id] = []
        patient_samples[patient_id].append(sample)

    # Obter lista de patient IDs
    patient_ids = list(patient_samples.keys())
    random.seed(random_seed)
    random.shuffle(patient_ids)

    # Calcular split
    n_patients = len(patient_ids)
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)

    train_patients = patient_ids[:n_train]
    val_patients = patient_ids[n_train:n_train+n_val]
    test_patients = patient_ids[n_train+n_val:]

    # Coletar samples de cada split
    train_samples = [s for pid in train_patients for s in patient_samples[pid]]
    val_samples = [s for pid in val_patients for s in patient_samples[pid]]
    test_samples = [s for pid in test_patients for s in patient_samples[pid]]

    return train_samples, val_samples, test_samples


def get_dataloaders(dataset_path: Path = None,
                   train_transform=None,
                   val_transform=None,
                   batch_size: int = None,
                   num_workers: int = None) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Cria DataLoaders para train/val/test com configurações da cfg.

    Args:
        dataset_path: Caminho do dataset (padrão: cfg.DATASET_PATH)
        train_transform: Transformações para treino (padrão: augmentation habilitado)
        val_transform: Transformações para val/test (padrão: apenas normalização)
        batch_size: Tamanho do batch (padrão: cfg.BATCH_SIZE)
        num_workers: Workers do DataLoader (padrão: cfg.NUM_WORKERS)

    Returns:
        train_loader, val_loader, test_loader, stats (dicionário com estatísticas)
    """
    # Usar configurações padrão se não fornecidas
    dataset_path = dataset_path or cfg.DATASET_PATH
    batch_size = batch_size or cfg.BATCH_SIZE
    num_workers = num_workers or cfg.NUM_WORKERS

    # Escanear dataset
    print("Escaneando dataset...")
    all_samples = scan_dataset(dataset_path, min_masks=cfg.MIN_MASKS_REQUIRED)
    print(f"✓ {len(all_samples)} amostras válidas encontradas")

    # Split train/val/test por pacientes
    print("Dividindo dataset (train/val/test por pacientes)...")
    train_samples, val_samples, test_samples = split_dataset_by_patients(
        all_samples,
        train_ratio=cfg.TRAIN_SPLIT,
        val_ratio=cfg.VAL_SPLIT,
        test_ratio=cfg.TEST_SPLIT,
        random_seed=cfg.RANDOM_SEED
    )

    # Estatísticas
    stats = {
        'total_samples': len(all_samples),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'test_samples': len(test_samples),
        'train_patients': len(set(s['patient_id'] for s in train_samples)),
        'val_patients': len(set(s['patient_id'] for s in val_samples)),
        'test_patients': len(set(s['patient_id'] for s in test_samples)),
    }

    print(f"✓ Train: {stats['train_samples']} samples ({stats['train_patients']} pacientes)")
    print(f"✓ Val:   {stats['val_samples']} samples ({stats['val_patients']} pacientes)")
    print(f"✓ Test:  {stats['test_samples']} samples ({stats['test_patients']} pacientes)")

    # Criar datasets
    train_dataset = PulmoSegDataset(train_samples, transform=train_transform, is_train=True)
    val_dataset = PulmoSegDataset(val_samples, transform=val_transform, is_train=False)
    test_dataset = PulmoSegDataset(test_samples, transform=val_transform, is_train=False)

    # Criar DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Evitar batch incompleto no final
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, stats


if __name__ == '__main__':
    # Teste do dataset
    print("Testando dataset...")

    # Criar loaders sem augmentation para teste
    train_loader, val_loader, test_loader, stats = get_dataloaders()

    # Testar uma amostra
    image, mask, path = next(iter(train_loader))
    print(f"\nShape da imagem: {image.shape}")  # [batch, 3, H, W]
    print(f"Shape da máscara: {mask.shape}")    # [batch, 1, H, W]
    print(f"Range da imagem: [{image.min():.3f}, {image.max():.3f}]")
    print(f"Range da máscara: [{mask.min():.3f}, {mask.max():.3f}]")
    print(f"\nPath: {path[0]}")
