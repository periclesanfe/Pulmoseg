"""
PulmoSeg Processor - Segmentação Clássica de Nódulos Pulmonares
Implementa pipeline: Grayscale -> CLAHE -> Otsu -> Morfologia
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional


class PulmoSegProcessor:
    """
    Processador de segmentação de nódulos pulmonares usando técnicas clássicas.
    Pipeline: Conversão Grayscale -> CLAHE -> Limiarização de Otsu -> Morfologia
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        """
        Inicializa o processador com parâmetros de CLAHE.

        Args:
            clip_limit: Limite de contraste para CLAHE (padrão: 2.0)
            tile_grid_size: Tamanho da grade de tiles para CLAHE (padrão: (8, 8))
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def process_slice(self, image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Processa uma única fatia de TC aplicando o pipeline completo.

        Etapas:
        1. Carregamento e conversão para escala de cinza
        2. Realce com CLAHE
        3. Segmentação com Otsu
        4. Operação morfológica (abertura) para remover ruído
        5. Extração de atributos (área)

        Args:
            image_path: Caminho para a imagem da fatia

        Returns:
            Tupla contendo:
            - gray: Imagem em escala de cinza
            - clahe_result: Imagem após CLAHE
            - otsu_mask: Máscara binária após Otsu + morfologia
            - area: Área do nódulo segmentado em pixels
        """
        # Etapa 1: Aquisição e Conversão para Grayscale
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Não foi possível carregar a imagem: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Etapa 2: Realce com CLAHE
        clahe_result = self.clahe.apply(gray)

        # Etapa 3: Segmentação com Otsu
        _, otsu_mask = cv2.threshold(clahe_result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Etapa 4: Morfologia - Abertura para remover ruído pequeno
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        otsu_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Etapa 5: Extração de Atributos - Área
        area = cv2.countNonZero(otsu_mask)

        return gray, clahe_result, otsu_mask, area


def create_consensus_mask(mask_paths: List[str], majority_threshold: int = 2) -> Optional[np.ndarray]:
    """
    Cria uma máscara de consenso a partir de múltiplas anotações de radiologistas.

    Args:
        mask_paths: Lista de caminhos para as máscaras (mask-0 até mask-3)
        majority_threshold: Número mínimo de radiologistas que devem concordar (padrão: 2)

    Returns:
        Máscara de consenso binária (0 ou 255) ou None se nenhuma máscara válida for encontrada
    """
    masks = []

    for mask_path in mask_paths:
        if Path(mask_path).exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None and cv2.countNonZero(mask) > 0:
                # Normalizar para binário (0 ou 1)
                _, binary_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
                masks.append(binary_mask)

    if len(masks) == 0:
        return None

    # Somar todas as máscaras e aplicar threshold de maioria
    consensus = np.sum(masks, axis=0)
    consensus_mask = (consensus >= majority_threshold).astype(np.uint8) * 255

    return consensus_mask


def calculate_dice(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calcula o coeficiente de Dice (F1-Score) entre predição e ground truth.

    Dice = 2 * |A ∩ B| / (|A| + |B|)

    Args:
        predicted: Máscara predita (binária)
        ground_truth: Máscara ground truth (binária)

    Returns:
        Coeficiente de Dice (0.0 a 1.0)
    """
    # Normalizar para binário
    pred_bin = (predicted > 127).astype(np.uint8)
    gt_bin = (ground_truth > 127).astype(np.uint8)

    # Calcular interseção e soma
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    sum_areas = pred_bin.sum() + gt_bin.sum()

    if sum_areas == 0:
        return 0.0 if intersection == 0 else 1.0

    dice = (2.0 * intersection) / sum_areas
    return dice


def calculate_iou(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calcula o Intersection over Union (IoU / Jaccard Index).

    IoU = |A ∩ B| / |A ∪ B|

    Args:
        predicted: Máscara predita (binária)
        ground_truth: Máscara ground truth (binária)

    Returns:
        IoU score (0.0 a 1.0)
    """
    # Normalizar para binário
    pred_bin = (predicted > 127).astype(np.uint8)
    gt_bin = (ground_truth > 127).astype(np.uint8)

    # Calcular interseção e união
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()

    if union == 0:
        return 0.0 if intersection == 0 else 1.0

    iou = intersection / union
    return iou
