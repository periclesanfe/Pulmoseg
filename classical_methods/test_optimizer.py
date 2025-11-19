"""
PulmoSeg Test Optimizer - Grid Search para Otimização de Parâmetros
Testa múltiplas combinações de parâmetros para encontrar a melhor configuração
"""

import argparse
import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
import random

from pulmoseg_processor import create_consensus_mask, calculate_dice, calculate_iou


@dataclass
class PipelineConfig:
    """Configuração de parâmetros do pipeline"""
    name: str
    # CLAHE
    clip_limit: float = 2.0
    tile_grid_size: Tuple[int, int] = (8, 8)
    # Pré-processamento
    preprocessing: Optional[str] = None  # None, 'gaussian', 'median', 'bilateral'
    preprocessing_kernel: int = 5
    # Threshold
    threshold_method: str = 'otsu'  # 'otsu', 'adaptive', 'binary'
    binary_threshold: int = 127
    # Morfologia
    morph_operation: str = 'opening'  # 'opening', 'closing', 'gradient', 'none'
    morph_kernel_size: int = 3
    morph_iterations: int = 1


class OptimizedProcessor:
    """Processador com parâmetros configuráveis"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.clahe = cv2.createCLAHE(
            clipLimit=config.clip_limit,
            tileGridSize=config.tile_grid_size
        )

    def apply_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Aplica filtro de pré-processamento se especificado"""
        if self.config.preprocessing is None:
            return image

        k = self.config.preprocessing_kernel

        if self.config.preprocessing == 'gaussian':
            return cv2.GaussianBlur(image, (k, k), 0)
        elif self.config.preprocessing == 'median':
            return cv2.medianBlur(image, k)
        elif self.config.preprocessing == 'bilateral':
            return cv2.bilateralFilter(image, k, 75, 75)

        return image

    def apply_threshold(self, image: np.ndarray) -> np.ndarray:
        """Aplica método de threshold configurado"""
        if self.config.threshold_method == 'otsu':
            _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif self.config.threshold_method == 'adaptive':
            mask = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        elif self.config.threshold_method == 'binary':
            _, mask = cv2.threshold(image, self.config.binary_threshold, 255, cv2.THRESH_BINARY)
        else:
            raise ValueError(f"Método de threshold desconhecido: {self.config.threshold_method}")

        return mask

    def apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """Aplica operação morfológica configurada"""
        if self.config.morph_operation == 'none':
            return mask

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.morph_kernel_size, self.config.morph_kernel_size)
        )

        if self.config.morph_operation == 'opening':
            return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=self.config.morph_iterations)
        elif self.config.morph_operation == 'closing':
            return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self.config.morph_iterations)
        elif self.config.morph_operation == 'gradient':
            return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)

        return mask

    def process_slice(self, image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Processa uma fatia com a configuração atual"""
        # 1. Carregar e converter
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Não foi possível carregar: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 2. Pré-processamento (opcional)
        preprocessed = self.apply_preprocessing(gray)

        # 3. CLAHE
        clahe_result = self.clahe.apply(preprocessed)

        # 4. Threshold
        mask = self.apply_threshold(clahe_result)

        # 5. Morfologia
        final_mask = self.apply_morphology(mask)

        # 6. Área
        area = cv2.countNonZero(final_mask)

        return gray, clahe_result, final_mask, area


def generate_test_configurations() -> List[PipelineConfig]:
    """
    Gera configurações de teste de forma inteligente.
    Ao invés de grid search completo (432 configs), usa estratégia incremental.
    """
    configs = []

    # BASELINE: Configuração original do projeto
    configs.append(PipelineConfig(
        name="baseline",
        clip_limit=2.0,
        tile_grid_size=(8, 8),
        preprocessing=None,
        threshold_method='otsu',
        morph_operation='opening',
        morph_kernel_size=3
    ))

    # FASE 1: Otimizar CLAHE (12 configurações)
    for clip in [1.0, 2.0, 3.0, 4.0]:
        for tile_size in [(4, 4), (8, 8), (16, 16)]:
            configs.append(PipelineConfig(
                name=f"clahe_clip{clip}_tile{tile_size[0]}x{tile_size[1]}",
                clip_limit=clip,
                tile_grid_size=tile_size,
                preprocessing=None,
                threshold_method='otsu',
                morph_operation='opening',
                morph_kernel_size=3
            ))

    # FASE 2: Testar pré-processamento (4 configurações)
    for preproc in ['gaussian', 'median', 'bilateral']:
        configs.append(PipelineConfig(
            name=f"preproc_{preproc}",
            clip_limit=2.0,
            tile_grid_size=(8, 8),
            preprocessing=preproc,
            preprocessing_kernel=5,
            threshold_method='otsu',
            morph_operation='opening',
            morph_kernel_size=3
        ))

    # FASE 3: Testar métodos de threshold (3 configurações)
    for thresh_method in ['otsu', 'adaptive', 'binary']:
        configs.append(PipelineConfig(
            name=f"threshold_{thresh_method}",
            clip_limit=2.0,
            tile_grid_size=(8, 8),
            preprocessing=None,
            threshold_method=thresh_method,
            binary_threshold=127,
            morph_operation='opening',
            morph_kernel_size=3
        ))

    # FASE 4: Testar operações morfológicas (9 configurações)
    for morph_op in ['opening', 'closing', 'gradient']:
        for kernel_size in [3, 5, 7]:
            configs.append(PipelineConfig(
                name=f"morph_{morph_op}_k{kernel_size}",
                clip_limit=2.0,
                tile_grid_size=(8, 8),
                preprocessing=None,
                threshold_method='otsu',
                morph_operation=morph_op,
                morph_kernel_size=kernel_size,
                morph_iterations=1
            ))

    # FASE 5: Combinações promissoras (baseadas em conhecimento prévio)
    # Gaussian + CLAHE forte + Adaptive
    configs.append(PipelineConfig(
        name="combo_gaussian_adaptive",
        clip_limit=3.0,
        tile_grid_size=(8, 8),
        preprocessing='gaussian',
        preprocessing_kernel=5,
        threshold_method='adaptive',
        morph_operation='closing',
        morph_kernel_size=5
    ))

    # Bilateral + CLAHE moderado + Otsu
    configs.append(PipelineConfig(
        name="combo_bilateral_otsu",
        clip_limit=2.0,
        tile_grid_size=(4, 4),
        preprocessing='bilateral',
        preprocessing_kernel=5,
        threshold_method='otsu',
        morph_operation='opening',
        morph_kernel_size=3
    ))

    # Median + CLAHE forte + Closing
    configs.append(PipelineConfig(
        name="combo_median_closing",
        clip_limit=4.0,
        tile_grid_size=(16, 16),
        preprocessing='median',
        preprocessing_kernel=5,
        threshold_method='otsu',
        morph_operation='closing',
        morph_kernel_size=7
    ))

    return configs


def sample_patients(dataset_path: Path, n_patients: int, seed: int = 42) -> List[Path]:
    """Amostra pacientes de forma reprodutível com seed fixa"""
    random.seed(seed)
    np.random.seed(seed)

    all_patients = sorted([d for d in dataset_path.iterdir()
                          if d.is_dir() and d.name.startswith('LIDC-IDRI-')])

    if n_patients >= len(all_patients):
        return all_patients

    return random.sample(all_patients, n_patients)


def has_valid_masks(nodule_dir: Path) -> bool:
    """Verifica se nódulo possui máscaras válidas"""
    for i in range(4):
        mask_dir = nodule_dir / f'mask-{i}'
        if mask_dir.exists():
            for mask_file in mask_dir.glob('*.png'):
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if mask is not None and cv2.countNonZero(mask) > 0:
                    return True
    return False


def save_visualization(original_path: str, clahe_img: np.ndarray, pred_mask: np.ndarray,
                       gt_mask: np.ndarray, patient_id: str, nodule_id: str, slice_id: str,
                       dice: float, iou: float, save_dir: Path, category: str):
    """Salva visualização de caso extremo"""
    original_img = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f'{category.upper()} | {patient_id}/{nodule_id}/{slice_id}\nDice: {dice:.4f} | IoU: {iou:.4f}',
                 fontsize=12, fontweight='bold')

    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(clahe_img, cmap='gray')
    axes[1].set_title('CLAHE')
    axes[1].axis('off')

    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title('Predição')
    axes[2].axis('off')

    axes[3].imshow(gt_mask, cmap='gray')
    axes[3].set_title('Ground Truth')
    axes[3].axis('off')

    plt.tight_layout()
    save_path = save_dir / f"{patient_id}_{nodule_id}_{slice_id}.png"
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def test_configuration(config: PipelineConfig, patient_dirs: List[Path],
                       save_extremes: bool = True) -> pd.DataFrame:
    """
    Testa uma configuração específica no conjunto de pacientes
    """
    processor = OptimizedProcessor(config)
    results = []

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        nodule_dirs = sorted([d for d in patient_dir.iterdir()
                             if d.is_dir() and d.name.startswith('nodule-')])

        for nodule_dir in nodule_dirs:
            nodule_id = nodule_dir.name

            if not has_valid_masks(nodule_dir):
                continue

            images_dir = nodule_dir / 'images'
            if not images_dir.exists():
                continue

            slice_files = sorted(images_dir.glob('*.png'))

            for slice_file in slice_files:
                slice_id = slice_file.stem

                try:
                    # Processar slice
                    gray, clahe_result, pred_mask, area = processor.process_slice(slice_file)

                    # Criar máscara de consenso
                    mask_paths = [nodule_dir / f'mask-{i}' / slice_file.name for i in range(4)]
                    gt_mask = create_consensus_mask(mask_paths, majority_threshold=2)

                    if gt_mask is None:
                        continue

                    # Calcular métricas
                    dice = calculate_dice(pred_mask, gt_mask)
                    iou = calculate_iou(pred_mask, gt_mask)

                    # Armazenar resultados
                    results.append({
                        'config_name': config.name,
                        'PacienteID': patient_id,
                        'NoduloID': nodule_id,
                        'FatiaID': slice_id,
                        'Area_Detectada_Pixels': area,
                        'Dice_Score': dice,
                        'IoU': iou,
                        'original_path': str(slice_file),
                        'clahe_img': clahe_result,
                        'pred_mask': pred_mask,
                        'gt_mask': gt_mask
                    })

                except Exception as e:
                    print(f"\nErro em {patient_id}/{nodule_id}/{slice_id}: {e}")
                    continue

    df = pd.DataFrame(results)

    # Salvar casos extremos (top 5 e bottom 5 por Dice)
    if save_extremes and len(df) > 0:
        extremes_dir = Path('results') / 'visualizations' / config.name
        extremes_dir.mkdir(parents=True, exist_ok=True)

        # Top 5
        top5 = df.nlargest(5, 'Dice_Score')
        best_dir = extremes_dir / 'best_cases'
        best_dir.mkdir(exist_ok=True)

        for idx, row in top5.iterrows():
            save_visualization(
                row['original_path'], row['clahe_img'], row['pred_mask'], row['gt_mask'],
                row['PacienteID'], row['NoduloID'], row['FatiaID'],
                row['Dice_Score'], row['IoU'], best_dir, 'BEST'
            )

        # Bottom 5
        bottom5 = df.nsmallest(5, 'Dice_Score')
        worst_dir = extremes_dir / 'worst_cases'
        worst_dir.mkdir(exist_ok=True)

        for idx, row in bottom5.iterrows():
            save_visualization(
                row['original_path'], row['clahe_img'], row['pred_mask'], row['gt_mask'],
                row['PacienteID'], row['NoduloID'], row['FatiaID'],
                row['Dice_Score'], row['IoU'], worst_dir, 'WORST'
            )

    # Remover colunas de imagens para salvar no CSV
    df_clean = df.drop(columns=['original_path', 'clahe_img', 'pred_mask', 'gt_mask'])

    return df_clean


def main():
    parser = argparse.ArgumentParser(description='PulmoSeg Test Optimizer - Grid Search')
    parser.add_argument('--dataset', type=str, default='LIDC-IDRI-slices',
                       help='Caminho do dataset')
    parser.add_argument('--patients', type=int, default=50,
                       help='Número de pacientes para amostra')
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed para reprodutibilidade')
    parser.add_argument('--save-extremes', action='store_true', default=True,
                       help='Salvar visualizações de casos extremos')

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise ValueError(f"Dataset não encontrado: {dataset_path}")

    print(f"\n{'='*70}")
    print(f"PulmoSeg Test Optimizer - Grid Search de Parâmetros")
    print(f"{'='*70}")
    print(f"Dataset: {dataset_path}")
    print(f"Pacientes: {args.patients}")
    print(f"Seed: {args.seed}")
    print(f"{'='*70}\n")

    # Amostrar pacientes
    patient_dirs = sample_patients(dataset_path, args.patients, args.seed)
    print(f"✓ {len(patient_dirs)} pacientes selecionados (seed={args.seed})\n")

    # Gerar configurações
    configs = generate_test_configurations()
    print(f"✓ {len(configs)} configurações geradas\n")

    # Criar diretório de resultados
    results_dir = Path('results') / 'test_runs'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Testar cada configuração
    all_summaries = []

    for i, config in enumerate(tqdm(configs, desc="Testando configurações", unit="config")):
        print(f"\n[{i+1}/{len(configs)}] Testando: {config.name}")

        # Testar configuração
        df_results = test_configuration(config, patient_dirs, args.save_extremes)

        if len(df_results) > 0:
            # Salvar CSV individual
            csv_path = results_dir / f"{config.name}.csv"
            df_results.to_csv(csv_path, index=False)

            # Calcular estatísticas
            summary = {
                'config_name': config.name,
                'n_slices': len(df_results),
                'dice_mean': df_results['Dice_Score'].mean(),
                'dice_std': df_results['Dice_Score'].std(),
                'dice_median': df_results['Dice_Score'].median(),
                'iou_mean': df_results['IoU'].mean(),
                'iou_std': df_results['IoU'].std(),
                'iou_median': df_results['IoU'].median(),
                **asdict(config)
            }
            all_summaries.append(summary)

            print(f"  ✓ Dice: {summary['dice_mean']:.4f} ± {summary['dice_std']:.4f}")
            print(f"  ✓ IoU:  {summary['iou_mean']:.4f} ± {summary['iou_std']:.4f}")
            print(f"  ✓ Slices: {summary['n_slices']}")

    # Salvar resumo comparativo
    df_summary = pd.DataFrame(all_summaries)
    df_summary = df_summary.sort_values('dice_mean', ascending=False)
    summary_path = Path('results') / 'comparison_summary.csv'
    df_summary.to_csv(summary_path, index=False)

    # Salvar melhor configuração
    if len(df_summary) > 0:
        best_config = df_summary.iloc[0]
        best_config_dict = best_config.to_dict()
        best_config_path = Path('results') / 'best_config.json'
        with open(best_config_path, 'w') as f:
            json.dump(best_config_dict, f, indent=2)

        print(f"\n{'='*70}")
        print(f"Testes Concluídos!")
        print(f"{'='*70}")
        print(f"Configurações testadas: {len(configs)}")
        print(f"Slices processadas: {df_summary['n_slices'].sum()}")
        print(f"\nMELHOR CONFIGURAÇÃO:")
        print(f"  Nome: {best_config['config_name']}")
        print(f"  Dice: {best_config['dice_mean']:.4f} ± {best_config['dice_std']:.4f}")
        print(f"  IoU:  {best_config['iou_mean']:.4f} ± {best_config['iou_std']:.4f}")
        print(f"\nResultados salvos em:")
        print(f"  - {summary_path}")
        print(f"  - {best_config_path}")
        print(f"  - {results_dir}/")
        print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
