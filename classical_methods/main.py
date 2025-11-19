"""
PulmoSeg - Segmentação Automática de Nódulos Pulmonares
Script principal para processar dataset LIDC-IDRI
"""

import argparse
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from pulmoseg_processor import (
    PulmoSegProcessor,
    create_consensus_mask,
    calculate_dice,
    calculate_iou
)


def find_patient_directories(dataset_path: Path, limit: int = None) -> list:
    """
    Encontra diretórios de pacientes no dataset.

    Args:
        dataset_path: Caminho para LIDC-IDRI-slices/
        limit: Limite de pacientes a processar (None = todos)

    Returns:
        Lista de caminhos para diretórios de pacientes
    """
    patient_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith('LIDC-IDRI-')])

    if limit:
        patient_dirs = patient_dirs[:limit]

    return patient_dirs


def has_valid_masks(nodule_dir: Path) -> bool:
    """
    Verifica se um nódulo possui pelo menos uma máscara válida.

    Args:
        nodule_dir: Diretório do nódulo

    Returns:
        True se há pelo menos uma máscara não-vazia
    """
    for i in range(4):
        mask_dir = nodule_dir / f'mask-{i}'
        if mask_dir.exists():
            # Verificar se há pelo menos uma imagem não-preta
            for mask_file in mask_dir.glob('*.png'):
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if mask is not None and cv2.countNonZero(mask) > 0:
                    return True
    return False


def visualize_results(original: str, clahe_img, otsu_mask, consensus_mask,
                      patient_id: str, nodule_id: str, slice_id: str,
                      dice: float, iou: float):
    """
    Visualiza resultados lado a lado: Original | CLAHE | Otsu | Ground Truth.

    Args:
        original: Caminho da imagem original
        clahe_img: Imagem após CLAHE
        otsu_mask: Máscara gerada por Otsu
        consensus_mask: Máscara de consenso (ground truth)
        patient_id: ID do paciente
        nodule_id: ID do nódulo
        slice_id: ID da fatia
        dice: Coeficiente de Dice
        iou: IoU score
    """
    # Carregar imagem original para visualização
    original_img = cv2.imread(str(original), cv2.IMREAD_GRAYSCALE)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f'{patient_id} | {nodule_id} | {slice_id} | Dice: {dice:.4f} | IoU: {iou:.4f}',
                 fontsize=12, fontweight='bold')

    # Imagem Original
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    # CLAHE
    axes[1].imshow(clahe_img, cmap='gray')
    axes[1].set_title('CLAHE\n(clipLimit=2.0)')
    axes[1].axis('off')

    # Máscara Otsu (Predição)
    axes[2].imshow(otsu_mask, cmap='gray')
    axes[2].set_title('Otsu (Predição)\nTécnica Clássica')
    axes[2].axis('off')

    # Máscara Ground Truth (Consenso)
    axes[3].imshow(consensus_mask, cmap='gray')
    axes[3].set_title('Ground Truth\n(Consenso ≥2 Radiologistas)')
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()


def process_dataset(dataset_path: str, limit: int = None, visualize: bool = True):
    """
    Processa o dataset LIDC-IDRI aplicando o pipeline PulmoSeg.

    Args:
        dataset_path: Caminho para a pasta LIDC-IDRI-slices/
        limit: Número máximo de pacientes a processar
        visualize: Se True, exibe visualizações matplotlib
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise ValueError(f"Dataset não encontrado: {dataset_path}")

    # Inicializar processador
    processor = PulmoSegProcessor(clip_limit=2.0, tile_grid_size=(8, 8))

    # Coletar pacientes
    patient_dirs = find_patient_directories(dataset_path, limit)
    print(f"\n{'='*60}")
    print(f"PulmoSeg - Segmentação Automática de Nódulos Pulmonares")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_path}")
    print(f"Pacientes a processar: {len(patient_dirs)}")
    print(f"{'='*60}\n")

    # Armazenar resultados
    results = []

    # Processar cada paciente
    for patient_dir in tqdm(patient_dirs, desc="Processando pacientes", unit="paciente"):
        patient_id = patient_dir.name

        # Processar cada nódulo do paciente
        nodule_dirs = sorted([d for d in patient_dir.iterdir() if d.is_dir() and d.name.startswith('nodule-')])

        for nodule_dir in nodule_dirs:
            nodule_id = nodule_dir.name

            # Pular nódulos sem máscaras (conforme especificação do usuário)
            if not has_valid_masks(nodule_dir):
                continue

            # Processar cada slice do nódulo
            images_dir = nodule_dir / 'images'
            if not images_dir.exists():
                continue

            slice_files = sorted(images_dir.glob('*.png'))

            for slice_file in slice_files:
                slice_id = slice_file.stem  # Ex: 'slice-0'

                try:
                    # Etapa 1: Processar slice com pipeline CLAHE + Otsu
                    gray, clahe_result, otsu_mask, area = processor.process_slice(slice_file)

                    # Etapa 2: Criar máscara de consenso (Ground Truth)
                    mask_paths = [nodule_dir / f'mask-{i}' / slice_file.name for i in range(4)]
                    consensus_mask = create_consensus_mask(mask_paths, majority_threshold=2)

                    if consensus_mask is None:
                        continue  # Pular se não há consenso válido

                    # Etapa 3: Calcular métricas
                    dice = calculate_dice(otsu_mask, consensus_mask)
                    iou = calculate_iou(otsu_mask, consensus_mask)

                    # Etapa 4: Visualizar (se solicitado)
                    if visualize:
                        visualize_results(
                            slice_file, clahe_result, otsu_mask, consensus_mask,
                            patient_id, nodule_id, slice_id, dice, iou
                        )

                    # Etapa 5: Armazenar resultados
                    results.append({
                        'PacienteID': patient_id,
                        'NoduloID': nodule_id,
                        'FatiaID': slice_id,
                        'Area_Detectada_Pixels': area,
                        'Dice_Score': dice,
                        'IoU': iou
                    })

                except Exception as e:
                    print(f"\nErro ao processar {patient_id}/{nodule_id}/{slice_id}: {e}")
                    continue

    # Salvar resultados em CSV
    if results:
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)

        df = pd.DataFrame(results)
        csv_path = results_dir / 'metrics.csv'
        df.to_csv(csv_path, index=False)

        print(f"\n{'='*60}")
        print(f"Processamento concluído!")
        print(f"{'='*60}")
        print(f"Total de slices processadas: {len(results)}")
        print(f"Dice Score médio: {df['Dice_Score'].mean():.4f} ± {df['Dice_Score'].std():.4f}")
        print(f"IoU médio: {df['IoU'].mean():.4f} ± {df['IoU'].std():.4f}")
        print(f"\nResultados salvos em: {csv_path}")
        print(f"{'='*60}\n")
    else:
        print("\nNenhum resultado para processar. Verifique o dataset.")


def main():
    """Função principal com argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description='PulmoSeg - Segmentação Automática de Nódulos Pulmonares',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python main.py                    # Processar dataset completo
  python main.py --limit 10         # Processar apenas 10 primeiros pacientes
  python main.py --no-visualize     # Processar sem exibir visualizações
  python main.py --limit 5 --no-visualize  # Modo rápido para testes

Dataset esperado: LIDC-IDRI-slices/
        """
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='LIDC-IDRI-slices',
        help='Caminho para o dataset (padrão: LIDC-IDRI-slices/)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Número máximo de pacientes a processar (padrão: todos)'
    )

    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Desabilita visualizações matplotlib (modo rápido)'
    )

    args = parser.parse_args()

    # Executar processamento
    process_dataset(
        dataset_path=args.dataset,
        limit=args.limit,
        visualize=not args.no_visualize
    )


if __name__ == '__main__':
    main()
