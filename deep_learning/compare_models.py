"""
PulmoSeg - Comparação entre Modelos DL e Métodos Clássicos
"""

import pandas as pd
from pathlib import Path

# Comparar resultados de DL vs clássico
classical_results = Path('results/comparison_summary.csv')
dl_results = Path('results/dl_models')

if classical_results.exists():
    df_classical = pd.read_csv(classical_results)
    best_classical = df_classical.iloc[0]

    print(f"\n{'='*60}")
    print("Comparação: Deep Learning vs Técnicas Clássicas")
    print(f"{'='*60}\n")

    print("Melhor Método Clássico:")
    print(f"  Configuração: {best_classical['config_name']}")
    print(f"  Dice Score: {best_classical['dice_mean']:.4f}")
    print(f"  IoU: {best_classical['iou_mean']:.4f}\n")

    print("Modelos de Deep Learning:")
    print("  (Execute train.py primeiro para gerar resultados)\n")

    print(f"{'='*60}\n")
