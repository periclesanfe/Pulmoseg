"""
PulmoSeg Analyze Results - Análise Comparativa de Configurações
Gera gráficos e estatísticas para comparar os resultados dos testes
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json


def load_results():
    """Carrega o resumo comparativo dos testes"""
    summary_path = Path('results') / 'comparison_summary.csv'
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Arquivo de resumo não encontrado: {summary_path}\n"
            "Execute primeiro: python3 test_optimizer.py"
        )

    df = pd.read_csv(summary_path)
    return df


def plot_dice_ranking(df: pd.DataFrame, top_n: int = 20, save_path: Path = None):
    """Plota ranking de configurações por Dice Score"""
    plt.figure(figsize=(14, 8))

    # Selecionar top N
    df_top = df.head(top_n)

    # Criar barplot
    bars = plt.barh(range(len(df_top)), df_top['dice_mean'], xerr=df_top['dice_std'],
                    capsize=3, alpha=0.7, color='steelblue')

    # Colorir baseline em destaque
    for i, config_name in enumerate(df_top['config_name']):
        if config_name == 'baseline':
            bars[i].set_color('red')
            bars[i].set_alpha(0.8)

    plt.yticks(range(len(df_top)), df_top['config_name'], fontsize=9)
    plt.xlabel('Dice Score (média ± std)', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Configurações - Ranking por Dice Score', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {save_path}")

    plt.close()


def plot_dice_iou_scatter(df: pd.DataFrame, save_path: Path = None):
    """Scatter plot: Dice vs IoU"""
    plt.figure(figsize=(10, 8))

    # Scatter plot
    plt.scatter(df['dice_mean'], df['iou_mean'], alpha=0.6, s=100, c='steelblue', edgecolors='black', linewidth=0.5)

    # Destacar baseline
    baseline = df[df['config_name'] == 'baseline']
    if len(baseline) > 0:
        plt.scatter(baseline['dice_mean'], baseline['iou_mean'],
                   s=150, c='red', marker='*', edgecolors='darkred', linewidth=2, label='Baseline', zorder=5)

    # Destacar top 3
    top3 = df.head(3)
    plt.scatter(top3['dice_mean'], top3['iou_mean'],
               s=120, c='gold', marker='D', edgecolors='orange', linewidth=1.5, label='Top 3', zorder=4)

    # Anotar top 3
    for idx, row in top3.iterrows():
        plt.annotate(row['config_name'], (row['dice_mean'], row['iou_mean']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

    plt.xlabel('Dice Score (média)', fontsize=12, fontweight='bold')
    plt.ylabel('IoU (média)', fontsize=12, fontweight='bold')
    plt.title('Dice Score vs IoU - Comparação de Configurações', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {save_path}")

    plt.close()


def plot_parameter_analysis(df: pd.DataFrame, save_dir: Path):
    """Análise de impacto de cada parâmetro"""
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. CLAHE clipLimit
    if 'clip_limit' in df.columns:
        plt.figure(figsize=(10, 6))
        clip_groups = df.groupby('clip_limit')['dice_mean'].agg(['mean', 'std', 'count'])
        clip_groups = clip_groups[clip_groups['count'] >= 3]  # Apenas grupos com ≥3 amostras

        if len(clip_groups) > 0:
            plt.bar(clip_groups.index.astype(str), clip_groups['mean'],
                   yerr=clip_groups['std'], capsize=5, alpha=0.7, color='coral')
            plt.xlabel('CLAHE clipLimit', fontsize=12, fontweight='bold')
            plt.ylabel('Dice Score (média)', fontsize=12, fontweight='bold')
            plt.title('Impacto do CLAHE clipLimit no Desempenho', fontsize=14, fontweight='bold')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            save_path = save_dir / 'clahe_clip_limit_impact.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Gráfico salvo: {save_path}")
            plt.close()

    # 2. Threshold method
    if 'threshold_method' in df.columns:
        plt.figure(figsize=(10, 6))
        thresh_groups = df.groupby('threshold_method')['dice_mean'].agg(['mean', 'std', 'count'])
        thresh_groups = thresh_groups[thresh_groups['count'] >= 3]

        if len(thresh_groups) > 0:
            plt.bar(thresh_groups.index, thresh_groups['mean'],
                   yerr=thresh_groups['std'], capsize=5, alpha=0.7, color='lightgreen')
            plt.xlabel('Método de Threshold', fontsize=12, fontweight='bold')
            plt.ylabel('Dice Score (média)', fontsize=12, fontweight='bold')
            plt.title('Impacto do Método de Threshold no Desempenho', fontsize=14, fontweight='bold')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            save_path = save_dir / 'threshold_method_impact.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Gráfico salvo: {save_path}")
            plt.close()

    # 3. Morfologia
    if 'morph_operation' in df.columns:
        plt.figure(figsize=(10, 6))
        morph_groups = df.groupby('morph_operation')['dice_mean'].agg(['mean', 'std', 'count'])
        morph_groups = morph_groups[morph_groups['count'] >= 3]

        if len(morph_groups) > 0:
            plt.bar(morph_groups.index, morph_groups['mean'],
                   yerr=morph_groups['std'], capsize=5, alpha=0.7, color='skyblue')
            plt.xlabel('Operação Morfológica', fontsize=12, fontweight='bold')
            plt.ylabel('Dice Score (média)', fontsize=12, fontweight='bold')
            plt.title('Impacto da Operação Morfológica no Desempenho', fontsize=14, fontweight='bold')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            save_path = save_dir / 'morph_operation_impact.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Gráfico salvo: {save_path}")
            plt.close()

    # 4. Pré-processamento
    if 'preprocessing' in df.columns:
        plt.figure(figsize=(10, 6))
        # Substituir NaN por 'none' para visualização
        df_prep = df.copy()
        df_prep['preprocessing'] = df_prep['preprocessing'].fillna('none')
        prep_groups = df_prep.groupby('preprocessing')['dice_mean'].agg(['mean', 'std', 'count'])
        prep_groups = prep_groups[prep_groups['count'] >= 3]

        if len(prep_groups) > 0:
            plt.bar(prep_groups.index, prep_groups['mean'],
                   yerr=prep_groups['std'], capsize=5, alpha=0.7, color='plum')
            plt.xlabel('Pré-processamento', fontsize=12, fontweight='bold')
            plt.ylabel('Dice Score (média)', fontsize=12, fontweight='bold')
            plt.title('Impacto do Pré-processamento no Desempenho', fontsize=14, fontweight='bold')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            save_path = save_dir / 'preprocessing_impact.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Gráfico salvo: {save_path}")
            plt.close()


def generate_statistics_report(df: pd.DataFrame) -> str:
    """Gera relatório estatístico em texto"""
    report = []
    report.append("=" * 70)
    report.append("RELATÓRIO ESTATÍSTICO - ANÁLISE DE CONFIGURAÇÕES")
    report.append("=" * 70)
    report.append("")

    # Estatísticas gerais
    report.append("### ESTATÍSTICAS GERAIS")
    report.append(f"Total de configurações testadas: {len(df)}")
    report.append(f"Total de slices processadas: {df['n_slices'].sum()}")
    report.append("")

    report.append("### DICE SCORE")
    report.append(f"  Média geral: {df['dice_mean'].mean():.4f}")
    report.append(f"  Mediana geral: {df['dice_median'].median():.4f}")
    report.append(f"  Melhor config: {df.iloc[0]['dice_mean']:.4f} ({df.iloc[0]['config_name']})")
    report.append(f"  Pior config: {df.iloc[-1]['dice_mean']:.4f} ({df.iloc[-1]['config_name']})")
    report.append(f"  Melhoria vs baseline: {((df.iloc[0]['dice_mean'] / df[df['config_name']=='baseline']['dice_mean'].values[0] - 1) * 100):.2f}%")
    report.append("")

    report.append("### IoU")
    report.append(f"  Média geral: {df['iou_mean'].mean():.4f}")
    report.append(f"  Mediana geral: {df['iou_median'].median():.4f}")
    report.append(f"  Melhor config: {df.iloc[0]['iou_mean']:.4f} ({df.iloc[0]['config_name']})")
    report.append("")

    # Top 10
    report.append("### TOP 10 CONFIGURAÇÕES")
    report.append("")
    for i, row in df.head(10).iterrows():
        report.append(f"{i+1}. {row['config_name']}")
        report.append(f"   Dice: {row['dice_mean']:.4f} ± {row['dice_std']:.4f}")
        report.append(f"   IoU:  {row['iou_mean']:.4f} ± {row['iou_std']:.4f}")
        report.append(f"   Slices: {row['n_slices']}")
        report.append("")

    report.append("=" * 70)

    return "\n".join(report)


def main():
    print("\n" + "="*70)
    print("PulmoSeg - Análise de Resultados")
    print("="*70 + "\n")

    # Carregar resultados
    print("Carregando resultados...")
    df = load_results()
    print(f"✓ {len(df)} configurações carregadas\n")

    # Criar diretório de análises
    analysis_dir = Path('results') / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Gerar gráficos
    print("Gerando gráficos...")

    plot_dice_ranking(df, top_n=20, save_path=analysis_dir / 'dice_ranking_top20.png')
    plot_dice_iou_scatter(df, save_path=analysis_dir / 'dice_vs_iou_scatter.png')
    plot_parameter_analysis(df, save_dir=analysis_dir / 'parameter_analysis')

    # Gerar relatório estatístico
    print("\nGerando relatório estatístico...")
    report_text = generate_statistics_report(df)
    report_path = analysis_dir / 'statistics_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"✓ Relatório salvo: {report_path}\n")

    # Exibir relatório
    print(report_text)

    print(f"\n{'='*70}")
    print(f"Análise concluída! Arquivos salvos em: {analysis_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
