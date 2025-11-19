"""
PulmoSeg Report Generator - Geração de Relatório Markdown
Cria relatório completo com resultados, análises e recomendações
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime


def load_data():
    """Carrega dados necessários para o relatório"""
    summary_path = Path('results') / 'comparison_summary.csv'
    best_config_path = Path('results') / 'best_config.json'

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Arquivo de resumo não encontrado: {summary_path}\n"
            "Execute primeiro: python3 test_optimizer.py"
        )

    df_summary = pd.read_csv(summary_path)

    best_config = None
    if best_config_path.exists():
        with open(best_config_path, 'r') as f:
            best_config = json.load(f)

    return df_summary, best_config


def generate_markdown_report(df_summary: pd.DataFrame, best_config: dict) -> str:
    """Gera relatório em formato Markdown"""

    # Header
    report = []
    report.append("# PulmoSeg - Relatório de Otimização de Parâmetros")
    report.append("")
    report.append(f"**Data de Geração**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("---")
    report.append("")

    # Sumário Executivo
    report.append("## Sumário Executivo")
    report.append("")
    report.append(f"Este relatório apresenta os resultados de **{len(df_summary)} configurações** testadas "
                 f"para otimizar o pipeline de segmentação de nódulos pulmonares PulmoSeg.")
    report.append("")

    baseline = df_summary[df_summary['config_name'] == 'baseline']
    best = df_summary.iloc[0]

    if len(baseline) > 0:
        baseline_dice = baseline['dice_mean'].values[0]
        best_dice = best['dice_mean']
        improvement = ((best_dice / baseline_dice - 1) * 100)

        report.append(f"- **Baseline (Configuração Original)**: Dice = {baseline_dice:.4f}")
        report.append(f"- **Melhor Configuração**: {best['config_name']}")
        report.append(f"  - Dice Score: {best_dice:.4f} ± {best['dice_std']:.4f}")
        report.append(f"  - IoU: {best['iou_mean']:.4f} ± {best['iou_std']:.4f}")
        report.append(f"  - **Melhoria**: {improvement:+.2f}%")
    else:
        report.append(f"- **Melhor Configuração**: {best['config_name']}")
        report.append(f"  - Dice Score: {best['dice_mean']:.4f} ± {best['dice_std']:.4f}")
        report.append(f"  - IoU: {best['iou_mean']:.4f} ± {best['iou_std']:.4f}")

    report.append("")
    report.append(f"- **Total de Slices Processadas**: {df_summary['n_slices'].sum():,}")
    report.append("")
    report.append("---")
    report.append("")

    # Metodologia
    report.append("## Metodologia")
    report.append("")
    report.append("### Amostragem")
    report.append("")
    report.append("- **Seleção de Pacientes**: Amostragem aleatória com seed fixa (seed=42) para reprodutibilidade")
    report.append("- **Critério de Inclusão**: Nódulos com pelo menos 2 máscaras válidas de radiologistas")
    report.append("- **Ground Truth**: Consenso de maioria (≥2 radiologistas concordam)")
    report.append("")

    report.append("### Parâmetros Testados")
    report.append("")
    report.append("Foram testadas combinações de:")
    report.append("")
    report.append("1. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**")
    report.append("   - clipLimit: [1.0, 2.0, 3.0, 4.0]")
    report.append("   - tileGridSize: [(4,4), (8,8), (16,16)]")
    report.append("")
    report.append("2. **Pré-processamento**")
    report.append("   - None (sem filtro)")
    report.append("   - Gaussian Blur")
    report.append("   - Median Filter")
    report.append("   - Bilateral Filter")
    report.append("")
    report.append("3. **Métodos de Threshold**")
    report.append("   - Otsu (automático)")
    report.append("   - Adaptive Threshold")
    report.append("   - Binary Threshold (fixo)")
    report.append("")
    report.append("4. **Operações Morfológicas**")
    report.append("   - Abertura (Opening)")
    report.append("   - Fechamento (Closing)")
    report.append("   - Gradiente Morfológico")
    report.append("   - Tamanhos de kernel: [3x3, 5x5, 7x7]")
    report.append("")
    report.append("### Métricas de Avaliação")
    report.append("")
    report.append("- **Dice Coefficient**: Mede sobreposição entre predição e ground truth")
    report.append("- **IoU (Intersection over Union)**: Métrica de similaridade geométrica")
    report.append("")
    report.append("---")
    report.append("")

    # Resultados
    report.append("## Resultados")
    report.append("")

    # Top 10
    report.append("### Top 10 Configurações")
    report.append("")
    report.append("| Rank | Configuração | Dice Score | IoU | Slices |")
    report.append("|------|--------------|------------|-----|--------|")

    for i, row in df_summary.head(10).iterrows():
        report.append(f"| {i+1} | {row['config_name']} | {row['dice_mean']:.4f} ± {row['dice_std']:.4f} | "
                     f"{row['iou_mean']:.4f} ± {row['iou_std']:.4f} | {row['n_slices']} |")

    report.append("")

    # Estatísticas Gerais
    report.append("### Estatísticas Gerais")
    report.append("")
    report.append(f"- **Dice Score Médio (todas configs)**: {df_summary['dice_mean'].mean():.4f}")
    report.append(f"- **Dice Score Mediano**: {df_summary['dice_median'].median():.4f}")
    report.append(f"- **Melhor Dice**: {df_summary['dice_mean'].max():.4f}")
    report.append(f"- **Pior Dice**: {df_summary['dice_mean'].min():.4f}")
    report.append(f"- **Range**: {df_summary['dice_mean'].max() - df_summary['dice_mean'].min():.4f}")
    report.append("")

    # Gráficos
    report.append("### Visualizações")
    report.append("")
    report.append("#### Ranking de Configurações")
    report.append("")
    report.append("![Dice Ranking](analysis/dice_ranking_top20.png)")
    report.append("")
    report.append("*Figura 1: Top 20 configurações rankeadas por Dice Score. "
                 "Baseline em vermelho para referência.*")
    report.append("")

    report.append("#### Dice vs IoU")
    report.append("")
    report.append("![Dice vs IoU](analysis/dice_vs_iou_scatter.png)")
    report.append("")
    report.append("*Figura 2: Correlação entre Dice Score e IoU. "
                 "Top 3 destacados em dourado, baseline em vermelho.*")
    report.append("")

    # Análise de Parâmetros
    report.append("---")
    report.append("")
    report.append("## Análise de Impacto dos Parâmetros")
    report.append("")

    # CLAHE
    if 'clip_limit' in df_summary.columns:
        report.append("### CLAHE clipLimit")
        report.append("")
        clip_analysis = df_summary.groupby('clip_limit')['dice_mean'].agg(['mean', 'std', 'count'])
        clip_analysis = clip_analysis[clip_analysis['count'] >= 3]

        if len(clip_analysis) > 0:
            best_clip = clip_analysis['mean'].idxmax()
            report.append(f"**Melhor valor**: {best_clip} (Dice médio: {clip_analysis.loc[best_clip, 'mean']:.4f})")
            report.append("")
            report.append("![CLAHE Impact](analysis/parameter_analysis/clahe_clip_limit_impact.png)")
            report.append("")

    # Threshold
    if 'threshold_method' in df_summary.columns:
        report.append("### Método de Threshold")
        report.append("")
        thresh_analysis = df_summary.groupby('threshold_method')['dice_mean'].agg(['mean', 'std', 'count'])
        thresh_analysis = thresh_analysis[thresh_analysis['count'] >= 3]

        if len(thresh_analysis) > 0:
            best_thresh = thresh_analysis['mean'].idxmax()
            report.append(f"**Melhor método**: {best_thresh} (Dice médio: {thresh_analysis.loc[best_thresh, 'mean']:.4f})")
            report.append("")
            report.append("![Threshold Impact](analysis/parameter_analysis/threshold_method_impact.png)")
            report.append("")

    # Morfologia
    if 'morph_operation' in df_summary.columns:
        report.append("### Operação Morfológica")
        report.append("")
        morph_analysis = df_summary.groupby('morph_operation')['dice_mean'].agg(['mean', 'std', 'count'])
        morph_analysis = morph_analysis[morph_analysis['count'] >= 3]

        if len(morph_analysis) > 0:
            best_morph = morph_analysis['mean'].idxmax()
            report.append(f"**Melhor operação**: {best_morph} (Dice médio: {morph_analysis.loc[best_morph, 'mean']:.4f})")
            report.append("")
            report.append("![Morph Impact](analysis/parameter_analysis/morph_operation_impact.png)")
            report.append("")

    # Pré-processamento
    if 'preprocessing' in df_summary.columns:
        report.append("### Pré-processamento")
        report.append("")
        df_prep = df_summary.copy()
        df_prep['preprocessing'] = df_prep['preprocessing'].fillna('none')
        prep_analysis = df_prep.groupby('preprocessing')['dice_mean'].agg(['mean', 'std', 'count'])
        prep_analysis = prep_analysis[prep_analysis['count'] >= 3]

        if len(prep_analysis) > 0:
            best_prep = prep_analysis['mean'].idxmax()
            report.append(f"**Melhor método**: {best_prep} (Dice médio: {prep_analysis.loc[best_prep, 'mean']:.4f})")
            report.append("")
            report.append("![Preprocessing Impact](analysis/parameter_analysis/preprocessing_impact.png)")
            report.append("")

    # Melhor Configuração Detalhada
    report.append("---")
    report.append("")
    report.append("## Configuração Recomendada")
    report.append("")

    if best_config:
        report.append(f"### {best_config.get('config_name', 'N/A')}")
        report.append("")
        report.append("**Parâmetros:**")
        report.append("")
        report.append("```json")
        # Filtrar apenas parâmetros relevantes
        params = {k: v for k, v in best_config.items()
                 if k not in ['config_name', 'n_slices', 'dice_mean', 'dice_std',
                             'dice_median', 'iou_mean', 'iou_std', 'iou_median']}
        report.append(json.dumps(params, indent=2))
        report.append("```")
        report.append("")
        report.append("**Desempenho:**")
        report.append("")
        report.append(f"- Dice Score: **{best_config.get('dice_mean', 0):.4f}** ± {best_config.get('dice_std', 0):.4f}")
        report.append(f"- IoU: **{best_config.get('iou_mean', 0):.4f}** ± {best_config.get('iou_std', 0):.4f}")
        report.append(f"- Slices processadas: {best_config.get('n_slices', 0)}")
        report.append("")

    # Conclusões
    report.append("---")
    report.append("")
    report.append("## Conclusões e Recomendações")
    report.append("")

    if len(baseline) > 0 and best_dice > baseline_dice:
        report.append(f"1. **Melhoria Significativa**: A otimização de parâmetros resultou em "
                     f"**{improvement:+.2f}%** de melhoria no Dice Score em relação à configuração baseline.")
    elif len(baseline) > 0:
        report.append("1. **Sem Melhoria**: A configuração baseline já apresenta desempenho próximo ao ótimo "
                     "para este dataset.")
    else:
        report.append("1. **Otimização Concluída**: Identificada a melhor combinação de parâmetros dentre "
                     "as configurações testadas.")

    report.append("")
    report.append("2. **Limitações das Técnicas Clássicas**: Mesmo otimizado, o pipeline clássico "
                 "(CLAHE + Otsu + Morfologia) apresenta limitações inerentes para segmentação de nódulos pulmonares, "
                 "especialmente em casos de baixo contraste ou presença de múltiplas estruturas.")
    report.append("")
    report.append("3. **Próximos Passos Sugeridos**:")
    report.append("   - Implementar abordagens de Deep Learning (U-Net, Mask R-CNN)")
    report.append("   - Explorar métodos híbridos (clássico + DL)")
    report.append("   - Aumentar tamanho da amostra para validação robusta")
    report.append("   - Testar em dados externos (generalização)")
    report.append("")

    # Referências
    report.append("---")
    report.append("")
    report.append("## Arquivos Gerados")
    report.append("")
    report.append("- `results/comparison_summary.csv` - Resumo comparativo de todas configurações")
    report.append("- `results/best_config.json` - Melhor configuração em formato JSON")
    report.append("- `results/test_runs/*.csv` - Resultados detalhados por configuração")
    report.append("- `results/visualizations/` - Casos extremos (best/worst) por configuração")
    report.append("- `results/analysis/` - Gráficos e análises")
    report.append("")

    report.append("---")
    report.append("")
    report.append("*Relatório gerado automaticamente por PulmoSeg Report Generator*")
    report.append("")

    return "\n".join(report)


def main():
    print("\n" + "="*70)
    print("PulmoSeg - Geração de Relatório")
    print("="*70 + "\n")

    # Carregar dados
    print("Carregando dados...")
    df_summary, best_config = load_data()
    print(f"✓ {len(df_summary)} configurações carregadas\n")

    # Gerar relatório
    print("Gerando relatório markdown...")
    report_content = generate_markdown_report(df_summary, best_config)

    # Salvar relatório
    report_path = Path('results') / 'OPTIMIZATION_REPORT.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"✓ Relatório salvo: {report_path}\n")

    print("="*70)
    print(f"Relatório completo gerado com sucesso!")
    print(f"Visualize em: {report_path}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
