# Experimentos com Técnicas Clássicas de Processamento de Imagens

Esta pasta contém a **Fase 1** do projeto PulmoSeg: implementação e otimização de técnicas clássicas (não-Deep Learning) para segmentação de nódulos pulmonares.

---

## 📋 Objetivo

Avaliar a eficácia de técnicas tradicionais de processamento de imagens médicas antes de explorar abordagens de Deep Learning, estabelecendo uma **baseline** para comparação.

---

## 🔬 Metodologia

### Pipeline Base (main.py)

**Etapas do processamento:**

1. **Aquisição e Conversão**: Carregamento da imagem TC em escala de cinza
2. **Realce (CLAHE)**: Contrast Limited Adaptive Histogram Equalization
   - `clipLimit=2.0`
   - `tileGridSize=(8, 8)`
3. **Segmentação (Otsu)**: Limiarização automática de Otsu
4. **Morfologia**: Operação de abertura (3x3) para remover ruído
5. **Extração de Atributos**: Cálculo da área do nódulo segmentado

### Validação

- **Ground Truth**: Máscara de consenso (≥2 radiologistas concordam)
- **Métricas**: Dice Coefficient e IoU (Intersection over Union)
- **Dataset**: LIDC-IDRI com split por pacientes (seed=42)

---

## 🧪 Otimização de Parâmetros (test_optimizer.py)

Grid search sistemático testando **31 configurações**:

### Parâmetros Testados

| Componente | Valores Testados |
|------------|------------------|
| **CLAHE clipLimit** | 1.0, 2.0, 3.0, 4.0 |
| **CLAHE tileGridSize** | (4,4), (8,8), (16,16) |
| **Threshold** | Otsu, Adaptive, Binary |
| **Morfologia** | Abertura, Fechamento, Gradiente |
| **Kernel Size** | 3x3, 5x5, 7x7 |
| **Pré-processamento** | None, Gaussian, Median, Bilateral |

### Estratégia de Teste

- **Amostragem**: 10 pacientes (seed=42 para reprodutibilidade)
- **Total de slices processadas**: 3,565
- **Casos extremos salvos**: Top 5 e Bottom 5 por Dice Score
- **Tempo de execução**: ~45 segundos (31 configurações)

---

## 📊 Resultados

### Melhor Configuração: `morph_gradient_k5`

**Parâmetros:**
```json
{
  "clip_limit": 2.0,
  "tile_grid_size": [8, 8],
  "preprocessing": null,
  "threshold_method": "otsu",
  "morph_operation": "gradient",
  "morph_kernel_size": 5
}
```

**Performance:**

| Métrica | Baseline | Melhor Config | Melhoria |
|---------|----------|---------------|----------|
| **Dice Score** | 0.0131 ± 0.0160 | **0.0233 ± 0.0294** | **+78.26%** |
| **IoU** | 0.0066 ± 0.0082 | **0.0120 ± 0.0155** | **+81.82%** |

### Top 5 Configurações

| Rank | Configuração | Dice Score | Observação |
|------|--------------|------------|------------|
| 1️⃣ | `morph_gradient_k5` | 0.0233 ± 0.0294 | Gradiente morfológico 5x5 |
| 2️⃣ | `morph_gradient_k3` | 0.0228 ± 0.0334 | Gradiente morfológico 3x3 |
| 3️⃣ | `morph_gradient_k7` | 0.0220 ± 0.0259 | Gradiente morfológico 7x7 |
| 4️⃣ | `threshold_adaptive` | 0.0191 ± 0.0171 | Threshold adaptativo |
| 5️⃣ | `clahe_clip4.0_tile8x8` | 0.0173 ± 0.0178 | CLAHE forte (clip=4.0) |

### Insights Principais

✅ **Gradiente Morfológico** foi a mudança mais impactante (+78% melhoria)
✅ **Adaptive Threshold** superou Otsu em alguns casos
✅ **CLAHE com clip=4.0** melhorou detecção de bordas
❌ **Pré-processamento adicional** (Gaussian/Median) não melhorou resultados
❌ **Scores baixos** (~2.3% Dice) confirmam limitações de técnicas clássicas

---

## 📁 Arquivos

```
classical_methods/
├── README.md                    # Este arquivo
├── main.py                      # Pipeline base + visualização
├── pulmoseg_processor.py        # Classe processadora e métricas
├── test_optimizer.py            # Grid search de parâmetros
├── analyze_results.py           # Análise comparativa + gráficos
├── report_generator.py          # Geração de relatório markdown
├── requirements.txt             # Dependências Python
└── results/
    ├── metrics.csv              # Resultados do baseline
    ├── comparison_summary.csv   # Resumo de todas configs
    ├── best_config.json         # Melhor configuração
    ├── OPTIMIZATION_REPORT.md   # Relatório completo
    ├── test_runs/               # CSVs detalhados (31 arquivos)
    ├── visualizations/          # Casos extremos por config
    └── analysis/                # Gráficos de análise
        ├── dice_ranking_top20.png
        ├── dice_vs_iou_scatter.png
        └── parameter_analysis/
```

---

## 🚀 Como Reproduzir

### 1. Instalar Dependências

```bash
cd classical_methods
pip install -r requirements.txt
```

### 2. Executar Pipeline Base

```bash
# Processar 10 pacientes com visualização
python main.py --limit 10

# Processar dataset completo sem visualização
python main.py --no-visualize
```

### 3. Executar Otimização

```bash
# Testar 31 configurações com 10 pacientes
python test_optimizer.py --patients 10 --seed 42

# Testar com 50 pacientes (mais robusto)
python test_optimizer.py --patients 50 --seed 42
```

### 4. Analisar Resultados

```bash
# Gerar gráficos comparativos
python analyze_results.py

# Gerar relatório markdown completo
python report_generator.py
```

---

## 📈 Visualizações Geradas

### Gráficos de Análise

- **Ranking de Configurações** (Top 20 por Dice Score)
- **Scatter Plot** (Dice vs IoU)
- **Impacto de Parâmetros**:
  - CLAHE clipLimit
  - Método de Threshold
  - Operação Morfológica
  - Pré-processamento

### Casos Extremos

Para cada configuração testada:
- **5 melhores casos** (maior Dice Score)
- **5 piores casos** (menor Dice Score)

---

## 🎯 Conclusões

1. **Técnicas clássicas têm limitações inerentes** para segmentação de nódulos pulmonares
2. **Gradiente morfológico** foi significativamente superior às operações tradicionais
3. **Scores baixos** (2.3% Dice) justificam necessidade de Deep Learning
4. **Otimização sistemática** pode melhorar resultados, mas não resolve o problema fundamental

### Próximos Passos

➡️ **Deep Learning** (U-Net, Attention U-Net, U-Net++) implementado em `../deep_learning/`

---

## 📚 Referências Técnicas

- **CLAHE**: Zuiderveld, Karel. "Contrast limited adaptive histogram equalization." (1994)
- **Otsu's Method**: Otsu, Nobuyuki. "A threshold selection method from gray-level histograms." (1979)
- **LIDC-IDRI Dataset**: Armato III et al. "The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI)" (2011)

---

**Desenvolvido como baseline para o projeto PulmoSeg - Segmentação de Nódulos Pulmonares**
