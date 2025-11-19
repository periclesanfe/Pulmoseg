# PulmoSeg

**Segmentação Automática de Nódulos Pulmonares: Da Visão Computacional Clássica ao Deep Learning**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## 📋 Sumário

- [Visão Geral](#-visão-geral)
- [Motivação](#-motivação)
- [Dataset](#-dataset-lidc-idri)
- [Metodologia](#-metodologia)
  - [Fase 1: Técnicas Clássicas](#fase-1-técnicas-clássicas-de-processamento-de-imagens)
  - [Fase 2: Deep Learning](#fase-2-deep-learning)
- [Resultados](#-resultados)
- [Instalação e Uso](#-instalação-e-uso)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Contribuições](#-contribuições)
- [Licença](#-licença)
- [Autores](#-autores)
- [Referências](#-referências)

---

## 🔬 Visão Geral

**PulmoSeg** é um sistema completo de segmentação automática de nódulos pulmonares desenvolvido como prova de conceito para avaliar e comparar duas abordagens distintas:

1. **Técnicas Clássicas** de Processamento de Imagens (CLAHE + Otsu + Morfologia)
2. **Deep Learning** com arquiteturas estado-da-arte (U-Net, Attention U-Net, U-Net++)

O projeto utiliza o dataset **LIDC-IDRI** (Lung Image Database Consortium), que contém anotações de múltiplos radiologistas para validação rigorosa dos métodos propostos.

---

## 💡 Motivação

### Por que Segmentação de Nódulos Pulmonares?

- **Câncer de Pulmão** é a principal causa de morte por câncer no mundo
- **Detecção precoce** aumenta significativamente a taxa de sobrevivência (>70% se detectado no estágio I)
- **Radiologistas** precisam analisar centenas de imagens por paciente manualmente
- **Segmentação automática** pode:
  - Reduzir tempo de análise
  - Aumentar consistência no diagnóstico
  - Auxiliar no planejamento de tratamento
  - Permitir monitoramento quantitativo da evolução

### Por que Comparar Técnicas Clássicas e Deep Learning?

- **Baseline científico**: Estabelecer performance de métodos tradicionais
- **Justificar complexidade**: Demonstrar necessidade de Deep Learning
- **Entender limitações**: Identificar onde técnicas clássicas falham
- **Educacional**: Mostrar evolução de métodos na área médica

---

## 🗂️ Dataset: LIDC-IDRI

### Características

- **1,018 casos** de TC de tórax
- **Anotações de 4 radiologistas** experientes por nódulo
- **~2,600 nódulos** anotados
- **Fatias 2D** extraídas de volumes 3D
- **Formato**: PNG (512×512 pixels)

### Estrutura

```
LIDC-IDRI-slices/
└── LIDC-IDRI-XXXX/          # ID do Paciente (874 pacientes)
    └── nodule-Y/             # ID do Nódulo
        ├── images/           # Fatias da TC
        │   ├── slice-0.png
        │   ├── slice-1.png
        │   └── slice-N.png
        ├── mask-0/           # Anotação do Radiologista 1
        ├── mask-1/           # Anotação do Radiologista 2
        ├── mask-2/           # Anotação do Radiologista 3
        └── mask-3/           # Anotação do Radiologista 4
```

### Ground Truth

**Máscara de Consenso**: Um pixel é considerado nódulo se **≥2 radiologistas concordam**

Essa abordagem:
- Balanceia sensibilidade e especificidade
- Reduz impacto de anotações individuais ruidosas
- É amplamente aceita na literatura médica

---

## 🔬 Metodologia

---

## Fase 1: Técnicas Clássicas de Processamento de Imagens

> 📁 **Código**: `classical_methods/`
>
> 📖 **Documentação completa**: [classical_methods/README.md](classical_methods/README.md)

### Pipeline Básico

```
Imagem TC → Grayscale → CLAHE → Otsu → Morfologia → Máscara Binária
```

**Etapas detalhadas:**

1. **Aquisição e Conversão**
   - Carregamento da imagem em escala de cinza (8-bit)

2. **Realce de Contraste (CLAHE)**
   - **Algoritmo**: Contrast Limited Adaptive Histogram Equalization
   - **Parâmetros**: `clipLimit=2.0`, `tileGridSize=(8,8)`
   - **Objetivo**: Melhorar contraste local em TCs de baixo contraste

3. **Segmentação (Limiarização de Otsu)**
   - **Algoritmo**: Threshold automático baseado em histograma
   - **Objetivo**: Separar nódulo do fundo e tecido pulmonar

4. **Operações Morfológicas**
   - **Algoritmo**: Abertura (erosão + dilatação)
   - **Kernel**: Elíptico 3×3
   - **Objetivo**: Remover ruído e pequenas estruturas espúrias

5. **Extração de Atributos**
   - Cálculo da área do nódulo segmentado (número de pixels)

### Otimização Sistemática

Para encontrar a melhor combinação de parâmetros, implementamos **Grid Search** testando:

| Componente | Valores Testados | Total |
|------------|------------------|-------|
| CLAHE clipLimit | 1.0, 2.0, 3.0, 4.0 | 4 |
| CLAHE tileGridSize | (4,4), (8,8), (16,16) | 3 |
| Threshold | Otsu, Adaptive, Binary | 3 |
| Morfologia | Abertura, Fechamento, Gradiente | 3 |
| Kernel Size | 3×3, 5×5, 7×7 | 3 |
| Pré-processamento | None, Gaussian, Median, Bilateral | 4 |

**Total**: **31 configurações** testadas

### Estratégia de Testes

- **Amostragem**: 10 pacientes (seed=42 para reprodutibilidade)
- **Slices processadas**: 3,565 (115 slices × 31 configurações)
- **Métricas**: Dice Coefficient e IoU
- **Validação**: Comparação contra máscara de consenso
- **Visualização**: Top 5 e Bottom 5 casos por configuração
- **Tempo de execução**: ~45 segundos

### Resultados - Fase 1

#### Melhor Configuração: `morph_gradient_k5`

**Parâmetros otimizados:**
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

| Métrica | Baseline | Otimizado | Melhoria |
|---------|----------|-----------|----------|
| **Dice Score** | 0.0131 ± 0.0160 | **0.0233 ± 0.0294** | **+78.26%** |
| **IoU** | 0.0066 ± 0.0082 | **0.0120 ± 0.0155** | **+81.82%** |

#### Top 5 Configurações

| Rank | Configuração | Dice Score | Insight Principal |
|------|--------------|------------|-------------------|
| 1️⃣ | morph_gradient_k5 | 0.0233 ± 0.0294 | **Gradiente morfológico** superior a abertura/fechamento |
| 2️⃣ | morph_gradient_k3 | 0.0228 ± 0.0334 | Kernel 3×3 vs 5×5 tem performance similar |
| 3️⃣ | morph_gradient_k7 | 0.0220 ± 0.0259 | Kernel maior não melhora significativamente |
| 4️⃣ | threshold_adaptive | 0.0191 ± 0.0171 | **Adaptive threshold** superior a Otsu |
| 5️⃣ | clahe_clip4.0_tile8x8 | 0.0173 ± 0.0178 | CLAHE forte melhora detecção de bordas |

#### Insights Principais

✅ **O que funcionou:**
- **Gradiente Morfológico** foi a mudança mais impactante (+78% melhoria)
- **Adaptive Threshold** superou Otsu tradicional
- **CLAHE com clip=4.0** melhorou detecção de bordas

❌ **O que não funcionou:**
- **Pré-processamento adicional** (Gaussian/Median/Bilateral) não melhorou
- **Kernels grandes** (7×7) não trouxeram benefícios
- **Fechamento morfológico** teve performance inferior

#### Limitações Identificadas

1. **Scores muito baixos** (~2.3% Dice): Mesmo otimizado, desempenho insuficiente para uso clínico
2. **Alto desvio padrão**: Resultados muito variáveis entre slices
3. **Falhas em baixo contraste**: Método falha quando nódulo tem intensidade similar ao tecido
4. **Múltiplas estruturas**: Não diferencia nódulo de vasos sanguíneos e outras estruturas
5. **Sem contexto**: Processa cada slice independentemente (sem informação 3D)

**Conclusão da Fase 1**: Técnicas clássicas estabelecem baseline, mas performance inadequada justifica necessidade de Deep Learning.

---

## Fase 2: Deep Learning

> 📁 **Código**: `deep_learning/`
>
> 📖 **Documentação completa**: [deep_learning/README.md](deep_learning/README.md)

### Arquiteturas Implementadas

#### 1. **U-Net 2D** (Baseline DL)

**Características:**
- Arquitetura encoder-decoder com skip connections
- Encoder: ResNet34 pré-treinado (ImageNet)
- Decoder: Upsampling com concatenação de features
- **Total de parâmetros**: ~24M

**Por que U-Net?**
- Padrão-ouro para segmentação médica
- Skip connections preservam detalhes espaciais
- Transfer learning acelera convergência

#### 2. **Attention U-Net** (MAnet)

**Características:**
- Attention gates nas skip connections
- Supressão de features irrelevantes
- Foco automático em regiões de interesse
- **Total de parâmetros**: ~26M

**Vantagem:**
- Aprende onde focar automaticamente
- Reduz impacto de artefatos e ruído

#### 3. **U-Net++**

**Características:**
- Nested e dense skip connections
- Sub-redes em múltiplas profundidades
- Ensemble implícito de U-Nets
- **Total de parâmetros**: ~32M

**Vantagem:**
- Estado-da-arte em segmentação médica
- Melhor propagação de gradientes

### Pipeline de Treinamento

#### Split do Dataset

- **70% Treino** (~600 pacientes, ~2000 slices)
- **15% Validação** (~130 pacientes, ~400 slices)
- **15% Teste** (~130 pacientes, ~400 slices)

**Importante**: Split por **pacientes**, não por slices (evita data leakage)

#### Data Augmentation

Pipeline conservador para imagens médicas:

| Transformação | Parâmetros | Prob. |
|---------------|------------|-------|
| Rotação | ±15° | 70% |
| Flip H/V | - | 50% |
| Escala | 0.9-1.1 | 70% |
| Brilho/Contraste | ±20% | 50% |
| Elastic Deformation | α=50, σ=8 | 30% |
| Grid Distortion | - | 20% |
| Optical Distortion | - | 20% |

#### Loss Function

**Combined Loss**: `0.5 × Dice Loss + 0.5 × Binary Cross-Entropy`

**Justificativa:**
- **Dice Loss**: Foca em sobreposição (similar à métrica de avaliação)
- **BCE**: Penaliza predições pixel-a-pixel
- **Combinação**: Balanceia sensibilidade regional e precisão local

#### Otimizador e Scheduler

- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-5)
- **Scheduler**: ReduceLROnPlateau (patience=5, factor=0.5)
- **Early Stopping**: Patience=15 epochs
- **Gradient Clipping**: max_norm=1.0

#### Configurações Técnicas

```python
BATCH_SIZE = 8              # Otimizado para M2 16GB RAM
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = 'mps'              # Metal Performance Shaders (Apple Silicon)
USE_MIXED_PRECISION = True  # FP16 para economizar memória
GRADIENT_CLIP_VAL = 1.0
NUM_WORKERS = 4
```

### Resultados - Fase 2

> ⚠️ **Status**: 🔄 **Treinamento pendente** - Resultados serão adicionados após execução

#### Resultados Esperados (Baseados em Literatura)

| Modelo | Dice Score Esperado | IoU Esperado | Melhoria vs Clássico |
|--------|---------------------|--------------|----------------------|
| Baseline Clássico | 0.013 | 0.007 | - |
| Otimizado Clássico | 0.023 | 0.012 | +77% |
| **U-Net** | **0.70-0.75** | **0.60-0.65** | **+5285%** |
| **Attention U-Net** | **0.73-0.80** | **0.63-0.70** | **+5515%** |
| **U-Net++** | **0.75-0.85** | **0.65-0.75** | **+5670%** |

#### Resultados Obtidos

*Seção será preenchida após treinamento dos modelos*

**Para gerar resultados:**
```bash
cd deep_learning
python train.py --model unet --experiment-name unet_baseline
python train.py --model manet --experiment-name attention_unet
python train.py --model unetplusplus --experiment-name unet_plusplus
```

---

## 📊 Comparação Final: Clássico vs Deep Learning

### Performance por Método

| Rank | Método | Tipo | Dice Score | IoU | Tempo Inf. |
|------|--------|------|------------|-----|------------|
| 🥉 | Baseline (Otsu) | Clássico | 0.0131 | 0.0066 | ~10ms |
| 🥈 | Otimizado (Gradient) | Clássico | 0.0233 | 0.0120 | ~15ms |
| 🥇 | **U-Net** | **DL** | **~0.72** | **~0.62** | **~50ms** |
| 🥇 | **Attention U-Net** | **DL** | **~0.76** | **~0.67** | **~60ms** |
| 🥇 | **U-Net++** | **DL** | **~0.80** | **~0.71** | **~70ms** |

*(Valores DL são estimativas baseadas em literatura - serão atualizados após treinamento)*

### Trade-offs

#### Técnicas Clássicas
**Vantagens:**
- ✅ Extremamente rápidas (~10-15ms por slice)
- ✅ Sem necessidade de GPU
- ✅ Interpretáveis (cada etapa é compreensível)
- ✅ Sem necessidade de dados de treinamento

**Desvantagens:**
- ❌ Performance insuficiente para uso clínico
- ❌ Sensível a variações de contraste e ruído
- ❌ Não aprende com dados
- ❌ Requer ajuste manual de parâmetros

#### Deep Learning
**Vantagens:**
- ✅ Performance próxima de especialistas humanos
- ✅ Aprende automaticamente features relevantes
- ✅ Robusta a variações e ruído
- ✅ Generaliza bem para novos dados

**Desvantagens:**
- ❌ Requer GPU para treinamento
- ❌ Tempo de treinamento longo (horas)
- ❌ "Black box" (difícil interpretabilidade)
- ❌ Requer grande volume de dados anotados

---

## 🚀 Instalação e Uso

### Pré-requisitos

- Python 3.8+
- 16GB RAM recomendado
- GPU NVIDIA (opcional, mas recomendado para DL)
- ou Apple Silicon M1/M2/M3 (suporte MPS)

### Instalação

```bash
# 1. Clonar repositório
git clone https://github.com/seu-usuario/Pulmoseg.git
cd Pulmoseg

# 2. Criar ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# 3. Instalar dependências clássicas
cd classical_methods
pip install -r requirements.txt

# 4. Instalar dependências DL
cd ../deep_learning
pip install -r requirements-dl.txt
```

### Uso Rápido

#### Técnicas Clássicas

```bash
cd classical_methods

# Processar 10 pacientes
python main.py --limit 10

# Otimização completa
python test_optimizer.py --patients 10 --seed 42

# Gerar análises
python analyze_results.py
python report_generator.py
```

#### Deep Learning

```bash
cd deep_learning

# Treinar U-Net
python train.py --model unet

# Treinar Attention U-Net
python train.py --model manet

# Monitorar TensorBoard
tensorboard --logdir=runs

# Avaliar
python evaluate.py --checkpoint checkpoints/unet_resnet34/best.pth

# Visualizar
python visualize_predictions.py --checkpoint checkpoints/unet_resnet34/best.pth
```

---

## 📂 Estrutura do Projeto

```
Pulmoseg/
│
├── README.md                        # 👈 Este arquivo (documentação principal)
├── LICENSE                          # MIT License
├── requirements.txt                 # Dependências compartilhadas
│
├── LIDC-IDRI-slices/               # 📊 Dataset (não versionado)
│   └── LIDC-IDRI-XXXX/
│       └── nodule-Y/
│           ├── images/
│           └── mask-{0,1,2,3}/
│
├── classical_methods/              # 🔬 FASE 1: Técnicas Clássicas
│   ├── README.md                    # Documentação detalhada
│   ├── main.py                      # Pipeline base
│   ├── pulmoseg_processor.py        # Classe processadora
│   ├── test_optimizer.py            # Grid search
│   ├── analyze_results.py           # Análise comparativa
│   ├── report_generator.py          # Geração de relatórios
│   ├── requirements.txt
│   └── results/                     # Resultados gerados
│       ├── metrics.csv
│       ├── comparison_summary.csv
│       ├── best_config.json
│       ├── OPTIMIZATION_REPORT.md
│       └── visualizations/
│
└── deep_learning/                  # 🧠 FASE 2: Deep Learning
    ├── README.md                    # Documentação detalhada
    ├── train.py                     # Script de treinamento
    ├── evaluate.py                  # Avaliação no test set
    ├── compare_models.py            # Comparação DL vs Clássico
    ├── visualize_predictions.py     # Visualização
    ├── requirements-dl.txt
    └── src/
        ├── config.py                # Configurações
        ├── dataset.py               # DataLoader
        ├── augmentation.py          # Augmentation pipeline
        ├── losses.py                # Loss functions
        ├── metrics.py               # Métricas
        ├── trainer.py               # Training loop
        └── models/
            └── unet.py              # Arquiteturas (U-Net, Attention, U-Net++)
```

---

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

---

## 📄 Licença

Este projeto está licenciado sob a **MIT License** - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

## 👨‍💻 Autores

**Pericles Feitoza** - *Desenvolvimento e Pesquisa* - [GitHub](https://github.com/seu-usuario)

Desenvolvido como prova de conceito para segmentação de nódulos pulmonares.

---

## 📚 Referências

### Artigos Científicos

#### Técnicas Clássicas
1. **CLAHE**: Zuiderveld, Karel. "Contrast limited adaptive histogram equalization." Graphics gems (1994): 474-485.
2. **Otsu's Method**: Otsu, Nobuyuki. "A threshold selection method from gray-level histograms." IEEE transactions on systems, man, and cybernetics 9.1 (1979): 62-66.

#### Deep Learning
3. **U-Net**: Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.
4. **Attention U-Net**: Oktay, Ozan, et al. "Attention u-net: Learning where to look for the pancreas." arXiv preprint arXiv:1804.03999 (2018).
5. **U-Net++**: Zhou, Zongwei, et al. "Unet++: Redesigning skip connections to exploit multiscale features in image segmentation." IEEE transactions on medical imaging 39.6 (2019): 1856-1867.

### Dataset
6. **LIDC-IDRI**: Armato III, Samuel G., et al. "The lung image database consortium (LIDC) and image database resource initiative (IDRI): a completed reference database of lung nodules on CT scans." Medical physics 38.2 (2011): 915-931.

### Frameworks e Bibliotecas
7. **PyTorch**: Paszke, Adam, et al. "Pytorch: An imperative style, high-performance deep learning library." Advances in neural information processing systems 32 (2019).
8. **Albumentations**: Buslaev, Alexander, et al. "Albumentations: fast and flexible image augmentations." Information 11.2 (2020): 125.
9. **segmentation-models-pytorch**: Iakubovskii, Pavel. "Segmentation Models Pytorch." GitHub repository (2019).

---

## 🎯 Próximos Passos

### Melhorias Planejadas

- [ ] **Implementar segmentação 3D**: Processar volumes completos ao invés de slices 2D
- [ ] **Ensemble de modelos**: Combinar predições de múltiplos modelos
- [ ] **Análise de incerteza**: Quantificar confiança das predições
- [ ] **Explicabilidade**: Implementar Grad-CAM para visualizar regiões relevantes
- [ ] **Transfer learning avançado**: Fine-tuning com dados médicos específicos
- [ ] **Otimização de inferência**: Quantização e pruning para deployment
- [ ] **Interface web**: Criar interface amigável para uso clínico
- [ ] **Validação externa**: Testar em outros datasets (LUNA16, NLST)

### Oportunidades de Pesquisa

- Comparação com métodos híbridos (clássico + DL)
- Estudo de generalização cross-dataset
- Análise de viés e fairness em diferentes populações
- Integração com sistemas PACS hospitalares

---

## 🙏 Agradecimentos

- **LIDC-IDRI Consortium** pelo dataset público
- **PyTorch Team** pelo framework excepcional
- **segmentation-models-pytorch** pela biblioteca de modelos
- **Comunidade open-source** de computer vision médica

---

## 📧 Contato

Para questões, sugestões ou colaborações:

- **Email**: seu-email@example.com
- **GitHub Issues**: [github.com/seu-usuario/Pulmoseg/issues](https://github.com/seu-usuario/Pulmoseg/issues)

---

<div align="center">

**Desenvolvido com ❤️ para avançar a segmentação automática de nódulos pulmonares**

⭐ Se este projeto foi útil, considere dar uma estrela no GitHub!

</div>
