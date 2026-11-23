# Treinamento e avaliação de detectores DETR em GPU

Sistema desktop completo para detecção de objetos usando modelos DETR (DEtection TRansformer), otimizado para **NVIDIA RTX 5080** com arquitetura Blackwell.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.1+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Arquivos de referência**

- [main.py](main.py)
- [requirements.txt](requirements.txt)

## ✨ Funcionalidades

### 📁 Gerenciamento de Dataset
- Download direto do Roboflow (formato COCO)
- Seleção de datasets locais
- Validação automática de estrutura COCO
- Relatório de estatísticas do dataset

### 🎯 Treinamento
- **Modelos suportados:**
  - DAB-DETR (recomendado)
  - DETR ResNet-50/101
  - Deformable DETR
- **Otimizações RTX 5080:**
  - Mixed Precision (BF16/FP16)
  - TF32 para Tensor Cores
  - torch.compile (PyTorch 2.x)
  - DataLoader otimizado
- Auto-tuning baseado em VRAM
- Checkpoints automáticos
- Métricas em tempo real
- TensorBoard integrado

### 🔍 Inferência
- Processamento de vídeo em tempo real
- Preview com limitador de FPS
- Modo rápido (alta performance)
- Exportação de vídeo anotado
- Estatísticas de performance

### 📊 Avaliação
- Métricas COCO completas:
  - mAP (IoU=0.50:0.95)
  - AP@50, AP@75
  - AP por tamanho (small/medium/large)
  - Precision e Recall
- Relatório por classe
- Exportação JSON

## 🚀 Instalação

### Requisitos
- Windows 10/11
- Python 3.10+
- NVIDIA GPU (RTX 5080 recomendado)
- CUDA 12.1+
- 16GB+ RAM

### Setup

```bash
# 1. Clonar ou navegar até o diretório
cd C:\Users\Usuario\OneDrive\Documentos\Sistema_DETR

# 2. Criar ambiente virtual (recomendado)
python -m venv venv
venv\Scripts\activate

# 3. Instalar PyTorch com CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Instalar dependências
pip install -r requirements.txt
```

## 🎮 Uso

### Iniciar Aplicação
```bash
python main.py
```

### Fluxo de Trabalho Típico

1. **Dataset:** Configure a API key do Roboflow e baixe um dataset, ou selecione um dataset local no formato COCO.

2. **Treinamento:** Selecione o modelo, ajuste hiperparâmetros e inicie o treinamento. O sistema ajusta automaticamente batch size e image size baseado na VRAM disponível.

3. **Predição:** Carregue o modelo treinado e processe vídeos. O preview em tempo real mostra as detecções.

4. **Avaliação:** Execute avaliação COCO completa para obter métricas detalhadas.

## 📁 Estrutura do Projeto

```
Sistema_DETR/
├── main.py                 # Ponto de entrada
├── requirements.txt        # Dependências
├── README.md              
│
├── src/
│   ├── core/              # Módulos centrais
│   │   ├── config.py      # Configurações
│   │   ├── logger.py      # Sistema de log
│   │   ├── device_manager.py  # Gerenciamento GPU
│   │   └── session.py     # Gerenciamento de sessão
│   │
│   ├── data/              # Manipulação de dados
│   │   └── coco_dataset.py
│   │
│   ├── models/            # Modelos e treinamento
│   │   ├── detr_trainer.py
│   │   └── detr_inference.py
│   │
│   ├── evaluation/        # Avaliação
│   │   └── coco_evaluator.py
│   │
│   └── ui/                # Interface gráfica
│       ├── main_window.py
│       ├── dataset_tab.py
│       ├── training_tab.py
│       ├── prediction_tab.py
│       └── evaluation_tab.py
│
├── data/
│   ├── datasets/          # Datasets COCO
│   ├── models/
│   │   └── experiments/   # Experimentos salvos
│   ├── outputs/
│   │   └── predictions/   # Vídeos anotados
│   ├── logs/              # Logs do sistema
│   └── reports/           # Relatórios de avaliação
│
└── runs/                  # TensorBoard logs
```

## ⚙️ Configurações

### Classes de Detecção
O sistema utiliza 3 classes configuradas para o dataset:
1. **ok** (0) - itens em bom estado
2. **ref** (1) - itens com referência/defeito
3. **vazio** (2) - posições vazias

### Dataset Padrão (Roboflow)
URL: `https://app.roboflow.com/ds/28uCRbc4wr?key=q7qHpXbRBo`
Versão: 5

### Otimizações RTX 5080

| Otimização | Descrição | Ganho Esperado |
|------------|-----------|----------------|
| BF16 | Brain Float 16 | 2x throughput |
| TF32 | TensorFloat-32 | 1.5x matmul |
| torch.compile | Compilação JIT | 1.3x geral |
| cuDNN benchmark | Auto-tuning | 1.1x conv |

### Auto-tuning por VRAM

| VRAM | Batch Size | Image Size | Grad Accum |
|------|------------|------------|------------|
| ≥16GB | 16 | 800 | 1 |
| 8-16GB | 8 | 800 | 2 |
| <8GB | 4 | 640 | 4 |

## 📈 Monitoramento

### TensorBoard
```bash
tensorboard --logdir=runs
```

### Métricas CSV
Métricas de treinamento são salvas em:
```
data/models/experiments/{nome}/train_metrics.csv
```

## 🔧 Troubleshooting

### CUDA Out of Memory
O sistema tenta automaticamente reduzir batch_size e image_size. Se persistir:
1. Reduza manualmente batch_size
2. Reduza image_size para 640
3. Aumente grad_accum

### Modelo não carrega
Verifique se o arquivo `best_model.pt` existe no diretório do experimento.

### DataLoader lento no Windows
Reduza `num_workers` para 4 ou 0.

## 📝 Licença

MIT License - Veja LICENSE para detalhes.

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor, abra uma issue ou pull request.

---

**Desenvolvido para NVIDIA RTX 5080 (Blackwell) - Máxima Performance em Detecção de Objetos**
