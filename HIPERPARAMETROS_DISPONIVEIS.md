# Hiperparâmetros Disponíveis - Plataforma DETR

## ✅ Hiperparâmetros Implementados

### 📊 **Treinamento Básico**
| Hiperparâmetro | Descrição | Valores Recomendados |
|----------------|-----------|----------------------|
| **Épocas** | Número total de épocas | RT-DETR: 12-20, DETR: 50-100 |
| **Batch Size** | Imagens por batch | RTX 5080: 12-16 (RT-DETR R101), 8-12 (RT-DETR R50) |
| **Grad Accum** | Acumulação de gradientes | Batch efetivo = batch_size × grad_accum |
| **Image Size** | Tamanho da imagem (px) | 640-800px (objetos grandes), 960px (objetos pequenos) |
| **Learning Rate** | Taxa de aprendizado | 1e-5 (padrão), 5e-6 (objetos pequenos) |
| **Weight Decay** | Regularização L2 | 1e-4 (padrão), 5e-4 (objetos pequenos) |
| **Warmup Steps** | Steps com LR crescente | 500-1000 |
| **Max Grad Norm** | Gradient clipping | 1.0 (padrão), 0.5 (objetos pequenos) |

### 📈 **Scheduler (Learning Rate)**
| Tipo | Descrição | Uso |
|------|-----------|-----|
| **Cosine** | Decaimento cosseno | ✅ Recomendado (padrão) |
| **Linear** | Decaimento linear | Alternativa simples |
| **Polynomial** | Decaimento polinomial | Raro |
| **Constant** | LR fixo | Não recomendado |

### 🚀 **Otimizações RTX 5080**
| Otimização | Descrição | Status |
|------------|-----------|--------|
| **Precision** | BF16/FP16/FP32 | ✅ BF16 recomendado |
| **TF32** | Tensor Cores | ✅ Habilitado (RTX 5080) |
| **torch.compile** | Compilação PyTorch 2.x | ⚠️ Desabilitado no Windows |

### 📦 **DataLoader**
| Parâmetro | Descrição | Valor |
|-----------|-----------|-------|
| **Workers** | Processos paralelos | 8 (recomendado) |
| **Pin Memory** | Acelera CPU→GPU | ✅ Habilitado |

### 🛡️ **Anti-Overfitting**
| Hiperparâmetro | Descrição | Valores |
|----------------|-----------|---------|
| **Dropout** | Regularização | 0.1 (padrão), 0.2 (objetos pequenos) |
| **Early Stopping** | Patience (épocas) | 10 (padrão), 7 (objetos pequenos) |
| **Label Smoothing** | Suavização de labels | 0.0 (padrão), 0.1 (objetos pequenos) |
| **Augmentation** | Data augmentation | ✅ Habilitado (flip, color, blur, cutout) |

### ⚖️ **Loss Weights**
| Loss | Descrição | Padrão | Objetos Pequenos | Objetos Grandes |
|------|-----------|--------|------------------|-----------------|
| **Bbox Loss** | Localização (L1) | 5.0 | 8.0 | 5.0 |
| **GIoU Loss** | Sobreposição (IoU) | 2.0 | 2.5 | 2.0 |
| **Class Loss** | Classificação | 1.0 | 1.0 | 1.0 |
| **EOS Coeff** | "No object" | 0.1 | 0.1 | 0.1 |

---

## ✅ **Hiperparâmetros Importantes - Todos Disponíveis**

### Para RT-DETR:
- ✅ Learning Rate, Weight Decay, Warmup
- ✅ Batch Size, Gradient Accumulation
- ✅ Image Size
- ✅ Scheduler (Cosine)
- ✅ Loss Weights (Bbox, GIoU, Class, EOS)
- ✅ Dropout, Label Smoothing
- ✅ Early Stopping
- ✅ Data Augmentation
- ✅ Precision (BF16/FP16/FP32)
- ✅ TF32, torch.compile

### Para DETR/DAB-DETR/Deformable DETR:
- ✅ Todos os acima
- ✅ Mesmos hiperparâmetros aplicáveis

---

## 📝 **Hiperparâmetros Não Implementados (e por quê)**

| Hiperparâmetro | Status | Motivo |
|----------------|--------|--------|
| **Auxiliary Loss Weight** | ❌ Não implementado | Controlado internamente pelo modelo |
| **Focal Loss Alpha/Gamma** | ❌ Não implementado | DETR usa Cross-Entropy, não Focal Loss |
| **Temperature Scaling** | ❌ Não implementado | Não aplicável a DETR |
| **NMS Threshold** | ❌ Não implementado | Apenas para inferência, não treinamento |
| **Num Queries** | ❌ Não implementado | Fixo no modelo pré-treinado (100) |

**Conclusão**: Todos os hiperparâmetros **importantes e ajustáveis** para treinamento DETR/RT-DETR estão disponíveis na interface.

---

## 🎯 **Presets Disponíveis**

### 🎯 Preset: Objetos Pequenos (20-60px)
- Image Size: 960px
- Batch Size: 8
- Grad Accum: 2
- Dropout: 0.2
- Label Smoothing: 0.1
- Augmentation: Agressiva
- Loss Weights: Ajustados (Bbox=8.0, GIoU=2.5)

### 📦 Preset: Objetos Grandes (>100px)
- Image Size: 800px
- Batch Size: 12
- Grad Accum: 1
- Dropout: 0.1
- Label Smoothing: 0.0
- Augmentation: Moderada
- Loss Weights: Padrão (Bbox=5.0, GIoU=2.0)

---

## 💡 **Tooltips Implementados**

Todos os campos e botões agora possuem tooltips explicativos que aparecem ao passar o mouse:
- ✅ Modelo
- ✅ Todos os hiperparâmetros
- ✅ Otimizações
- ✅ DataLoader
- ✅ Anti-Overfitting
- ✅ Loss Weights
- ✅ Botões (Presets, Iniciar, Parar)

**Como usar**: Passe o mouse sobre qualquer campo ou botão e aguarde 0.5s para ver a explicação.
