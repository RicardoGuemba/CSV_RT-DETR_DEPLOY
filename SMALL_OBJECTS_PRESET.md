# Preset de Hiperparâmetros para Objetos Retangulares Pequenos (20-60px)

## 📋 Resumo

Configuração otimizada para detectar objetos retangulares pequenos (20-60px) com técnicas avançadas de anti-overfitting, otimizada para RTX 5080 (16GB VRAM).

## 🎯 Hiperparâmetros Otimizados

### Treinamento Básico
- **Épocas**: 100 (mais épocas para objetos pequenos)
- **Batch Size**: 8 (batch efetivo: 16 com grad_accum=2)
- **Gradient Accumulation**: 2
- **Image Size**: 960px (resolução otimizada para objetos 20-60px)
- **Learning Rate**: 5e-6 (LR conservador para treinamento estável)

### Otimização
- **Weight Decay**: 5e-4 (aumentado para regularização)
- **Warmup Steps**: 1000 (warmup mais longo)
- **Max Grad Norm**: 0.5 (gradient clipping mais agressivo)
- **Scheduler**: Cosine

### Mixed Precision
- **Precision**: BF16 (otimizado para RTX 5080)
- **TF32**: Habilitado
- **torch.compile**: Desabilitado (Windows não suporta Triton)

### DataLoader
- **Workers**: 10 (mais workers para I/O)
- **Pin Memory**: True
- **Persistent Workers**: True
- **Prefetch Factor**: 3 (mais prefetch)

## 🛡️ Técnicas Anti-Overfitting

### 1. Dropout Aumentado
- **Dropout**: 0.2 (aumentado de 0.1 padrão)
- Ajuda a prevenir memorização dos dados de treino

### 2. Early Stopping Agressivo
- **Patience**: 7 épocas (reduzido de 10)
- Para o treinamento mais cedo se não houver melhoria

### 3. Label Smoothing
- **Label Smoothing**: 0.1
- Suaviza os rótulos para reduzir overconfidence

### 4. Data Augmentation Agressiva
- **Horizontal Flip**: 50% probabilidade
- **Color Jitter**: 60% probabilidade (brilho, contraste, saturação, hue)
- **Gaussian Blur**: 30% probabilidade (blur leve)
- **Cutout**: 30% probabilidade (tamanho 20-60px, similar aos objetos)

### 5. Weight Decay Aumentado
- **Weight Decay**: 5e-4 (5x maior que padrão)
- Regularização L2 mais forte

## 📊 Loss Weights Ajustados

Para melhor detecção de objetos retangulares 20-60px:
- **BBox Loss Coefficient**: 8.0 (aumentado de 5.0)
- **GIoU Loss Coefficient**: 2.5 (aumentado de 2.0)
- **Class Loss Coefficient**: 1.0
- **EOS Coefficient**: 0.1

## 🚀 Como Usar

### Na Interface Gráfica

1. Abra a aba **Treinamento**
2. Clique no botão **"🎯 Aplicar Preset: Objetos Retangulares (20-60px)"**
3. Todos os hiperparâmetros serão configurados automaticamente
4. Inicie o treinamento normalmente

### Programaticamente

```python
from src.core.config import SmallObjectTrainingDefaults

preset = SmallObjectTrainingDefaults()
# Use os valores do preset para configurar seu treinamento
```

## ✅ Validação

Todos os testes passaram:
- ✓ Preset criado corretamente
- ✓ Augmentation funcionando
- ✓ Técnicas anti-overfitting ativas
- ✓ Otimizações para objetos pequenos aplicadas

## 📝 Notas Importantes

1. **VRAM**: Com image_size=960px e batch_size=8, o uso de VRAM será ~10-12GB (dentro dos 16GB da RTX 5080)

2. **Tempo de Treinamento**: Esperado ~30-40 minutos por época (dependendo do dataset)

3. **Monitoramento**: Acompanhe a diferença entre train_loss e val_loss para detectar overfitting

4. **Early Stopping**: O treinamento pode parar após 7 épocas sem melhoria na validação

## 🔧 Ajustes Recomendados

Se ainda houver overfitting:
- Aumentar `weight_decay` para 1e-3
- Aumentar `dropout` para 0.3
- Reduzir `early_stopping_patience` para 5

Se não houver overfitting mas precisar de mais performance:
- Reduzir `weight_decay` para 1e-4
- Reduzir `dropout` para 0.15
- Aumentar `early_stopping_patience` para 10
