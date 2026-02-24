# Changelog - Requisitos 24/02/2026

## Resumo das Implementações

### 1. eval_threshold 0.5 para métricas coerentes

**Arquivos:** `src/core/session.py`, `src/models/detr_trainer.py`

- **ExperimentConfig:** Novo campo `eval_threshold: float = 0.5` para cálculo de Precision/Recall/IoU no Test.
- **Efeito:** Métricas mais alinhadas ao uso prático (menos FP de baixa confiança).

### 2. Classe ref exibida apenas uma vez por frame (Predição)

**Arquivo:** `src/models/detr_inference.py`

- **Método:** `_enforce_single_ref_per_frame()`
- **Comportamento:** Quando múltiplas detecções da classe `ref` (label 1) ocorrem no mesmo frame, apenas a de maior confiança é mantida.
- **Integração:** Chamado automaticamente em `predict()` após o pós-processamento.
- **Best practice:** NMS por classe para referência única, padrão em inspeção industrial onde há um único objeto de referência por cena.

### 3. NMS para ok e vazio (mitigação de duplicatas)

**Arquivo:** `src/models/detr_inference.py`

- **Método:** `_apply_nms_ok_vazio()` com IoU threshold 0.5.
- **Comportamento:** Non-Maximum Suppression por classe nas classes ok e vazio. Quando múltiplas detecções do mesmo objeto (caixas sobrepostas) ocorrem, mantém apenas a de maior confiança.
- **Integração:** Chamado em `predict()` após `_enforce_single_ref_per_frame`.

### 4. Métricas de predição no treinamento (Test set)

**Arquivos modificados:**
- `src/models/detr_trainer.py` – Loop de treinamento e avaliação
- `src/evaluation/metrics.py` – Módulo de métricas (novo)

**Comportamento:**
- Ao final de cada época, o modelo é avaliado no diretório **Test** do dataset.
- Métricas calculadas: **Precision**, **Recall** e **Mean IoU** (IoU médio das predições matched).
- Protocolo: PASCAL VOC / COCO (IoU threshold 0.5 para matching).
- Log: `Test Epoch N | Precision: X.XXXX | Recall: X.XXXX | IoU: X.XXXX`
- TensorBoard: escalares `Test/precision`, `Test/recall`, `Test/mean_iou`.

**Pré-requisito:** O dataset deve conter o diretório `test/` com `_annotations.coco.json`. Se não existir, as métricas de Test são omitidas (sem erro).

### 5. Testes

**Arquivo:** `test_new_features.py`

- Teste de `_enforce_single_ref_per_frame` (ref única)
- Teste de `_apply_nms_ok_vazio` (NMS ok/vazio)
- Teste de `compute_precision_recall_iou`
- Verificação da estrutura do DETRTrainer
- Verificação da integração no DETRInference

**Execução:** `python test_new_features.py`
