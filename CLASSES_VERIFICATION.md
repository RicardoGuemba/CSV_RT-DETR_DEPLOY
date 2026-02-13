# Verificação de Classes - Padrão do Sistema

## ✅ Classes Definidas (Padrão Correto)

O sistema utiliza **3 classes fixas** conforme especificado:

| ID | Nome | Cor RGB | Cor BGR (OpenCV) |
|----|------|---------|------------------|
| 0  | `ok` | (0, 255, 0) - Verde | (0, 255, 0) |
| 1  | `ref` | (255, 165, 0) - Laranja | (0, 165, 255) |
| 2  | `vazio` | (255, 0, 0) - Vermelho | (0, 0, 255) |

## 📍 Locais onde as Classes são Definidas

### 1. `src/utils/class_manager.py` (PRINCIPAL)
```python
DEFAULT_CLASSES = {
    0: {"name": "ok", "color": (0, 255, 0), "color_bgr": (0, 255, 0)},
    1: {"name": "ref", "color": (255, 165, 0), "color_bgr": (0, 165, 255)},
    2: {"name": "vazio", "color": (255, 0, 0), "color_bgr": (0, 0, 255)},
}
```

### 2. `src/core/config.py` (Compatibilidade)
```python
FIXED_CLASSES: List[str] = [
    "ok",
    "ref", 
    "vazio"
]
NUM_CLASSES: int = 3
```

## ✅ Verificação de Consistência

### Módulos que Usam as Classes

1. ✅ **ClassManager** (`src/utils/class_manager.py`)
   - Define `DEFAULT_CLASSES` com 3 classes
   - Mapeamento ID ↔ Nome ↔ Cor

2. ✅ **ModelConfig** (`src/core/config.py`)
   - Define `FIXED_CLASSES` com 3 classes
   - Define `NUM_CLASSES = 3`

3. ✅ **coco_utils.py**
   - Mapeamento de categorias COCO para classes do sistema
   - Keywords para cada classe: ok, ref, vazio

4. ✅ **train_detr.py**
   - Usa `ClassManager` para obter classes
   - Valida mapeamento de classes

5. ✅ **infer_video.py**
   - Usa `ClassManager` para obter cores e nomes

6. ✅ **eval_coco.py**
   - Usa `ClassManager` para mapeamento reverso

7. ✅ **UI (dataset_tab.py)**
   - Exibe classes: "ok, ref, vazio"

## 🔍 Padrão de Mapeamento

### Mapeamento Direto (Prioridade 1)
- `"ok"` → ID 0
- `"ref"` → ID 1
- `"vazio"` → ID 2

### Mapeamento por Keywords (Prioridade 2)
- **ok**: ["ok", "good", "normal", "pass", "objects"]
- **ref**: ["ref", "reference", "defeito", "defect", "nok"]
- **vazio**: ["vazio", "empty", "null", "none", "blank"]

### Fallback (Prioridade 3)
- Se `category_id` estiver em [0, 1, 2]: mapear diretamente
- Caso contrário: default para ID 0 (ok)

## ✅ Validação

Execute para verificar:
```python
from src.utils.class_manager import ClassManager
cm = ClassManager()
print(cm.class_names)  # ['ok', 'ref', 'vazio']
print(cm.num_classes)  # 3
print(cm.get_id2label())  # {0: 'ok', 1: 'ref', 2: 'vazio'}
```

## 📝 Conclusão

**Todas as classes estão definidas corretamente como:**
- **0: ok**
- **1: ref**
- **2: vazio**

**Total: 3 classes**

O padrão está consistente em todos os módulos do sistema.
