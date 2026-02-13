"""
Utilitários COCO para processamento de datasets e anotações.
Conforme PRD - Sistema de Detecção de Objetos com DETR
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image


def load_coco_annotations(annotations_file: Path) -> Dict:
    """
    Carrega anotações COCO de arquivo JSON.
    
    Args:
        annotations_file: Caminho para arquivo _annotations.coco.json
    
    Returns:
        Dicionário com estrutura COCO
    """
    with open(annotations_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_annotations_file(dataset_dir: Path, split: str = "train") -> Optional[Path]:
    """
    Encontra arquivo de anotações COCO em um diretório de dataset.
    
    Args:
        dataset_dir: Diretório raiz do dataset
        split: Split do dataset (train, valid, test)
    
    Returns:
        Path para arquivo de anotações ou None
    """
    dataset_dir = Path(dataset_dir)
    
    # Possíveis nomes de diretórios
    split_dirs = [split, f"{split}_images", f"{split}ing"]
    if split == "valid":
        split_dirs.extend(["val", "validation", "val_images"])
    
    # Procurar em cada possível diretório
    for split_dir_name in split_dirs:
        split_dir = dataset_dir / split_dir_name
        if split_dir.exists():
            # Procurar arquivo de anotações
            possible_names = [
                "_annotations.coco.json",
                "annotations.json",
                "_annotations.json",
                "instances_default.json"
            ]
            
            for ann_name in possible_names:
                ann_path = split_dir / ann_name
                if ann_path.exists():
                    return ann_path
    
    return None


def create_category_mapping(
    coco_categories: List[Dict],
    system_classes: List[str]
) -> Dict[int, int]:
    """
    Cria mapeamento de IDs de categorias COCO para IDs do sistema.
    
    IMPORTANTE: O dataset do Roboflow pode ter IDs de categoria diferentes
    dos IDs do sistema. Por exemplo:
    - Dataset: {0: 'objects-69mY-ok-b5Hm', 1: 'Ref', 2: 'ok', 3: 'vazio'}
    - Sistema: {0: 'ok', 1: 'ref', 2: 'vazio'}
    
    Este método mapeia PELO NOME da categoria, não pelo ID.
    
    Args:
        coco_categories: Lista de categorias do COCO
        system_classes: Lista de classes do sistema (ex: ["ok", "ref", "vazio"])
    
    Returns:
        Dicionário mapeando category_id (COCO) -> class_id (sistema)
    """
    mapping = {}
    
    # Keywords para cada classe do sistema (case insensitive)
    # Sistema: 0=ok, 1=ref, 2=vazio
    class_keywords = {
        0: ["ok", "good", "normal", "pass", "objects", "b5hm"],  # ok
        1: ["ref", "reference", "defeito", "defect", "nok"],     # ref
        2: ["vazio", "empty", "null", "none", "blank"]           # vazio
    }
    
    print(f"Criando mapeamento de categorias COCO -> Sistema")
    print(f"Categorias no dataset: {len(coco_categories)}")
    print(f"Classes do sistema: {system_classes}")
    
    for cat in coco_categories:
        cat_id = cat["id"]
        cat_name_original = cat["name"]
        cat_name = cat_name_original.lower().strip()
        
        system_id = None
        matched_by = None
        
        # Prioridade 1: Mapeamento direto por nome exato
        if cat_name == "ok":
            system_id = 0
            matched_by = "nome exato"
        elif cat_name == "ref":
            system_id = 1
            matched_by = "nome exato"
        elif cat_name == "vazio":
            system_id = 2
            matched_by = "nome exato"
        else:
            # Prioridade 2: Mapeamento por keywords
            for sys_id, keywords in class_keywords.items():
                if any(kw in cat_name for kw in keywords):
                    system_id = sys_id
                    matched_by = f"keyword '{[kw for kw in keywords if kw in cat_name][0]}'"
                    break
        
        # Prioridade 3: Fallback para ok (ID 0)
        if system_id is None:
            system_id = 0
            matched_by = "fallback (ok)"
            print(f"  AVISO: Categoria '{cat_name_original}' nao reconhecida, usando fallback 'ok'")
        
        mapping[cat_id] = system_id
        print(f"  Dataset ID {cat_id} ('{cat_name_original}') -> Sistema ID {system_id} ({system_classes[system_id]}) [{matched_by}]")
    
    print(f"Mapeamento final: {mapping}")
    return mapping


def create_reverse_category_mapping(
    coco_categories: List[Dict],
    system_classes: List[str]
) -> Dict[int, int]:
    """
    Cria mapeamento reverso: system_id -> dataset_category_id.
    
    Necessário para avaliação COCO, onde predições do modelo
    (com IDs do sistema 0,1,2) precisam ser convertidas para
    IDs do dataset COCO.
    
    Args:
        coco_categories: Lista de categorias do COCO
        system_classes: Lista de classes do sistema
    
    Returns:
        Dicionário mapeando system_id -> dataset_category_id
    """
    forward_mapping = create_category_mapping(coco_categories, system_classes)
    reverse_mapping = {}
    
    # Criar mapeamento reverso (pode haver múltiplos dataset IDs para um system ID)
    for dataset_id, system_id in forward_mapping.items():
        if system_id not in reverse_mapping:
            reverse_mapping[system_id] = dataset_id
        # Se houver múltiplos, usar o primeiro encontrado
    
    return reverse_mapping


def format_annotation_for_detr(
    annotation: Dict,
    image_id: int,
    class_mapping: Dict[int, int]
) -> Dict:
    """
    Formata anotação COCO para formato esperado pelo DetrImageProcessor.
    
    Args:
        annotation: Anotação COCO
        image_id: ID da imagem
        class_mapping: Mapeamento category_id -> system_class_id
    
    Returns:
        Dicionário no formato esperado pelo DetrImageProcessor
    """
    category_id = annotation["category_id"]
    system_class_id = class_mapping.get(category_id)
    
    if system_class_id is None:
        return None
    
    return {
        "image_id": image_id,
        "annotations": [{
            "bbox": annotation["bbox"],  # [x, y, w, h] em pixels
            "category_id": system_class_id,
            "area": annotation.get("area", 0.0),
            "iscrowd": annotation.get("iscrowd", 0)
        }]
    }


def validate_coco_dataset(dataset_dir: Path, split: str = "train") -> Tuple[bool, str]:
    """
    Valida estrutura de dataset COCO.
    
    Args:
        dataset_dir: Diretório raiz do dataset
        split: Split a validar
    
    Returns:
        Tupla (is_valid, error_message)
    """
    dataset_dir = Path(dataset_dir)
    
    # Encontrar arquivo de anotações
    ann_file = find_annotations_file(dataset_dir, split)
    if not ann_file:
        return False, f"Arquivo de anotações não encontrado para split '{split}'"
    
    # Carregar anotações
    try:
        coco_data = load_coco_annotations(ann_file)
    except Exception as e:
        return False, f"Erro ao carregar anotações: {e}"
    
    # Validar estrutura básica
    if "images" not in coco_data:
        return False, "Anotações não contêm campo 'images'"
    
    if "annotations" not in coco_data:
        return False, "Anotações não contêm campo 'annotations'"
    
    if "categories" not in coco_data:
        return False, "Anotações não contêm campo 'categories'"
    
    # Verificar se imagens existem
    split_dir = ann_file.parent
    missing_images = []
    for img_info in coco_data["images"]:
        img_path = split_dir / img_info["file_name"]
        if not img_path.exists():
            missing_images.append(img_info["file_name"])
    
    if missing_images:
        return False, f"{len(missing_images)} imagens não encontradas (ex: {missing_images[0]})"
    
    return True, "Dataset válido"
