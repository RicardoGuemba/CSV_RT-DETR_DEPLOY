"""
Gerenciador de classes de detecção.
Gerencia mapeamento ID ↔ Nome ↔ Cor e validação de anotações.
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# Classes padrão do sistema
DEFAULT_CLASSES = {
    0: {"name": "ok", "color": (0, 255, 0), "color_bgr": (0, 255, 0)},
    1: {"name": "ref", "color": (255, 165, 0), "color_bgr": (0, 165, 255)},
    2: {"name": "vazio", "color": (255, 0, 0), "color_bgr": (0, 0, 255)},
}


class ClassManager:
    """Gerencia classes de detecção e mapeamentos."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: Caminho opcional para arquivo JSON com configuração de classes
        """
        self.classes = DEFAULT_CLASSES.copy()
        
        if config_path and Path(config_path).exists():
            self._load_config(config_path)
        
        # Criar mapeamentos auxiliares
        self._id2name = {cid: info["name"] for cid, info in self.classes.items()}
        self._name2id = {info["name"]: cid for cid, info in self.classes.items()}
    
    def _load_config(self, config_path: str):
        """Carrega configuração de classes de arquivo JSON."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Normalizar formato
            for cid_str, info in config.items():
                cid = int(cid_str)
                self.classes[cid] = {
                    "name": info.get("name", f"class_{cid}"),
                    "color": tuple(info.get("color", [0, 255, 0])),
                    "color_bgr": tuple(info.get("color_bgr", info.get("color", [0, 255, 0])))
                }
        except Exception as e:
            print(f"Erro ao carregar configuração de classes: {e}")
    
    @property
    def num_classes(self) -> int:
        """Retorna número de classes."""
        return len(self.classes)
    
    @property
    def class_names(self) -> List[str]:
        """Retorna lista de nomes de classes."""
        return [self.classes[i]["name"] for i in sorted(self.classes.keys())]
    
    def get_class_name(self, class_id: int) -> str:
        """Retorna nome da classe dado o ID."""
        return self.classes.get(class_id, {}).get("name", f"unknown_{class_id}")
    
    def get_class_color(self, class_id: int, bgr: bool = False) -> Tuple[int, int, int]:
        """
        Retorna cor da classe.
        
        Args:
            class_id: ID da classe
            bgr: Se True, retorna em formato BGR (OpenCV), senão RGB
        
        Returns:
            Tupla (R, G, B) ou (B, G, R) dependendo de bgr
        """
        info = self.classes.get(class_id, {})
        if bgr:
            return info.get("color_bgr", (0, 0, 255))
        return info.get("color", (0, 255, 0))
    
    def get_id2label(self) -> Dict[int, str]:
        """Retorna dicionário ID -> Label."""
        return self._id2name.copy()
    
    def get_label2id(self) -> Dict[str, int]:
        """Retorna dicionário Label -> ID."""
        return self._name2id.copy()
    
    def validate_annotations(self, annotations: List[Dict]) -> Tuple[bool, str]:
        """
        Valida anotações COCO.
        
        Args:
            annotations: Lista de anotações no formato COCO
        
        Returns:
            Tupla (is_valid, error_message)
        """
        if not annotations:
            return True, ""
        
        for ann in annotations:
            # Verificar campos obrigatórios
            if "category_id" not in ann:
                return False, "Anotação sem 'category_id'"
            
            if "bbox" not in ann:
                return False, "Anotação sem 'bbox'"
            
            # Verificar se category_id está no range válido
            cat_id = ann["category_id"]
            if cat_id not in self.classes:
                return False, f"category_id {cat_id} não está nas classes do sistema"
            
            # Verificar formato de bbox [x, y, w, h]
            bbox = ann["bbox"]
            if not isinstance(bbox, list) or len(bbox) != 4:
                return False, f"bbox deve ser lista com 4 elementos [x, y, w, h], recebido: {bbox}"
            
            # Verificar valores não negativos
            if any(x < 0 for x in bbox):
                return False, f"bbox contém valores negativos: {bbox}"
        
        return True, ""
    
    def get_statistics(self, annotations: List[Dict]) -> Dict[str, int]:
        """
        Retorna estatísticas das anotações por classe.
        
        Args:
            annotations: Lista de anotações COCO
        
        Returns:
            Dicionário com contagem por classe
        """
        stats = {cid: 0 for cid in self.classes.keys()}
        
        for ann in annotations:
            cat_id = ann.get("category_id")
            if cat_id in stats:
                stats[cat_id] += 1
        
        return stats
