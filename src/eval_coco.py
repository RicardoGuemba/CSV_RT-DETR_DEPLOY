"""
Módulo de avaliação COCO.
Conforme PRD - Sistema de Detecção de Objetos com DETR
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
from torch.amp import autocast
from PIL import Image
import cv2

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    print("AVISO: pycocotools não disponível. Métricas COCO completas não estarão disponíveis.")

from src.utils.device_manager import DeviceManager
from src.utils.class_manager import ClassManager
from src.utils.paths import PathManager
from src.hf_model_utils import load_detr_model_for_inference
from src.coco_utils import (
    load_coco_annotations,
    find_annotations_file,
    create_reverse_category_mapping
)


@dataclass
class EvaluationConfig:
    """Configuração de avaliação conforme PRD."""
    model_path: str = ""
    dataset_path: str = ""
    split: str = "test"  # train, valid, test
    batch_size: int = 1
    confidence_threshold: float = 0.0  # 0 para avaliação completa


class COCOEvaluator:
    """Avaliador COCO conforme PRD."""
    
    def __init__(
        self,
        config: EvaluationConfig,
        class_manager: Optional[ClassManager] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ):
        """
        Args:
            config: Configuração de avaliação
            class_manager: Gerenciador de classes
            progress_callback: Callback para progresso
        """
        self.config = config
        self.class_manager = class_manager or ClassManager()
        self.progress_callback = progress_callback
        
        # Utilitários
        self.device_manager = DeviceManager()
        self.path_manager = PathManager()
        
        # Componentes
        self.model = None
        self.image_processor = None
        self.id2label = None
        
        # Dataset
        self.coco_gt = None
        self.reverse_mapping = None  # system_id -> dataset_category_id
    
    def _create_reverse_category_mapping(self, coco_gt) -> Dict[int, int]:
        """
        Mapeia IDs do sistema para IDs do dataset COCO.
        
        Necessário porque modelo prediz IDs do sistema (0,1,2)
        mas COCO espera IDs do dataset (1,2,3).
        """
        categories = coco_gt.loadCats(coco_gt.getCatIds())
        system_classes = self.class_manager.class_names
        
        forward_mapping = create_reverse_category_mapping(categories, system_classes)
        
        # Criar reverso: system_id -> dataset_id
        reverse = {}
        for dataset_id, system_id in forward_mapping.items():
            if system_id not in reverse:
                reverse[system_id] = dataset_id
        
        return reverse
    
    def setup(self) -> None:
        """
        Configura avaliador.
        Conforme PRD:
        - Carregar modelo
        - Carregar ground truth COCO
        - Criar mapeamento reverso
        """
        device = self.device_manager.get_device()
        
        # Carregar modelo
        self.model, self.image_processor, model_id2label = load_detr_model_for_inference(
            model_path=self.config.model_path,
            device=device
        )
        
        self.id2label = model_id2label
        self.model.eval()
        
        # Carregar ground truth COCO
        dataset_path = Path(self.config.dataset_path)
        ann_file = find_annotations_file(dataset_path, self.config.split)
        
        if not ann_file:
            raise FileNotFoundError(f"Arquivo de anotações não encontrado para split '{self.config.split}'")
        
        if PYCOCOTOOLS_AVAILABLE:
            self.coco_gt = COCO(str(ann_file))
        else:
            # Fallback: carregar JSON diretamente
            self.coco_gt = load_coco_annotations(ann_file)
        
        # Criar mapeamento reverso
        if PYCOCOTOOLS_AVAILABLE:
            self.reverse_mapping = self._create_reverse_category_mapping(self.coco_gt)
        else:
            categories = self.coco_gt.get("categories", [])
            system_classes = self.class_manager.class_names
            forward_mapping = create_reverse_category_mapping(categories, system_classes)
            self.reverse_mapping = {v: k for k, v in forward_mapping.items()}
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Executa avaliação COCO completa.
        Conforme PRD:
        1. Carregar ground truth COCO
        2. Criar mapeamento reverso
        3. Fazer predições em todas as imagens
        4. Converter predições para formato COCO
        5. Calcular métricas com pycocotools
        6. Calcular métricas por classe
        7. Salvar resultados
        """
        if not PYCOCOTOOLS_AVAILABLE:
            return self._evaluate_fallback()
        
        # Carregar imagens
        img_ids = self.coco_gt.getImgIds()
        images = self.coco_gt.loadImgs(img_ids)
        
        # Coletar predições
        coco_predictions = []
        
        dataset_path = Path(self.config.dataset_path)
        split_dir = find_annotations_file(dataset_path, self.config.split).parent
        
        total = len(images)
        for idx, img_info in enumerate(images):
            if self.progress_callback:
                progress = (idx / total) * 100
                self.progress_callback(progress)
            
            img_id = img_info["id"]
            img_path = split_dir / img_info["file_name"]
            
            if not img_path.exists():
                continue
            
            # Carregar e processar imagem
            image = Image.open(img_path).convert("RGB")
            inputs = self.image_processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device_manager.get_device())
            
            # Predição
            with torch.no_grad():
                if self.device_manager.cuda_available:
                    with autocast(device_type='cuda', dtype=torch.float16):
                        outputs = self.model(pixel_values=pixel_values)
                else:
                    outputs = self.model(pixel_values=pixel_values)
            
            # Pós-processamento
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.image_processor.post_process_object_detection(
                outputs,
                threshold=self.config.confidence_threshold,
                target_sizes=target_sizes
            )[0]
            
            # Converter para formato COCO
            for score, label, box in zip(
                results["scores"],
                results["labels"],
                results["boxes"]
            ):
                system_class_id = label.item()
                dataset_category_id = self.reverse_mapping.get(system_class_id, system_class_id + 1)
                
                # Converter box de [x1, y1, x2, y2] para [x, y, w, h]
                x1, y1, x2, y2 = box.tolist()
                w = x2 - x1
                h = y2 - y1
                
                coco_predictions.append({
                    "image_id": img_id,
                    "category_id": dataset_category_id,
                    "bbox": [x1, y1, w, h],
                    "score": score.item()
                })
        
        # Calcular métricas com pycocotools
        coco_dt = self.coco_gt.loadRes(coco_predictions)
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extrair métricas
        metrics = {
            "mAP": coco_eval.stats[0],  # mAP@0.50:0.95
            "AP50": coco_eval.stats[1],  # AP@0.50
            "AP75": coco_eval.stats[2],  # AP@0.75
            "AP_small": coco_eval.stats[3],
            "AP_medium": coco_eval.stats[4],
            "AP_large": coco_eval.stats[5],
            "AR1": coco_eval.stats[6],
            "AR10": coco_eval.stats[7],
            "AR100": coco_eval.stats[8],
            "AR_small": coco_eval.stats[9],
            "AR_medium": coco_eval.stats[10],
            "AR_large": coco_eval.stats[11]
        }
        
        # Métricas por classe
        per_class_metrics = self._calculate_per_class_metrics(coco_eval)
        
        # Salvar resultados
        results = {
            "summary": metrics,
            "per_class": per_class_metrics,
            "evaluation_info": {
                "model_path": str(self.config.model_path),
                "dataset_path": str(self.config.dataset_path),
                "split": self.config.split,
                "num_images": len(images),
                "num_predictions": len(coco_predictions),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return results
    
    def _calculate_per_class_metrics(self, coco_eval: COCOeval) -> Dict[str, Dict]:
        """Calcula métricas por classe."""
        per_class = {}
        
        for cat_id in coco_eval.params.catIds:
            coco_eval.params.catIds = [cat_id]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            cat_info = self.coco_gt.loadCats([cat_id])[0]
            class_name = cat_info["name"]
            
            per_class[class_name] = {
                "AP": coco_eval.stats[0],
                "AP50": coco_eval.stats[1],
                "AP75": coco_eval.stats[2]
            }
        
        return per_class
    
    def _evaluate_fallback(self) -> Dict[str, Any]:
        """Avaliação simplificada sem pycocotools."""
        print("Usando avaliação simplificada (pycocotools não disponível)")
        
        # Implementação básica sem pycocotools
        # (pode ser expandida conforme necessário)
        
        return {
            "summary": {
                "mAP": 0.0,
                "AP50": 0.0,
                "AP75": 0.0
            },
            "per_class": {},
            "evaluation_info": {
                "note": "Avaliação simplificada - instale pycocotools para métricas completas"
            }
        }
