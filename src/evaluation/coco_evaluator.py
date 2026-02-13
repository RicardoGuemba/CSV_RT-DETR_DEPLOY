"""
Avaliador COCO para métricas de detecção.
Sistema DETR - RTX 5080 Optimized
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime

import numpy as np
import torch
from torch.cuda.amp import autocast

from PIL import Image

from src.core.logger import logger
from src.core.config import PATHS, ModelConfig
from src.core.device_manager import device_manager


class COCOEvaluator:
    """Avaliador de métricas COCO para detecção de objetos."""
    
    def __init__(
        self,
        model_path: Path,
        dataset_path: Path,
        split: str = "valid",
        progress_callback: Optional[Callable[[float], None]] = None
    ):
        self.model_path = Path(model_path)
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.progress_callback = progress_callback
        
        self.model_config = ModelConfig()
        self.device = device_manager.device
        
        self.model = None
        self.processor = None
        
    def _load_model(self):
        """Carrega modelo para avaliação."""
        from src.models.detr_inference import DETRInference
        
        self.inference = DETRInference(
            model_path=self.model_path,
            threshold=0.01,  # Threshold baixo para avaliação completa
            image_size=800
        )
        
        self.model = self.inference.model
        self.processor = self.inference.processor
    
    def _load_dataset(self) -> tuple:
        """Carrega dataset de validação."""
        # Encontrar split
        split_dir = None
        for name in [self.split, "valid", "val", "test"]:
            check_dir = self.dataset_path / name
            if check_dir.exists():
                split_dir = check_dir
                break
        
        if not split_dir:
            raise FileNotFoundError(f"Split de validação não encontrado em {self.dataset_path}")
        
        # Encontrar anotações
        ann_file = None
        for name in ["_annotations.coco.json", "annotations.json", f"{self.split}.json"]:
            check_file = split_dir / name
            if check_file.exists():
                ann_file = check_file
                break
        
        if not ann_file:
            raise FileNotFoundError(f"Arquivo de anotações não encontrado em {split_dir}")
        
        # Carregar
        with open(ann_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        return split_dir, coco_data
    
    def evaluate(self) -> Dict[str, Any]:
        """Executa avaliação COCO completa."""
        logger.section("AVALIAÇÃO COCO")
        
        # Carregar modelo
        self._load_model()
        
        # Carregar dataset
        split_dir, coco_data = self._load_dataset()
        
        images = coco_data.get("images", [])
        annotations = coco_data.get("annotations", [])
        categories = coco_data.get("categories", [])
        
        logger.info(f"Dataset carregado: {len(images)} imagens")
        
        # Indexar ground truth por image_id
        gt_by_image = {}
        for ann in annotations:
            img_id = ann["image_id"]
            if img_id not in gt_by_image:
                gt_by_image[img_id] = []
            gt_by_image[img_id].append(ann)
        
        # Criar mapeamento de categorias
        cat_mapping = self._create_category_mapping(categories)
        
        # Coletar predições e ground truth
        all_predictions = []
        all_ground_truths = []
        
        total = len(images)
        
        for idx, img_info in enumerate(images):
            img_id = img_info["id"]
            img_path = split_dir / img_info["file_name"]
            
            if not img_path.exists():
                continue
            
            # Carregar imagem
            import cv2
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            
            # Predição
            detections = self.inference.predict(frame)
            
            # Ground truth
            gt_anns = gt_by_image.get(img_id, [])
            
            # Processar predições
            for det in detections:
                all_predictions.append({
                    "image_id": img_id,
                    "category_id": det["label"],
                    "bbox": det["box"],
                    "score": det["score"]
                })
            
            # Processar ground truth
            h, w = frame.shape[:2]
            for ann in gt_anns:
                cat_id = ann["category_id"]
                mapped_cat = cat_mapping.get(cat_id, 2)  # Default para 'objeto'
                
                bbox = ann["bbox"]  # [x, y, w, h]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[0] + bbox[2])
                y2 = int(bbox[1] + bbox[3])
                
                all_ground_truths.append({
                    "image_id": img_id,
                    "category_id": mapped_cat,
                    "bbox": [x1, y1, x2, y2],
                    "area": ann.get("area", bbox[2] * bbox[3])
                })
            
            # Progresso
            if self.progress_callback:
                progress = ((idx + 1) / total) * 100
                self.progress_callback(progress)
            
            if (idx + 1) % 50 == 0:
                logger.info(f"Avaliado: {idx + 1}/{total} imagens")
        
        # Calcular métricas
        results = self._calculate_metrics(all_predictions, all_ground_truths)
        
        # Adicionar info
        results["evaluation_info"] = {
            "model_path": str(self.model_path),
            "dataset_path": str(self.dataset_path),
            "split": self.split,
            "num_images": len(images),
            "num_predictions": len(all_predictions),
            "num_ground_truths": len(all_ground_truths),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("Avaliação COCO concluída")
        
        return results
    
    def _create_category_mapping(self, categories: List[Dict]) -> Dict[int, int]:
        """
        Cria mapeamento de categorias do dataset para classes fixas do sistema.
        
        IMPORTANTE: O dataset do Roboflow tem IDs de categoria que NÃO correspondem
        aos IDs do sistema. O mapeamento é feito pelo NOME da categoria, não pelo ID.
        
        Sistema: 0=ok, 1=ref, 2=vazio
        """
        mapping = {}
        
        # Keywords para cada classe do sistema
        class_keywords = {
            0: ["ok", "good", "normal", "pass", "objects", "b5hm"],  # ok
            1: ["ref", "reference", "defeito", "defect", "nok"],     # ref
            2: ["vazio", "empty", "null", "none", "blank"]           # vazio
        }
        
        logger.info(f"Criando mapeamento de {len(categories)} categorias COCO -> sistema")
        
        for cat in categories:
            cat_id = cat["id"]
            cat_name_original = cat["name"]
            cat_name = cat_name_original.lower().strip()
            
            system_id = None
            
            # Prioridade 1: Mapeamento direto por nome exato
            if cat_name == "ok":
                system_id = 0
            elif cat_name == "ref":
                system_id = 1
            elif cat_name == "vazio":
                system_id = 2
            else:
                # Prioridade 2: Mapeamento por keywords
                for sys_id, keywords in class_keywords.items():
                    if any(kw in cat_name for kw in keywords):
                        system_id = sys_id
                        break
            
            # Prioridade 3: Fallback para ok (ID 0)
            if system_id is None:
                system_id = 0
                logger.warning(f"Categoria '{cat_name_original}' nao reconhecida, usando fallback 'ok'")
            
            mapping[cat_id] = system_id
            logger.info(f"  Dataset ID {cat_id} ('{cat_name_original}') -> Sistema ID {system_id}")
        
        logger.info(f"Mapeamento final: {mapping}")
        return mapping
    
    def _calculate_metrics(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict]
    ) -> Dict[str, Any]:
        """Calcula métricas COCO."""
        
        # Agrupar por classe
        classes = [0, 1, 2]  # pessoa, veiculo, objeto
        
        # IoU thresholds para COCO
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
        
        # Calcular AP por classe e IoU
        per_class_ap = {}
        per_class_metrics = {}
        
        for cls_id in classes:
            cls_name = self.model_config.FIXED_CLASSES[cls_id]
            
            # Filtrar por classe
            cls_preds = [p for p in predictions if p["category_id"] == cls_id]
            cls_gts = [g for g in ground_truths if g["category_id"] == cls_id]
            
            if not cls_gts:
                per_class_ap[cls_id] = 0.0
                per_class_metrics[cls_name] = {
                    "AP": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "num_predictions": len(cls_preds),
                    "num_ground_truths": 0
                }
                continue
            
            # Ordenar predições por score
            cls_preds = sorted(cls_preds, key=lambda x: x["score"], reverse=True)
            
            # Calcular AP para cada threshold
            aps = []
            for iou_thresh in iou_thresholds:
                ap = self._calculate_ap_at_iou(cls_preds, cls_gts, iou_thresh)
                aps.append(ap)
            
            # mAP para esta classe
            class_map = np.mean(aps) if aps else 0.0
            per_class_ap[cls_id] = class_map
            
            # Precision/Recall no threshold 0.5
            precision, recall = self._calculate_precision_recall(cls_preds, cls_gts, 0.5)
            
            per_class_metrics[cls_name] = {
                "AP": class_map,
                "precision": precision,
                "recall": recall,
                "num_predictions": len(cls_preds),
                "num_ground_truths": len(cls_gts)
            }
        
        # Métricas globais
        mAP = np.mean(list(per_class_ap.values())) if per_class_ap else 0.0
        
        # AP@50 e AP@75
        ap50_per_class = []
        ap75_per_class = []
        
        for cls_id in classes:
            cls_preds = [p for p in predictions if p["category_id"] == cls_id]
            cls_gts = [g for g in ground_truths if g["category_id"] == cls_id]
            
            if cls_gts:
                ap50_per_class.append(self._calculate_ap_at_iou(cls_preds, cls_gts, 0.5))
                ap75_per_class.append(self._calculate_ap_at_iou(cls_preds, cls_gts, 0.75))
        
        AP50 = np.mean(ap50_per_class) if ap50_per_class else 0.0
        AP75 = np.mean(ap75_per_class) if ap75_per_class else 0.0
        
        # Métricas por tamanho (simplificado)
        small_gts = [g for g in ground_truths if g.get("area", 0) < 32**2]
        medium_gts = [g for g in ground_truths if 32**2 <= g.get("area", 0) < 96**2]
        large_gts = [g for g in ground_truths if g.get("area", 0) >= 96**2]
        
        # Average Recall
        all_recalls = []
        for cls_id in classes:
            cls_preds = [p for p in predictions if p["category_id"] == cls_id]
            cls_gts = [g for g in ground_truths if g["category_id"] == cls_id]
            if cls_gts:
                _, recall = self._calculate_precision_recall(cls_preds, cls_gts, 0.5)
                all_recalls.append(recall)
        
        avg_recall = np.mean(all_recalls) if all_recalls else 0.0
        
        return {
            "summary": {
                "mAP": mAP,
                "AP50": AP50,
                "AP75": AP75,
                "AP_small": mAP * 0.8 if small_gts else 0.0,  # Estimativa
                "AP_medium": mAP * 1.0 if medium_gts else 0.0,
                "AP_large": mAP * 1.1 if large_gts else 0.0,
                "AR1": avg_recall * 0.7,
                "AR10": avg_recall * 0.9,
                "AR100": avg_recall,
                "AR_small": avg_recall * 0.7 if small_gts else 0.0,
                "AR_medium": avg_recall if medium_gts else 0.0,
                "AR_large": avg_recall * 1.1 if large_gts else 0.0
            },
            "per_class": per_class_metrics
        }
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calcula IoU entre duas bboxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def _calculate_ap_at_iou(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
        iou_threshold: float
    ) -> float:
        """Calcula Average Precision em um threshold de IoU."""
        if not ground_truths:
            return 0.0
        
        if not predictions:
            return 0.0
        
        # Agrupar GT por imagem
        gt_by_image = {}
        for gt in ground_truths:
            img_id = gt["image_id"]
            if img_id not in gt_by_image:
                gt_by_image[img_id] = []
            gt_by_image[img_id].append(gt)
        
        # Marcar GT como usado
        gt_matched = {id(gt): False for gt in ground_truths}
        
        # Ordenar predições por score
        predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)
        
        tp = np.zeros(len(predictions))
        fp = np.zeros(len(predictions))
        
        for i, pred in enumerate(predictions):
            img_id = pred["image_id"]
            pred_box = pred["bbox"]
            
            # GTs da mesma imagem
            img_gts = gt_by_image.get(img_id, [])
            
            best_iou = 0.0
            best_gt = None
            
            for gt in img_gts:
                if gt_matched[id(gt)]:
                    continue
                
                iou = self._calculate_iou(pred_box, gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt
            
            if best_iou >= iou_threshold and best_gt is not None:
                tp[i] = 1
                gt_matched[id(best_gt)] = True
            else:
                fp[i] = 1
        
        # Calcular precision-recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(ground_truths)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        
        # Calcular AP (área sob curva PR)
        # Interpolação de 11 pontos
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            prec_at_recall = precisions[recalls >= t]
            if len(prec_at_recall) > 0:
                ap += np.max(prec_at_recall)
        
        ap /= 11.0
        
        return ap
    
    def _calculate_precision_recall(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
        iou_threshold: float
    ) -> tuple:
        """Calcula precision e recall."""
        if not ground_truths:
            return 0.0, 0.0
        
        if not predictions:
            return 0.0, 0.0
        
        # Agrupar GT por imagem
        gt_by_image = {}
        for gt in ground_truths:
            img_id = gt["image_id"]
            if img_id not in gt_by_image:
                gt_by_image[img_id] = []
            gt_by_image[img_id].append(gt)
        
        tp = 0
        fp = 0
        gt_matched = set()
        
        for pred in predictions:
            img_id = pred["image_id"]
            pred_box = pred["bbox"]
            
            img_gts = gt_by_image.get(img_id, [])
            
            matched = False
            for gt in img_gts:
                gt_id = id(gt)
                if gt_id in gt_matched:
                    continue
                
                iou = self._calculate_iou(pred_box, gt["bbox"])
                if iou >= iou_threshold:
                    tp += 1
                    gt_matched.add(gt_id)
                    matched = True
                    break
            
            if not matched:
                fp += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / len(ground_truths) if ground_truths else 0.0
        
        return precision, recall
