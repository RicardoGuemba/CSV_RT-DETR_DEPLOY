"""
Métricas de detecção: Precision, Recall, IoU.
Sistema DETR - RTX 5080 Optimized
"""
from typing import Dict, List, Tuple, Any

import numpy as np


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calcula IoU entre duas bboxes [x1, y1, x2, y2].
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        IoU no intervalo [0, 1]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return float(inter_area / union_area)


def compute_precision_recall_iou(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    iou_threshold: float = 0.5
) -> Tuple[float, float, float]:
    """
    Calcula Precision, Recall e Mean IoU (de predições matched).
    
    Segue o protocolo PASCAL VOC / COCO:
    - TP: predição com IoU >= threshold com GT não matched
    - FP: predição sem match
    - Precision = TP / (TP + FP)
    - Recall = TP / num_ground_truths
    - Mean IoU: média dos IoUs dos matches (TP)
    
    Args:
        predictions: Lista de dicts com image_id, category_id, bbox [x1,y1,x2,y2], score
        ground_truths: Lista de dicts com image_id, category_id, bbox [x1,y1,x2,y2]
        iou_threshold: Threshold de IoU para considerar match
    
    Returns:
        (precision, recall, mean_iou)
    """
    if not ground_truths:
        return 0.0, 0.0, 0.0
    
    if not predictions:
        return 0.0, 0.0, 0.0
    
    # Agrupar GT por imagem
    gt_by_image: Dict[int, List[Dict]] = {}
    for gt in ground_truths:
        img_id = gt["image_id"]
        if img_id not in gt_by_image:
            gt_by_image[img_id] = []
        gt_by_image[img_id].append(gt)
    
    tp = 0
    fp = 0
    gt_matched: Dict[int, bool] = {id(gt): False for gt in ground_truths}
    matched_ious: List[float] = []
    
    # Ordenar predições por score (maior primeiro)
    sorted_preds = sorted(predictions, key=lambda x: x["score"], reverse=True)
    
    for pred in sorted_preds:
        img_id = pred["image_id"]
        pred_box = pred["bbox"]
        pred_cat = pred["category_id"]
        
        img_gts = gt_by_image.get(img_id, [])
        # Filtrar GTs da mesma classe
        cls_gts = [g for g in img_gts if g["category_id"] == pred_cat]
        
        best_iou = 0.0
        best_gt = None
        
        for gt in cls_gts:
            if gt_matched[id(gt)]:
                continue
            iou = compute_iou(pred_box, gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt
        
        if best_iou >= iou_threshold and best_gt is not None:
            tp += 1
            gt_matched[id(best_gt)] = True
            matched_ious.append(best_iou)
        else:
            fp += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / len(ground_truths) if ground_truths else 0.0
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    
    return precision, recall, mean_iou
