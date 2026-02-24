"""
Inferência DETR com otimizações RTX 5080.
Sistema DETR - RTX 5080 Optimized
"""
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable
from collections import deque
import threading

import numpy as np
import torch
from torch.amp import autocast

from PIL import Image
import cv2

from transformers import (
    DetrForObjectDetection,
    DetrImageProcessor,
    AutoModelForObjectDetection,
    AutoImageProcessor
)

from src.core.logger import logger
from src.core.config import ModelConfig, PATHS
from src.core.device_manager import device_manager


class DETRInference:
    """Sistema de inferência DETR otimizado."""
    
    # Cores para visualização (BGR)
    COLORS = [
        (0, 255, 0),    # ok - verde
        (0, 165, 255),  # ref - laranja
        (0, 0, 255),    # vazio - vermelho
    ]
    
    def __init__(
        self,
        model_path: Path,
        threshold: float = 0.05,
        image_size: int = 800,
        use_amp: bool = True
    ):
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.image_size = image_size
        self.use_amp = use_amp and device_manager.gpu_info.is_available
        
        self.model_config = ModelConfig()
        self.device = device_manager.device
        
        self.model = None
        self.processor = None
        
        self._load_model()
    
    def _load_model(self):
        """Carrega modelo treinado."""
        logger.info(f"Carregando modelo de: {self.model_path}")
        
        # Carregar checkpoint
        if self.model_path.is_file():
            checkpoint_path = self.model_path
            exp_dir = self.model_path.parent
        else:
            # É um diretório - procurar best_model.pt
            exp_dir = self.model_path
            checkpoint_path = exp_dir / "best_model.pt"
            if not checkpoint_path.exists():
                pt_files = list(exp_dir.glob("*.pt"))
                if pt_files:
                    checkpoint_path = pt_files[0]
                else:
                    raise FileNotFoundError(f"Nenhum modelo encontrado em {exp_dir}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Recuperar config do experimento
        exp_config = checkpoint.get("experiment_config", {})
        model_id = exp_config.get("model_id", "facebook/detr-resnet-50")
        
        # Carregar arquitetura
        try:
            self.model = AutoModelForObjectDetection.from_pretrained(
                model_id,
                num_labels=self.model_config.NUM_CLASSES,
                ignore_mismatched_sizes=True
            )
            self.processor = AutoImageProcessor.from_pretrained(model_id)
        except Exception:
            self.model = DetrForObjectDetection.from_pretrained(
                "facebook/detr-resnet-50",
                num_labels=self.model_config.NUM_CLASSES,
                ignore_mismatched_sizes=True
            )
            self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        
        # Carregar pesos
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Modelo carregado com sucesso")
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Pré-processa frame para inferência."""
        # BGR para RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize mantendo aspect ratio
        h, w = frame_rgb.shape[:2]
        scale = self.image_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
        
        # Padding
        padded = np.full((self.image_size, self.image_size, 3), 128, dtype=np.uint8)
        padded[:new_h, :new_w] = frame_resized
        
        # Normalização
        img_array = padded.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        
        # Para tensor
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        return img_tensor, scale
    
    @torch.no_grad()
    def predict(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Executa predição em um frame."""
        orig_h, orig_w = frame.shape[:2]
        
        # Pré-processar
        img_tensor, scale = self.preprocess_frame(frame)
        
        # Inferência
        if self.use_amp:
            with autocast('cuda', dtype=torch.float16):
                outputs = self.model(pixel_values=img_tensor)
        else:
            outputs = self.model(pixel_values=img_tensor)
        
        # Pós-processar
        detections = self._postprocess(
            outputs,
            orig_w, orig_h,
            scale
        )
        
        # Garantir que a classe ref seja exibida apenas uma vez por frame
        detections = self._enforce_single_ref_per_frame(detections)
        
        # NMS para ok e vazio: mitiga múltiplas detecções do mesmo objeto
        detections = self._apply_nms_ok_vazio(detections)
        
        return detections
    
    def _postprocess(
        self,
        outputs,
        orig_w: int,
        orig_h: int,
        scale: float
    ) -> List[Dict[str, Any]]:
        """Pós-processa outputs do modelo."""
        # Logits e boxes
        logits = outputs.logits[0]  # [num_queries, num_classes]
        boxes = outputs.pred_boxes[0]  # [num_queries, 4]
        
        # Softmax para probabilidades
        probs = torch.softmax(logits, dim=-1)
        
        # Pegar classe com maior probabilidade (excluindo background se houver)
        scores, labels = probs[..., :-1].max(dim=-1) if probs.shape[-1] > self.model_config.NUM_CLASSES else probs.max(dim=-1)
        
        # Filtrar por threshold
        mask = scores > self.threshold
        
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]
        filtered_labels = labels[mask]
        
        detections = []
        
        for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
            # Converter box de [cx, cy, w, h] normalizado para [x1, y1, x2, y2] pixels
            cx, cy, w, h = box.cpu().numpy()
            
            # Desnormalizar (considerando que usamos image_size)
            cx_px = cx * self.image_size
            cy_px = cy * self.image_size
            w_px = w * self.image_size
            h_px = h * self.image_size
            
            # Converter para coordenadas originais
            x1 = (cx_px - w_px / 2) / scale
            y1 = (cy_px - h_px / 2) / scale
            x2 = (cx_px + w_px / 2) / scale
            y2 = (cy_px + h_px / 2) / scale
            
            # Clampar
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            
            detections.append({
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "score": float(score.cpu()),
                "label": int(label.cpu()),
                "class_name": self.model_config.FIXED_CLASSES[int(label.cpu())]
            })
        
        return detections
    
    def _enforce_single_ref_per_frame(
        self,
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Garante que a classe ref seja exibida apenas uma vez por frame.
        Quando há múltiplas detecções ref, mantém apenas a de maior confiança.
        
        Best practice: NMS por classe para referência única (padrão em inspeção industrial).
        """
        ref_class_id = 1  # ref = índice 1 em FIXED_CLASSES
        ref_detections = [d for d in detections if d["label"] == ref_class_id]
        other_detections = [d for d in detections if d["label"] != ref_class_id]
        
        if len(ref_detections) <= 1:
            return detections
        
        # Manter apenas a detecção ref com maior score
        best_ref = max(ref_detections, key=lambda d: d["score"])
        return other_detections + [best_ref]
    
    def _compute_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calcula IoU entre duas bboxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = a1 + a2 - inter_area
        return inter_area / union if union > 0 else 0.0
    
    def _apply_nms_ok_vazio(
        self,
        detections: List[Dict[str, Any]],
        iou_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Aplica NMS (Non-Maximum Suppression) nas classes ok e vazio.
        Mitiga múltiplas detecções do mesmo objeto mantendo apenas a de maior confiança.
        
        Best practice: NMS por classe, padrão em detecção de objetos.
        """
        ok_class_id = 0
        vazio_class_id = 2
        ref_class_id = 1
        
        ref_dets = [d for d in detections if d["label"] == ref_class_id]
        ok_dets = [d for d in detections if d["label"] == ok_class_id]
        vazio_dets = [d for d in detections if d["label"] == vazio_class_id]
        
        def nms_per_class(dets: List[Dict]) -> List[Dict]:
            if len(dets) <= 1:
                return dets
            sorted_dets = sorted(dets, key=lambda d: d["score"], reverse=True)
            keep = []
            for det in sorted_dets:
                box = det["box"]
                suppressed = False
                for k in keep:
                    if self._compute_iou(box, k["box"]) >= iou_threshold:
                        suppressed = True
                        break
                if not suppressed:
                    keep.append(det)
            return keep
        
        return ref_dets + nms_per_class(ok_dets) + nms_per_class(vazio_dets)
    
    def filter_by_roi(
        self,
        detections: List[Dict[str, Any]],
        frame_width: int,
        frame_height: int,
        roi_x_percent: float = 25.0,
        roi_y_percent: float = 25.0
    ) -> List[Dict[str, Any]]:
        """
        Filtra detecções cujo centróide está dentro da ROI.
        
        Args:
            detections: Lista de detecções
            frame_width: Largura do frame
            frame_height: Altura do frame
            roi_x_percent: Margem horizontal em % (25 = ROI de 25% a 75%)
            roi_y_percent: Margem vertical em % (25 = ROI de 25% a 75%)
        
        Returns:
            Lista de detecções filtradas
        """
        # Calcular limites da ROI
        roi_x1 = int(frame_width * roi_x_percent / 100)
        roi_x2 = int(frame_width * (100 - roi_x_percent) / 100)
        roi_y1 = int(frame_height * roi_y_percent / 100)
        roi_y2 = int(frame_height * (100 - roi_y_percent) / 100)
        
        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            # Calcular centróide
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            # Verificar se centróide está dentro da ROI
            if roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2:
                filtered.append(det)
        
        return filtered
    
    def filter_large_boxes(
        self,
        detections: List[Dict[str, Any]],
        frame_width: int,
        frame_height: int,
        roi_x_percent: float = 25.0,
        roi_y_percent: float = 25.0,
        max_size_percent: float = 25.0
    ) -> List[Dict[str, Any]]:
        """
        Filtra bounding boxes grandes, excluindo aquelas maiores que max_size_percent da ROI.
        
        Args:
            detections: Lista de detecções
            frame_width: Largura do frame
            frame_height: Altura do frame
            roi_x_percent: Margem horizontal da ROI em %
            roi_y_percent: Margem vertical da ROI em %
            max_size_percent: Tamanho máximo permitido como % da ROI (padrão: 25%)
        
        Returns:
            Lista de detecções filtradas (sem boxes grandes)
        """
        # Calcular dimensões da ROI
        roi_width = int(frame_width * (100 - 2 * roi_x_percent) / 100)
        roi_height = int(frame_height * (100 - 2 * roi_y_percent) / 100)
        
        # Calcular tamanho máximo permitido (25% da ROI)
        max_box_width = int(roi_width * max_size_percent / 100)
        max_box_height = int(roi_height * max_size_percent / 100)
        
        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            box_width = x2 - x1
            box_height = y2 - y1
            
            # Excluir boxes maiores que o máximo permitido
            if box_width <= max_box_width and box_height <= max_box_height:
                filtered.append(det)
        
        return filtered
    
    def draw_roi(
        self,
        frame: np.ndarray,
        roi_x_percent: float = 25.0,
        roi_y_percent: float = 25.0,
        color: Tuple[int, int, int] = (255, 255, 0),  # Ciano
        thickness: int = 2
    ) -> np.ndarray:
        """
        Desenha a ROI no frame.
        
        Args:
            frame: Frame de vídeo
            roi_x_percent: Margem horizontal em %
            roi_y_percent: Margem vertical em %
            color: Cor da ROI (BGR)
            thickness: Espessura da linha
        
        Returns:
            Frame com ROI desenhada
        """
        h, w = frame.shape[:2]
        
        # Calcular limites da ROI
        roi_x1 = int(w * roi_x_percent / 100)
        roi_x2 = int(w * (100 - roi_x_percent) / 100)
        roi_y1 = int(h * roi_y_percent / 100)
        roi_y2 = int(h * (100 - roi_y_percent) / 100)
        
        # Desenhar retângulo da ROI
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), color, thickness)
        
        # Adicionar label "ROI" no canto
        cv2.putText(
            frame, "ROI",
            (roi_x1 + 5, roi_y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            color, 2
        )
        
        return frame
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        roi_config: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Desenha detecções no frame com bounding boxes sutis, centroides fixos e confiança.
        
        Cada detecção é representada por:
        - Bounding box com linha sutil (fina)
        - Círculo no centroide (tamanho fixo)
        - Label com classe e confiança
        
        Cores por classe:
        - ok (0): Verde
        - ref (1): Laranja  
        - vazio (2): Vermelho
        
        Args:
            frame: Frame de vídeo
            detections: Lista de detecções
            roi_config: Configuração da ROI (opcional)
                - enabled: bool
                - x_percent: float
                - y_percent: float
        """
        frame_drawn = frame.copy()
        
        # Desenhar ROI se configurada
        if roi_config and roi_config.get("enabled", False):
            self.draw_roi(
                frame_drawn,
                roi_config.get("x_percent", 25.0),
                roi_config.get("y_percent", 25.0)
            )
        
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            score = det["score"]
            label = det["label"]
            class_name = det["class_name"]
            
            # Cor da classe
            color = self.COLORS[label % len(self.COLORS)]
            
            # Calcular centroide da bounding box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            # ========== 1. DESENHAR BOUNDING BOX (LINHA SUTIL) ==========
            # Linha mais sutil: thickness=1 (ao invés de 2 ou 3)
            cv2.rectangle(frame_drawn, (x1, y1), (x2, y2), color, 1)
            
            # ========== 2. DESENHAR CENTRÓIDE (TAMANHO FIXO) ==========
            # Tamanho fixo do centróide (não varia com o tamanho da box)
            radius = 6  # Tamanho fixo de 6 pixels
            
            # Desenhar círculo preenchido no centroide
            cv2.circle(frame_drawn, (cx, cy), radius, color, -1)  # Círculo preenchido
            cv2.circle(frame_drawn, (cx, cy), radius, (255, 255, 255), 1)  # Borda branca sutil
            
            # ========== 3. EXIBIR LABEL COM CLASSE E CONFIANÇA ==========
            # Formatar confiança como porcentagem
            confidence_pct = score * 100
            label_text = f"{class_name} {confidence_pct:.1f}%"
            
            # Posição do texto (acima do centroide)
            text_x = cx - 30  # Centralizar aproximadamente
            text_y = cy - radius - 8  # Acima do círculo
            
            # Ajustar posição se sair da tela
            h, w = frame_drawn.shape[:2]
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            
            # Se texto sair pela esquerda, colocar à direita
            if text_x < 0:
                text_x = cx + radius + 5
            # Se texto sair pela direita, colocar à esquerda
            if text_x + tw > w:
                text_x = x1
            # Se texto sair por cima, colocar abaixo
            if text_y - th < 0:
                text_y = cy + radius + th + 5
            
            # Background sólido do texto (para melhor legibilidade)
            # Usar cor mais escura para contraste
            bg_color = tuple(max(0, c - 50) for c in color)  # Escurecer cor
            cv2.rectangle(
                frame_drawn,
                (text_x - 3, text_y - th - 3),
                (text_x + tw + 3, text_y + 3),
                bg_color,
                -1
            )
            
            # Texto branco
            cv2.putText(
                frame_drawn, label_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1
            )
        
        return frame_drawn
    
    def process_video(
        self,
        video_path: Path,
        show_preview: bool = True,
        fps_limit: int = 30,
        stop_event: Optional[threading.Event] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        frame_callback: Optional[Callable[[np.ndarray], None]] = None
    ) -> Dict[str, Any]:
        """
        Processa vídeo completo.
        
        Args:
            video_path: Caminho do vídeo
            show_preview: Se deve mostrar preview
            fps_limit: Limite de FPS para preview (-1 = usar FPS natural do vídeo)
            stop_event: Evento para parar processamento
            progress_callback: Callback de progresso
            frame_callback: Callback para cada frame processado
        """
        video_path = Path(video_path)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Processando vídeo: {video_path}")
        logger.info(f"  Frames: {total_frames}, FPS original: {video_fps:.1f}")
        
        # Estatísticas
        processed_frames = 0
        total_detections = 0
        fps_history = deque(maxlen=30)
        start_time = time.time()
        
        # Frame rate limiter para preview
        # Se fps_limit == -1, usar FPS natural do vídeo
        if fps_limit == -1:
            effective_fps = video_fps
            logger.info(f"  Preview: Velocidade natural ({video_fps:.1f} FPS)")
        elif fps_limit > 0:
            effective_fps = fps_limit
            logger.info(f"  Preview: Limitado a {fps_limit} FPS")
        else:
            effective_fps = 0  # Sem limite
            logger.info(f"  Preview: Sem limite de FPS")
        
        frame_interval = 1.0 / effective_fps if effective_fps > 0 else 0
        last_frame_time = time.time()
        
        results = {
            "video_path": str(video_path),
            "total_frames": total_frames,
            "video_fps": video_fps,
            "detections_per_frame": []
        }
        
        while cap.isOpened():
            if stop_event and stop_event.is_set():
                break
            
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start = time.time()
            
            # Inferência
            detections = self.predict(frame)
            total_detections += len(detections)
            
            # Desenhar
            frame_drawn = self.draw_detections(frame, detections)
            
            # FPS
            frame_time = time.time() - frame_start
            current_fps = 1.0 / frame_time if frame_time > 0 else 0
            fps_history.append(current_fps)
            avg_fps = sum(fps_history) / len(fps_history)
            
            processed_frames += 1
            
            # Guardar resultados
            results["detections_per_frame"].append({
                "frame": processed_frames,
                "detections": detections,
                "fps": current_fps
            })
            
            # Callbacks
            if progress_callback:
                progress_callback(processed_frames, total_frames, avg_fps)
            
            # Preview com sincronização de tempo (velocidade natural do vídeo)
            if show_preview and frame_callback:
                current_time = time.time()
                elapsed = current_time - last_frame_time
                
                # Se frame_interval > 0, aguardar para manter velocidade natural
                if frame_interval > 0 and elapsed < frame_interval:
                    wait_time = frame_interval - elapsed
                    time.sleep(wait_time)
                
                # Exibir frame
                frame_callback(frame_drawn)
                last_frame_time = time.time()
        
        cap.release()
        
        total_time = time.time() - start_time
        
        results.update({
            "processed_frames": processed_frames,
            "total_detections": total_detections,
            "total_time": total_time,
            "avg_fps": processed_frames / total_time if total_time > 0 else 0
        })
        
        logger.info(f"Vídeo processado: {processed_frames} frames em {total_time:.1f}s")
        logger.info(f"  FPS médio: {results['avg_fps']:.1f}")
        logger.info(f"  Total detecções: {total_detections}")
        
        return results
    
    def export_video(
        self,
        video_path: Path,
        output_path: Path,
        progress_callback: Optional[Callable[[int, int, float], None]] = None
    ):
        """Exporta vídeo com anotações."""
        video_path = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        logger.info(f"Exportando vídeo para: {output_path}")
        
        processed = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Inferência e desenho
            detections = self.predict(frame)
            frame_drawn = self.draw_detections(frame, detections)
            
            out.write(frame_drawn)
            processed += 1
            
            if progress_callback and processed % 10 == 0:
                elapsed = time.time() - start_time
                fps_export = processed / elapsed if elapsed > 0 else 0
                progress_callback(processed, total_frames, fps_export)
        
        cap.release()
        out.release()
        
        logger.info(f"Vídeo exportado: {output_path}")
