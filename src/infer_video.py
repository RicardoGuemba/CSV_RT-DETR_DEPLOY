"""
Módulo de inferência em vídeo.
Conforme PRD - Sistema de Detecção de Objetos com DETR
"""
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import json

import numpy as np
import torch
from torch.amp import autocast
from PIL import Image
import cv2

from src.utils.device_manager import DeviceManager
from src.utils.class_manager import ClassManager
from src.utils.paths import PathManager
from src.hf_model_utils import load_detr_model_for_inference


@dataclass
class InferenceConfig:
    """Configuração de inferência conforme PRD."""
    model_path: str = ""
    video_path: str = ""
    confidence_threshold: float = 0.5
    output_path: Optional[str] = None
    show_preview: bool = False
    save_video: bool = True


@dataclass
class DetectionResult:
    """Resultado de uma detecção."""
    bbox: List[float]  # [x1, y1, x2, y2]
    score: float
    class_id: int
    class_name: str
    color: Tuple[int, int, int]  # BGR para OpenCV


class VideoInference:
    """Sistema de inferência em vídeo conforme PRD."""
    
    def __init__(self, config: InferenceConfig, class_manager: Optional[ClassManager] = None):
        """
        Args:
            config: Configuração de inferência
            class_manager: Gerenciador de classes (opcional)
        """
        self.config = config
        self.class_manager = class_manager or ClassManager()
        
        # Utilitários
        self.device_manager = DeviceManager()
        self.path_manager = PathManager()
        
        # Componentes
        self.model = None
        self.image_processor = None
        self.id2label = None
        
        # Estatísticas
        self.stats = {
            "total_frames": 0,
            "processed_frames": 0,
            "detections_by_class": {cid: 0 for cid in range(self.class_manager.num_classes)},
            "fps": 0.0
        }
    
    def setup(self) -> None:
        """
        Carrega modelo e valida classes.
        Conforme PRD:
        - Carregar modelo treinado
        - Validar classes do modelo vs sistema
        - Log de compatibilidade
        """
        device = self.device_manager.get_device()
        
        # Carregar modelo treinado
        self.model, self.image_processor, model_id2label = load_detr_model_for_inference(
            model_path=self.config.model_path,
            device=device
        )
        
        # Validar classes do modelo vs sistema
        system_id2label = self.class_manager.get_id2label()
        
        if model_id2label != system_id2label:
            print("AVISO: Classes do modelo diferem das classes do sistema!")
            print(f"  Modelo: {model_id2label}")
            print(f"  Sistema: {system_id2label}")
            print("  Usando classes do modelo...")
            self.id2label = model_id2label
        else:
            print("✓ Classes do modelo compatíveis com sistema")
            self.id2label = system_id2label
        
        self.model.eval()
    
    @torch.no_grad()
    def detect_frame(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Detecta objetos em um frame.
        
        Args:
            frame: Frame BGR do OpenCV
        
        Returns:
            Lista de detecções
        """
        # Converter BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Processar com image_processor
        inputs = self.image_processor(images=pil_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device_manager.get_device())
        
        # Inferência com FP16 (se CUDA)
        use_amp = self.config.confidence_threshold > 0 and self.device_manager.cuda_available
        
        if use_amp:
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = self.model(pixel_values=pixel_values)
        else:
            outputs = self.model(pixel_values=pixel_values)
        
        # Pós-processamento
        target_sizes = torch.tensor([pil_image.size[::-1]])  # [height, width]
        results = self.image_processor.post_process_object_detection(
            outputs,
            threshold=self.config.confidence_threshold,
            target_sizes=target_sizes
        )[0]
        
        # Converter para DetectionResult
        detections = []
        for score, label, box in zip(
            results["scores"],
            results["labels"],
            results["boxes"]
        ):
            class_id = label.item()
            class_name = self.id2label.get(class_id, f"class_{class_id}")
            color = self.class_manager.get_class_color(class_id, bgr=True)
            
            # Converter box de [x1, y1, x2, y2] para lista
            bbox = box.tolist()
            
            detections.append(DetectionResult(
                bbox=bbox,
                score=score.item(),
                class_id=class_id,
                class_name=class_name,
                color=color
            ))
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        """
        Desenha bounding boxes e labels no frame.
        
        Args:
            frame: Frame BGR
            detections: Lista de detecções
        
        Returns:
            Frame com detecções desenhadas
        """
        frame_copy = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = [int(coord) for coord in det.bbox]
            
            # Desenhar bbox
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), det.color, 2)
            
            # Label com score
            label = f"{det.class_name}: {det.score:.2f}"
            
            # Tamanho do texto
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Background do texto
            cv2.rectangle(
                frame_copy,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                det.color,
                -1
            )
            
            # Texto
            cv2.putText(
                frame_copy,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return frame_copy
    
    def process_video(self) -> Dict[str, Any]:
        """
        Processa vídeo completo.
        Conforme PRD:
        - Abrir vídeo (OpenCV)
        - Processar frame por frame
        - Salvar vídeo de saída
        - Calcular estatísticas
        """
        video_path = Path(self.config.video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")
        
        # Abrir vídeo
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")
        
        # Propriedades do vídeo
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.stats["total_frames"] = total_frames
        
        # Configurar vídeo de saída
        output_path = self.config.output_path
        if not output_path:
            output_path = self.path_manager.get_prediction_output_path(
                str(video_path),
                Path(self.config.model_path).stem
            )
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Processar frames
        frame_times = []
        start_time = time.time()
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detectar objetos
            frame_start = time.time()
            detections = self.detect_frame(frame)
            frame_time = time.time() - frame_start
            frame_times.append(frame_time)
            
            # Atualizar estatísticas
            for det in detections:
                self.stats["detections_by_class"][det.class_id] += 1
            
            # Desenhar detecções
            frame_annotated = self.draw_detections(frame, detections)
            
            # Salvar frame
            if self.config.save_video:
                out.write(frame_annotated)
            
            # Preview (limitado a 30 FPS)
            if self.config.show_preview:
                preview_frame = cv2.resize(frame_annotated, (1280, 720))
                cv2.imshow('Preview', preview_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_idx += 1
            self.stats["processed_frames"] = frame_idx
            
            # Log progresso
            if frame_idx % 30 == 0:
                progress = (frame_idx / total_frames) * 100
                avg_fps = 1.0 / np.mean(frame_times[-30:]) if frame_times else 0
                print(f"Processado: {frame_idx}/{total_frames} ({progress:.1f}%) | FPS: {avg_fps:.1f}")
        
        # Finalizar
        cap.release()
        if self.config.save_video:
            out.release()
        if self.config.show_preview:
            cv2.destroyAllWindows()
        
        # Calcular estatísticas finais
        total_time = time.time() - start_time
        self.stats["fps"] = frame_idx / total_time if total_time > 0 else 0
        
        # Estatísticas por classe
        stats_by_class = {}
        for class_id, count in self.stats["detections_by_class"].items():
            class_name = self.id2label.get(class_id, f"class_{class_id}")
            stats_by_class[class_name] = count
        
        # Salvar estatísticas
        stats_path = output_path.parent / f"{output_path.stem}_stats.json"
        stats_data = {
            "video_path": str(video_path),
            "model_path": str(self.config.model_path),
            "output_path": str(output_path),
            "total_frames": self.stats["total_frames"],
            "processed_frames": self.stats["processed_frames"],
            "fps": self.stats["fps"],
            "detections_by_class": stats_by_class,
            "total_detections": sum(self.stats["detections_by_class"].values()),
            "confidence_threshold": self.config.confidence_threshold
        }
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Vídeo processado: {output_path}")
        print(f"  Frames: {self.stats['processed_frames']}/{self.stats['total_frames']}")
        print(f"  FPS médio: {self.stats['fps']:.2f}")
        print(f"  Detecções totais: {stats_data['total_detections']}")
        
        return stats_data
