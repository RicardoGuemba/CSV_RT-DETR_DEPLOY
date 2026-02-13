"""
Módulo de treinamento DETR.
Conforme PRD - Sistema de Detecção de Objetos com DETR
"""
import time
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

from transformers import (
    DetrForObjectDetection,
    DetrImageProcessor,
    AutoModelForObjectDetection,
    AutoImageProcessor
)

from src.utils.device_manager import DeviceManager
from src.utils.class_manager import ClassManager
from src.utils.paths import PathManager
from src.hf_model_utils import load_detr_model_for_training, save_model
from src.coco_utils import (
    load_coco_annotations,
    find_annotations_file,
    create_category_mapping,
    format_annotation_for_detr,
    validate_coco_dataset
)


@dataclass
class TrainingConfig:
    """Configuração completa de treinamento conforme PRD."""
    # Modelo e Dataset
    model_name: str = "facebook/detr-resnet-50"
    dataset_path: str = ""
    experiment_name: str = ""
    
    # Treinamento básico
    epochs: int = 50
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    image_size: int = 640
    num_workers: int = 4
    
    # Otimização (AdamW)
    learning_rate: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 0.0001
    betas: Tuple[float, float] = (0.9, 0.999)
    max_grad_norm: float = 0.1
    
    # Scheduler
    lr_scheduler: str = "cosine"  # linear, cosine, polynomial, constant, cosine_with_restarts
    warmup_steps: int = 500
    
    # Arquitetura DETR
    num_queries: int = 100
    dropout: float = 0.1
    auxiliary_loss: bool = True
    freeze_backbone: bool = False
    
    # Loss weights
    class_loss_coefficient: float = 1.0
    bbox_loss_coefficient: float = 5.0
    giou_loss_coefficient: float = 2.0
    eos_coefficient: float = 0.1
    
    # Data Augmentation
    augment_hflip: bool = True
    augment_color: bool = True
    augment_scale: bool = True
    augment_crop: bool = False
    
    # Opções avançadas
    fp16: bool = True
    gradient_checkpointing: bool = False
    compile_model: bool = False
    early_stopping_patience: int = 10
    
    # Logging e salvamento
    log_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 1000
    seed: int = 42


class COCODetectionDataset(Dataset):
    """Dataset COCO para DETR conforme PRD."""
    
    def __init__(
        self,
        images_dir: str,
        annotations_file: str,
        image_processor: DetrImageProcessor,
        image_size: int = 640,
        class_manager: Optional[ClassManager] = None,
        augmentation: Optional[Any] = None,
    ):
        """
        Args:
            images_dir: Diretório com imagens
            annotations_file: Caminho para arquivo _annotations.coco.json
            image_processor: DetrImageProcessor do Hugging Face
            image_size: Tamanho da imagem
            class_manager: Gerenciador de classes
            augmentation: Pipeline de augmentation (opcional)
        """
        self.images_dir = Path(images_dir)
        self.annotations_file = Path(annotations_file)
        self.image_processor = image_processor
        self.image_size = image_size
        self.class_manager = class_manager or ClassManager()
        self.augmentation = augmentation
        
        # Carregar anotações COCO
        self.coco_data = load_coco_annotations(self.annotations_file)
        
        # Criar mapeamento de categorias
        categories = self.coco_data.get("categories", [])
        system_classes = self.class_manager.class_names
        self.category_mapping = create_category_mapping(categories, system_classes)
        
        # Validar mapeamento
        self._validate_dataset_mapping()
        
        # Indexar imagens e anotações
        self.images = {img["id"]: img for img in self.coco_data.get("images", [])}
        self.annotations_by_image = {}
        for ann in self.coco_data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in self.annotations_by_image:
                self.annotations_by_image[img_id] = []
            self.annotations_by_image[img_id].append(ann)
        
        self.image_ids = list(self.images.keys())
    
    def _validate_dataset_mapping(self):
        """Valida que todas as classes foram mapeadas corretamente."""
        categories = self.coco_data.get("categories", [])
        mapped_ids = set(self.category_mapping.values())
        expected_ids = set(range(self.class_manager.num_classes))
        
        if mapped_ids != expected_ids:
            missing = expected_ids - mapped_ids
            raise ValueError(
                f"Mapeamento incompleto! Classes não mapeadas: {missing}. "
                f"Verifique as categorias do dataset."
            )
    
    def _create_category_mapping(self) -> Dict[int, int]:
        """Cria mapeamento de categorias do dataset para classes do sistema."""
        return self.category_mapping
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retorna imagem e anotações formatadas para DETR.
        
        Formato esperado pelo DetrImageProcessor:
        {
            "image_id": int,
            "annotations": [
                {
                    "bbox": [x, y, w, h],  # COCO format (pixels)
                    "category_id": int,
                    "area": float,
                    "iscrowd": int,
                }
            ]
        }
        """
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]
        
        # Carregar imagem
        image_path = self.images_dir / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")
        
        # Processar anotações
        annotations = self.annotations_by_image.get(image_id, [])
        formatted_annotations = []
        
        for ann in annotations:
            # Mapear category_id do dataset para system_id
            dataset_cat_id = ann["category_id"]
            system_class_id = self.category_mapping.get(dataset_cat_id)
            
            if system_class_id is not None:
                formatted_annotations.append({
                    "bbox": ann["bbox"],  # [x, y, w, h] em pixels
                    "category_id": system_class_id,
                    "area": ann.get("area", 0.0),
                    "iscrowd": ann.get("iscrowd", 0)
                })
        
        # Formato para DetrImageProcessor
        annotation_dict = {
            "image_id": image_id,
            "annotations": formatted_annotations
        }
        
        # Aplicar augmentation se disponível (apenas em treinamento)
        if self.augmentation is not None:
            image, annotation_dict = self.augmentation(image, annotation_dict)
        
        # Processar com image_processor
        inputs = self.image_processor(
            images=image,
            annotations=annotation_dict,
            return_tensors="pt"
        )
        
        # Remover batch dimension
        pixel_values = inputs["pixel_values"].squeeze(0)
        labels = inputs["labels"][0] if "labels" in inputs else None
        
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "image_id": image_id
        }


class DETRTrainer:
    """Trainer DETR conforme PRD."""
    
    def __init__(
        self,
        config: TrainingConfig,
        class_manager: ClassManager,
        progress_callback: Optional[Callable] = None,
        stop_flag: Optional[Callable] = None,
    ):
        """
        Args:
            config: Configuração de treinamento
            class_manager: Gerenciador de classes
            progress_callback: Callback para progresso
            stop_flag: Função que retorna True se deve parar
        """
        self.config = config
        self.class_manager = class_manager
        self.progress_callback = progress_callback
        self.stop_flag = stop_flag or (lambda: False)
        
        # Utilitários
        self.device_manager = DeviceManager()
        self.path_manager = PathManager()
        
        # Validar CUDA
        is_valid, msg = self.device_manager.validate_for_training()
        if not is_valid:
            print(f"AVISO: {msg}")
        
        # Componentes
        self.model = None
        self.image_processor = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        
        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        
        # Métricas
        self.best_loss = float('inf')
        self.best_map = 0.0
        self.train_metrics = []
    
    def setup(self) -> None:
        """
        Configura modelo, datasets, optimizer, scheduler.
        Conforme PRD:
        1. Validar CUDA
        2. Carregar modelo com classes corretas
        3. Validar mapeamento de classes
        4. Configurar optimizer (AdamW com LR separado para head/backbone)
        5. Configurar scheduler (warmup + main)
        6. Configurar datasets com validação de mapeamento
        7. Configurar mixed precision (FP16)
        8. Configurar gradient checkpointing (se suportado)
        9. Configurar torch.compile (se PyTorch 2.0+)
        """
        # 1. Validar CUDA
        is_valid, msg = self.device_manager.validate_for_training()
        if not is_valid:
            print(f"AVISO: {msg}")
        
        device = self.device_manager.get_device()
        
        # 2. Carregar modelo com classes corretas
        id2label = self.class_manager.get_id2label()
        num_classes = self.class_manager.num_classes
        
        self.model, self.image_processor = load_detr_model_for_training(
            model_name=self.config.model_name,
            num_classes=num_classes,
            id2label=id2label,
            device=device
        )
        
        # 3. Validar mapeamento de classes (será feito no dataset)
        
        # 4. Configurar optimizer (AdamW com LR separado para head/backbone)
        self._setup_optimizer()
        
        # 5. Configurar scheduler (warmup + main)
        self._setup_scheduler()
        
        # 6. Configurar datasets com validação de mapeamento
        self._setup_datasets()
        
        # 7. Configurar mixed precision (FP16)
        if self.config.fp16 and self.device_manager.cuda_available:
            self.scaler = GradScaler('cuda')
        
        # 8. Configurar gradient checkpointing (se suportado)
        if self.config.gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        # 9. Configurar torch.compile (se PyTorch 2.0+)
        if self.config.compile_model:
            try:
                self.model = torch.compile(self.model)
                print("torch.compile habilitado")
            except Exception as e:
                print(f"torch.compile não disponível: {e}")
    
    def _setup_optimizer(self):
        """Configura optimizer AdamW com LR separado para head/backbone."""
        # Separar parâmetros do backbone e head
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        # Optimizer com LR diferentes
        self.optimizer = AdamW([
            {'params': backbone_params, 'lr': self.config.lr_backbone},
            {'params': head_params, 'lr': self.config.learning_rate}
        ], weight_decay=self.config.weight_decay, betas=self.config.betas)
    
    def _setup_scheduler(self):
        """Configura scheduler (warmup + main)."""
        num_training_steps = len(self.train_loader) * self.config.epochs
        
        # Warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.config.warmup_steps
        )
        
        # Main scheduler (cosine)
        if self.config.lr_scheduler == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps - self.config.warmup_steps,
                eta_min=1e-7
            )
        else:
            # Fallback para constante
            main_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=1.0,
                total_iters=num_training_steps - self.config.warmup_steps
            )
        
        # Sequential scheduler
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[self.config.warmup_steps]
        )
    
    def _setup_datasets(self):
        """Configura datasets com validação de mapeamento."""
        dataset_path = Path(self.config.dataset_path)
        
        # Validar dataset
        is_valid, msg = validate_coco_dataset(dataset_path, "train")
        if not is_valid:
            raise ValueError(f"Dataset inválido: {msg}")
        
        # Encontrar arquivos de anotações
        train_ann_file = find_annotations_file(dataset_path, "train")
        val_ann_file = find_annotations_file(dataset_path, "valid")
        
        if not train_ann_file:
            raise FileNotFoundError("Arquivo de anotações de treino não encontrado")
        
        # Criar datasets
        train_images_dir = train_ann_file.parent
        self.train_dataset = COCODetectionDataset(
            images_dir=str(train_images_dir),
            annotations_file=str(train_ann_file),
            image_processor=self.image_processor,
            image_size=self.config.image_size,
            class_manager=self.class_manager
        )
        
        # Validar mapeamento
        self._validate_dataset_mapping(self.train_dataset, "train")
        
        # Dataset de validação (opcional)
        if val_ann_file:
            val_images_dir = val_ann_file.parent
            self.val_dataset = COCODetectionDataset(
                images_dir=str(val_images_dir),
                annotations_file=str(val_ann_file),
                image_processor=self.image_processor,
                image_size=self.config.image_size,
                class_manager=self.class_manager
            )
            self._validate_dataset_mapping(self.val_dataset, "valid")
        
        # Criar DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
        
        if self.val_dataset:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True,
                collate_fn=self._collate_fn
            )
    
    def _validate_dataset_mapping(self, dataset, split_name: str) -> None:
        """Valida que todas as classes foram mapeadas corretamente."""
        # Já validado no __init__ do dataset, mas podemos adicionar logs aqui
        mapping = dataset.category_mapping
        mapped_classes = set(mapping.values())
        expected_classes = set(range(self.class_manager.num_classes))
        
        if mapped_classes != expected_classes:
            missing = expected_classes - mapped_classes
            raise ValueError(
                f"Mapeamento incompleto no {split_name}: classes não mapeadas: {missing}"
            )
        
        print(f"✓ Mapeamento validado para {split_name}: {len(mapping)} categorias -> {len(mapped_classes)} classes")
    
    def _collate_fn(self, batch):
        """Função de collate para DataLoader."""
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = [item["labels"] for item in batch]
        return pixel_values, labels
    
    def train_epoch(self) -> Dict[str, float]:
        """Treina uma época completa."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, (pixel_values, labels) in enumerate(self.train_loader):
            if self.stop_flag():
                break
            
            # Mover para device
            device = self.device_manager.get_device()
            pixel_values = pixel_values.to(device, non_blocking=True)
            
            # Gradient accumulation - zero grad no início de cada grupo
            if batch_idx % self.config.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()
            
            # Forward pass com mixed precision
            if self.config.fp16 and self.device_manager.cuda_available:
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss / self.config.gradient_accumulation_steps
                
                # Backward
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                outputs = self.model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / self.config.gradient_accumulation_steps
                loss.backward()
            
            # Gradient clipping e step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                
                self.scheduler.step()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            
            # Logging
            if (batch_idx + 1) % self.config.log_steps == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                print(
                    f"Step {batch_idx + 1}/{num_batches} | "
                    f"Loss: {loss.item() * self.config.gradient_accumulation_steps:.4f} | "
                    f"LR: {current_lr:.2e}"
                )
        
        avg_loss = total_loss / num_batches
        return {"loss": avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """Valida modelo no dataset de validação."""
        if not self.val_loader:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        
        device = self.device_manager.get_device()
        
        with torch.no_grad():
            for pixel_values, labels in self.val_loader:
                pixel_values = pixel_values.to(device, non_blocking=True)
                
                if self.config.fp16 and self.device_manager.cuda_available:
                    with autocast(device_type='cuda', dtype=torch.float16):
                        outputs = self.model(pixel_values=pixel_values, labels=labels)
                else:
                    outputs = self.model(pixel_values=pixel_values, labels=labels)
                
                total_loss += outputs.loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return {"loss": avg_loss}
    
    def train(self) -> Dict[str, Any]:
        """Executa treinamento completo com early stopping."""
        # Criar diretório do experimento
        exp_dirs = self.path_manager.create_experiment_dirs(self.config.experiment_name)
        exp_dir = exp_dirs["experiment"]
        
        # Salvar configuração
        config_path = exp_dir / "training_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.config), f, indent=2, ensure_ascii=False)
        
        # Métricas
        metrics_file = exp_dir / "metrics.csv"
        self._init_metrics_file(metrics_file)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            if self.stop_flag():
                print("Treinamento interrompido")
                break
            
            # Treinar época
            train_metrics = self.train_epoch()
            
            # Validar
            val_metrics = self.validate()
            
            # Salvar métricas
            self._write_metrics(metrics_file, epoch, train_metrics, val_metrics)
            
            # Early stopping
            val_loss = val_metrics.get("loss", float('inf'))
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                # Salvar melhor modelo
                self._save_checkpoint(exp_dirs["model_best"] / "best_model.pt", epoch)
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping após {epoch + 1} épocas")
                    break
            
            # Checkpoint periódico
            if len(self.train_loader) > 0 and (epoch + 1) % max(1, self.config.save_steps // len(self.train_loader)) == 0:
                checkpoint_dir = exp_dirs["checkpoints"] / f"checkpoint-{epoch + 1}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                self._save_checkpoint(checkpoint_dir / "checkpoint.pt", epoch)
            
            # Progresso
            if self.progress_callback:
                progress = ((epoch + 1) / self.config.epochs) * 100
                self.progress_callback(progress)
        
        # Salvar modelo final
        self._save_checkpoint(exp_dirs["model_final"] / "final_model.pt", epoch)
        
        # Salvar modelo completo
        id2label = self.class_manager.get_id2label()
        save_model(
            self.model,
            self.image_processor,
            str(exp_dirs["model_final"]),
            id2label,
            asdict(self.config)
        )
        
        return {
            "experiment_dir": str(exp_dir),
            "best_loss": best_loss,
            "epochs_completed": epoch + 1
        }
    
    def _save_checkpoint(self, path: Path, epoch: int):
        """Salva checkpoint."""
        # Obter state_dict do modelo (lidar com torch.compile)
        model_state_dict = self.model.state_dict()
        if hasattr(self.model, '_orig_mod'):
            model_state_dict = self.model._orig_mod.state_dict()
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": asdict(self.config)
        }
        
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
    
    def _init_metrics_file(self, path: Path):
        """Inicializa arquivo CSV de métricas."""
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "lr"])
    
    def _write_metrics(self, path: Path, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Escreve métricas no CSV."""
        with open(path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics.get("loss", 0.0),
                val_metrics.get("loss", 0.0),
                self.scheduler.get_last_lr()[0]
            ])
