"""
Trainer DETR com otimizações RTX 5080.
Sistema DETR - RTX 5080 Optimized
"""
import time
import csv
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast, GradScaler

from transformers import (
    DetrForObjectDetection,
    DetrImageProcessor,
    AutoModelForObjectDetection,
    AutoImageProcessor
)

from src.core.logger import logger
from src.core.config import PATHS, ModelConfig
from src.core.device_manager import device_manager
from src.core.session import session, ExperimentConfig
from src.data.coco_dataset import create_dataloaders
from src.evaluation.metrics import compute_precision_recall_iou_per_class


class DETRTrainer:
    """Trainer para modelos DETR com otimizações RTX 5080."""
    
    def __init__(
        self,
        experiment: ExperimentConfig,
        progress_callback: Optional[Callable[[float], None]] = None,
        metrics_callback: Optional[Callable[[Dict], None]] = None
    ):
        self.experiment = experiment
        self.progress_callback = progress_callback
        self.metrics_callback = metrics_callback
        
        self.model_config = ModelConfig()
        self.device = device_manager.device
        
        # Diretório do experimento
        self.exp_dir = PATHS.experiments / experiment.name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Componentes
        self.model = None
        self.processor = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        
        # Métricas
        self.best_loss = float('inf')
        self.best_map = 0.0
        self.train_metrics = []
        
        # Early stopping
        self.early_stopping_patience = getattr(experiment, 'early_stopping_patience', 10)
        self.early_stopping_counter = 0
        
        # Configurar otimizações
        self._setup_optimizations()
        
    def _setup_optimizations(self):
        """Configura otimizações RTX 5080."""
        # TF32
        if self.experiment.use_tf32 and device_manager.gpu_info.supports_tf32:
            device_manager.enable_tf32(True)
            logger.info("TF32 habilitado")
        
        # cuDNN benchmark
        torch.backends.cudnn.benchmark = True
        logger.info("cuDNN benchmark habilitado")
        
        # Precisão
        self.use_amp = self.experiment.precision in ["bf16", "fp16"]
        self.amp_dtype = torch.bfloat16 if self.experiment.precision == "bf16" else torch.float16
        
        # GradScaler (apenas para fp16)
        if self.experiment.precision == "fp16":
            self.scaler = GradScaler('cuda')
            logger.info("GradScaler habilitado (FP16)")
        
        logger.info(f"Precisão: {self.experiment.precision.upper()}")
    
    def _load_model(self):
        """Carrega modelo DETR com loss weights e label smoothing otimizados."""
        logger.info(f"Carregando modelo: {self.experiment.model_id}")
        
        # Parâmetros de loss weights e label smoothing
        model_kwargs = {
            "num_labels": self.model_config.NUM_CLASSES,
            "ignore_mismatched_sizes": True,
        }
        
        # Aplicar loss weights se disponíveis no modelo
        # Verificar se o modelo suporta esses parâmetros
        try:
            # Tentar carregar modelo base primeiro para verificar configuração
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(self.experiment.model_id)
            
            # Aplicar loss weights se o modelo suportar
            if hasattr(config, 'bbox_loss_coefficient'):
                model_kwargs["bbox_loss_coefficient"] = getattr(
                    self.experiment, 'bbox_loss_coefficient', 5.0
                )
            if hasattr(config, 'giou_loss_coefficient'):
                model_kwargs["giou_loss_coefficient"] = getattr(
                    self.experiment, 'giou_loss_coefficient', 2.0
                )
            if hasattr(config, 'class_loss_coefficient'):
                model_kwargs["class_loss_coefficient"] = getattr(
                    self.experiment, 'class_loss_coefficient', 1.0
                )
            if hasattr(config, 'eos_coefficient'):
                model_kwargs["eos_coefficient"] = getattr(
                    self.experiment, 'eos_coefficient', 0.1
                )
            if hasattr(config, 'label_smoothing'):
                label_smoothing = getattr(self.experiment, 'label_smoothing', 0.0)
                if label_smoothing > 0:
                    model_kwargs["label_smoothing"] = label_smoothing
        except Exception as e:
            logger.warning(f"Não foi possível verificar configuração do modelo: {e}")
        
        try:
            # Tentar carregar como AutoModel primeiro
            self.model = AutoModelForObjectDetection.from_pretrained(
                self.experiment.model_id,
                **model_kwargs
            )
            self.processor = AutoImageProcessor.from_pretrained(self.experiment.model_id)
        except Exception as e:
            logger.warning(f"Falha ao carregar com Auto: {e}")
            # Fallback para DETR padrão
            self.model = DetrForObjectDetection.from_pretrained(
                "facebook/detr-resnet-50",
                **model_kwargs
            )
            # Configurar DetrImageProcessor conforme especificação técnica
            # Nota: DetrImageProcessor não tem parâmetros de augmentation diretamente
            # A augmentation é aplicada pelo dataset antes do processamento
            self.processor = DetrImageProcessor.from_pretrained(
                "facebook/detr-resnet-50",
                size={"shortest_edge": self.experiment.image_size, "longest_edge": 1333},
                do_resize=True,
                do_rescale=True,
                do_normalize=True,
                do_pad=True
            )
            logger.info(f"DetrImageProcessor configurado: image_size={self.experiment.image_size}")
        
        # Aplicar loss weights diretamente na configuração do modelo se possível
        try:
            if hasattr(self.model.config, 'bbox_loss_coefficient'):
                self.model.config.bbox_loss_coefficient = getattr(
                    self.experiment, 'bbox_loss_coefficient', 5.0
                )
            if hasattr(self.model.config, 'giou_loss_coefficient'):
                self.model.config.giou_loss_coefficient = getattr(
                    self.experiment, 'giou_loss_coefficient', 2.0
                )
            if hasattr(self.model.config, 'class_loss_coefficient'):
                self.model.config.class_loss_coefficient = getattr(
                    self.experiment, 'class_loss_coefficient', 1.0
                )
            if hasattr(self.model.config, 'eos_coefficient'):
                self.model.config.eos_coefficient = getattr(
                    self.experiment, 'eos_coefficient', 0.1
                )
            if hasattr(self.model.config, 'label_smoothing'):
                label_smoothing = getattr(self.experiment, 'label_smoothing', 0.0)
                if label_smoothing > 0:
                    self.model.config.label_smoothing = label_smoothing
                    logger.info(f"Label smoothing aplicado: {label_smoothing}")
        except Exception as e:
            logger.warning(f"Não foi possível aplicar loss weights diretamente: {e}")
        
        # Log dos loss weights aplicados
        bbox_coef = getattr(self.experiment, 'bbox_loss_coefficient', 5.0)
        giou_coef = getattr(self.experiment, 'giou_loss_coefficient', 2.0)
        class_coef = getattr(self.experiment, 'class_loss_coefficient', 1.0)
        logger.info(f"Loss weights: bbox={bbox_coef}, giou={giou_coef}, class={class_coef}")
        
        self.model.to(self.device)
        
        # torch.compile se habilitado e suportado
        if self.experiment.use_torch_compile and device_manager.supports_torch_compile():
            logger.info("Aplicando torch.compile...")
            try:
                self.model = torch.compile(self.model)
                logger.info("torch.compile aplicado com sucesso")
            except Exception as e:
                logger.warning(f"torch.compile falhou: {e}")
                logger.warning("Continuando sem torch.compile...")
        elif self.experiment.use_torch_compile:
            import platform
            if platform.system() == "Windows":
                logger.warning("torch.compile nao disponivel no Windows (requer Triton)")
            else:
                logger.warning("torch.compile nao suportado nesta configuracao")
        
        # Contar parâmetros
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Parâmetros totais: {total_params:,}")
        logger.info(f"Parâmetros treináveis: {trainable_params:,}")
    
    def _setup_optimizer(self, num_training_steps: int):
        """Configura optimizer e scheduler com LR separado para backbone e head."""
        # Separar parâmetros do backbone e head para LR diferentes
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if 'backbone' in name.lower():
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        # Usar lr_backbone se disponível, senão usar learning_rate para ambos
        lr_backbone = getattr(self.experiment, 'lr_backbone', self.experiment.learning_rate / 10)
        
        # AdamW optimizer com LR diferentes para backbone e head
        if backbone_params and lr_backbone != self.experiment.learning_rate:
            self.optimizer = AdamW([
                {'params': backbone_params, 'lr': lr_backbone},
                {'params': head_params, 'lr': self.experiment.learning_rate}
            ], weight_decay=self.experiment.weight_decay, betas=(0.9, 0.999))
            logger.info(f"Optimizer: AdamW (backbone lr={lr_backbone}, head lr={self.experiment.learning_rate})")
        else:
            # Fallback: LR único para todos os parâmetros
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.experiment.learning_rate,
                weight_decay=self.experiment.weight_decay
            )
            logger.info(f"Optimizer: AdamW (lr={self.experiment.learning_rate})")
        
        # Warmup + Cosine scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.experiment.warmup_steps
        )
        
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps - self.experiment.warmup_steps,
            eta_min=1e-7
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.experiment.warmup_steps]
        )
        
        logger.info(f"Scheduler: Warmup ({self.experiment.warmup_steps} steps) + Cosine")
    
    def _create_dataloaders(self):
        """Cria DataLoaders."""
        use_augmentation = getattr(self.experiment, 'use_augmentation', False)
        
        # Detectar se são objetos pequenos baseado em image_size (>= 900 = pequenos)
        small_objects = getattr(self.experiment, 'image_size', 800) >= 900
        
        return create_dataloaders(
            dataset_path=Path(self.experiment.dataset_path),
            batch_size=self.experiment.batch_size,
            image_size=self.experiment.image_size,
            num_workers=self.experiment.num_workers,
            pin_memory=self.experiment.pin_memory,
            use_augmentation=use_augmentation,
            # Augmentation flags - básicas
            enable_mosaic=getattr(self.experiment, 'enable_mosaic', False),  # Roboflow já faz
            enable_mixup=getattr(self.experiment, 'enable_mixup', True),
            enable_cutmix=getattr(self.experiment, 'enable_cutmix', False),
            enable_copy_paste=getattr(self.experiment, 'enable_copy_paste', False),  # Roboflow já faz
            enable_random_erasing=getattr(self.experiment, 'enable_random_erasing', True),
            enable_motion_blur=getattr(self.experiment, 'enable_motion_blur', True),
            enable_grid_mask=getattr(self.experiment, 'enable_grid_mask', True),
            enable_cutout=getattr(self.experiment, 'enable_cutout', True),
            # Augmentation flags - YOLO-style
            enable_hsv=getattr(self.experiment, 'enable_hsv', True),
            enable_scale=getattr(self.experiment, 'enable_scale', True),
            enable_translate=getattr(self.experiment, 'enable_translate', True),
            enable_shear=getattr(self.experiment, 'enable_shear', False),
            enable_vflip=getattr(self.experiment, 'enable_vflip', False),
            # Augmentation probabilities - básicas
            hflip_prob=getattr(self.experiment, 'hflip_prob', 0.5),
            vflip_prob=getattr(self.experiment, 'vflip_prob', 0.0),
            color_jitter_prob=getattr(self.experiment, 'color_jitter_prob', 0.5),
            gaussian_blur_prob=getattr(self.experiment, 'gaussian_blur_prob', 0.3),
            # Augmentation probabilities - avançadas
            mosaic_prob=getattr(self.experiment, 'mosaic_prob', 0.0),
            mixup_prob=getattr(self.experiment, 'mixup_prob', 0.15),
            cutmix_prob=getattr(self.experiment, 'cutmix_prob', 0.0),
            copy_paste_prob=getattr(self.experiment, 'copy_paste_prob', 0.0),
            random_erasing_prob=getattr(self.experiment, 'random_erasing_prob', 0.3),
            motion_blur_prob=getattr(self.experiment, 'motion_blur_prob', 0.2),
            grid_mask_prob=getattr(self.experiment, 'grid_mask_prob', 0.2),
            cutout_prob=getattr(self.experiment, 'cutout_prob', 0.3),
            # Augmentation probabilities - YOLO-style
            hsv_prob=getattr(self.experiment, 'hsv_prob', 0.5),
            scale_prob=getattr(self.experiment, 'scale_prob', 0.5),
            translate_prob=getattr(self.experiment, 'translate_prob', 0.3),
            shear_prob=getattr(self.experiment, 'shear_prob', 0.0),
            # Parâmetros HSV
            hsv_h=getattr(self.experiment, 'hsv_h', 0.015),
            hsv_s=getattr(self.experiment, 'hsv_s', 0.7),
            hsv_v=getattr(self.experiment, 'hsv_v', 0.4),
            # Parâmetros geométricos
            scale_range_min=getattr(self.experiment, 'scale_range_min', 0.5),
            scale_range_max=getattr(self.experiment, 'scale_range_max', 1.5),
            translate_range=getattr(self.experiment, 'translate_range', 0.1),
            shear_range=getattr(self.experiment, 'shear_range', 0.0),
            small_objects=small_objects
        )
    
    def train(self):
        """Loop principal de treinamento."""
        logger.section("INICIANDO TREINAMENTO")
        
        # Carregar modelo
        self._load_model()
        
        # Criar dataloaders
        train_loader, val_loader, test_loader = self._create_dataloaders()
        
        # Setup optimizer
        # Ajustar número de steps considerando gradient accumulation
        # Steps efetivos = batches por época / grad_accum * épocas
        batches_per_epoch = len(train_loader)
        effective_batches_per_epoch = (batches_per_epoch + self.experiment.grad_accum - 1) // self.experiment.grad_accum
        num_training_steps = effective_batches_per_epoch * self.experiment.epochs
        self._setup_optimizer(num_training_steps)
        
        logger.info(f"Gradient Accumulation: {self.experiment.grad_accum} steps")
        logger.info(f"Batch size efetivo: {self.experiment.batch_size * self.experiment.grad_accum}")
        logger.info(f"Batches por época: {batches_per_epoch} → Steps efetivos: {effective_batches_per_epoch}")
        logger.info(f"Total training steps: {num_training_steps}")
        
        # Logging configuration
        log_steps = getattr(self.experiment, 'log_steps', 50)
        eval_steps = getattr(self.experiment, 'eval_steps', 500)
        save_steps = getattr(self.experiment, 'save_steps', 1000)
        logger.info(f"Logging configurado: log_steps={log_steps}, eval_steps={eval_steps}, save_steps={save_steps}")
        
        # Arquivo de métricas
        metrics_file = self.exp_dir / "train_metrics.csv"
        self._init_metrics_file(metrics_file)
        
        # TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=str(PATHS.runs / self.experiment.name))
        except ImportError:
            writer = None
            logger.warning("TensorBoard não disponível")
        
        global_step = 0
        start_time = time.time()
        
        logger.info("=" * 70)
        logger.info("INICIANDO LOOP DE TREINAMENTO")
        logger.info(f"Total de épocas: {self.experiment.epochs}")
        logger.info(f"Logging a cada {getattr(self.experiment, 'log_steps', 50)} steps")
        logger.info("=" * 70)
        
        for epoch in range(self.experiment.epochs):
            if session.is_stop_requested():
                logger.info("Treinamento interrompido pelo usuário")
                break
            
            # Treinar época
            epoch_loss, global_step = self._train_epoch(
                epoch, train_loader, global_step, metrics_file, writer
            )
            
            # Validação
            if val_loader:
                val_loss = self._validate(epoch, val_loader)
                
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.early_stopping_counter = 0  # Reset counter
                    self._save_checkpoint("best_model.pt", epoch, val_loss)
                    logger.info(f"Melhor modelo salvo! Val Loss: {val_loss:.4f}")
                else:
                    self.early_stopping_counter += 1
                    logger.info(f"Early stopping: {self.early_stopping_counter}/{self.early_stopping_patience}")
                
                # Early stopping
                if self.early_stopping_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping ativado após {epoch+1} épocas")
                    logger.info(f"Melhor val loss: {self.best_loss:.4f}")
                    break
            
            # Métricas de predição no Test por classe (precision, recall, IoU)
            if test_loader:
                test_metrics = self._evaluate_test_metrics(epoch, test_loader)
                if test_metrics and test_metrics.get("per_class"):
                    logger.info(f"Test Epoch {epoch+1} | Métricas por classe:")
                    for class_name, m in test_metrics["per_class"].items():
                        logger.info(
                            f"  {class_name}: Precision={m['precision']:.4f} | "
                            f"Recall={m['recall']:.4f} | IoU={m['mean_iou']:.4f}"
                        )
                    if writer:
                        for class_name, m in test_metrics["per_class"].items():
                            safe = class_name.replace(" ", "_")
                            writer.add_scalar(f"Test/{safe}/precision", m["precision"], epoch)
                            writer.add_scalar(f"Test/{safe}/recall", m["recall"], epoch)
                            writer.add_scalar(f"Test/{safe}/mean_iou", m["mean_iou"], epoch)
            
            # Checkpoint periódico
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt", epoch, epoch_loss)
            
            # Progresso
            progress = ((epoch + 1) / self.experiment.epochs) * 100
            if self.progress_callback:
                self.progress_callback(progress)
        
        # Salvar modelo final
        self._save_checkpoint("final_model.pt", epoch, epoch_loss)
        
        total_time = time.time() - start_time
        logger.info(f"Treinamento concluído em {total_time/60:.1f} minutos")
        
        if writer:
            writer.close()
    
    def _train_epoch(
        self,
        epoch: int,
        train_loader,
        global_step: int,
        metrics_file: Path,
        writer
    ) -> tuple:
        """Treina uma época com gradient accumulation."""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        grad_accum = self.experiment.grad_accum
        log_steps = getattr(self.experiment, 'log_steps', 50)
        
        # Log inicial da época
        logger.info(f"Iniciando época {epoch+1}/{self.experiment.epochs} | Total de batches: {num_batches}")
        
        # Contador para acumulação
        accum_step = 0
        accum_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            if session.is_stop_requested():
                break
            
            iter_start = time.time()
            
            # Mover para device
            images = images.to(self.device)
            
            # Preparar labels no formato DETR
            labels = self._prepare_labels(targets)
            
            # Zero grad apenas no início de cada grupo de acumulação
            if accum_step == 0:
                self.optimizer.zero_grad()
            
            # Forward pass com mixed precision
            if self.use_amp:
                with autocast('cuda', dtype=self.amp_dtype):
                    outputs = self.model(pixel_values=images, labels=labels)
                    # Dividir loss pelo número de steps de acumulação
                    loss = outputs.loss / grad_accum
                
                # Backward (acumula gradientes)
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                outputs = self.model(pixel_values=images, labels=labels)
                # Dividir loss pelo número de steps de acumulação
                loss = outputs.loss / grad_accum
                loss.backward()
            
            accum_loss += loss.item() * grad_accum  # Desnormalizar para logging
            
            accum_step += 1
            
            # Step do optimizer apenas após acumular todos os gradientes
            if accum_step == grad_accum:
                # Gradient clipping antes do step
                if self.use_amp and self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.experiment.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.experiment.max_grad_norm
                    )
                    self.optimizer.step()
                
                # Step do scheduler apenas após optimizer step
                self.scheduler.step()
                
                # Métricas
                iter_time = (time.time() - iter_start) * 1000
                batch_loss = accum_loss / grad_accum  # Loss média do grupo
                total_loss += batch_loss
                
                # VRAM
                vram_used = 0
                if device_manager.gpu_info.is_available:
                    _, vram_used, _ = device_manager.get_vram_status()
                
                # Logging - usar global_step e log_steps conforme especificação técnica
                log_steps = getattr(self.experiment, 'log_steps', 50)
                # Log no primeiro step e depois a cada log_steps
                should_log = (global_step == 0) or (global_step > 0 and global_step % log_steps == 0)
                
                if should_log:
                    current_lr = self.scheduler.get_last_lr()[0]
                    
                    logger.info(
                        f"Epoch {epoch+1}/{self.experiment.epochs} | "
                        f"Step {global_step} | "
                        f"Batch {batch_idx-grad_accum+1}-{batch_idx+1}/{num_batches} "
                        f"(Accum: {grad_accum}) | "
                        f"Loss: {batch_loss:.4f} | "
                        f"LR: {current_lr:.2e} | "
                        f"VRAM: {vram_used:.1f}GB"
                    )
                    
                    # Callback de métricas
                    if self.metrics_callback:
                        self.metrics_callback({
                            "epoch": epoch + 1,
                            "total_epochs": self.experiment.epochs,
                            "loss": batch_loss,
                            "lr": current_lr,
                            "vram_gb": vram_used,
                            "iter_time_ms": iter_time
                        })
                    
                    # TensorBoard
                    if writer:
                        writer.add_scalar("Loss/train", batch_loss, global_step)
                        writer.add_scalar("LR", current_lr, global_step)
                        writer.add_scalar("VRAM_GB", vram_used, global_step)
                
                # Salvar métricas em CSV (apenas após step completo)
                self._write_metrics(
                    metrics_file,
                    epoch, batch_idx, batch_loss,
                    self.scheduler.get_last_lr()[0],
                    vram_used, iter_time
                )
                
                # Reset acumulação
                accum_step = 0
                accum_loss = 0.0
                global_step += 1
        
        # Processar batches restantes se houver (último grupo incompleto)
        if accum_step > 0:
            if self.use_amp and self.scaler:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.experiment.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.experiment.max_grad_norm
                )
                self.optimizer.step()
            
            self.scheduler.step()
            batch_loss = accum_loss / accum_step
            total_loss += batch_loss
            global_step += 1
        
        # Calcular loss médio considerando apenas steps completos
        effective_batches = (num_batches // grad_accum) + (1 if num_batches % grad_accum > 0 else 0)
        avg_loss = total_loss / max(1, effective_batches)
        logger.info(f"Epoch {epoch+1} concluída | Loss médio: {avg_loss:.4f}")
        
        return avg_loss, global_step
    
    def _prepare_labels(self, targets):
        """Prepara labels no formato esperado pelo DETR."""
        labels = []
        for target in targets:
            labels.append({
                "class_labels": target["labels"].to(self.device),
                "boxes": target["boxes"].to(self.device)
            })
        return labels
    
    def _validate(self, epoch: int, val_loader) -> float:
        """Executa validação."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)
                labels = self._prepare_labels(targets)
                
                if self.use_amp:
                    with autocast('cuda', dtype=self.amp_dtype):
                        outputs = self.model(pixel_values=images, labels=labels)
                else:
                    outputs = self.model(pixel_values=images, labels=labels)
                
                total_loss += outputs.loss.item()
        
        avg_loss = total_loss / len(val_loader)
        logger.info(f"Validação Epoch {epoch+1} | Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _postprocess_outputs_for_eval(
        self,
        outputs,
        batch_idx: int,
        image_size: int,
        threshold: float = 0.05
    ) -> List[Dict[str, Any]]:
        """
        Pós-processa outputs do modelo para avaliação.
        Converte logits e pred_boxes em lista de detecções [image_id, category_id, bbox, score].
        """
        logits = outputs.logits  # [B, num_queries, num_classes]
        boxes = outputs.pred_boxes  # [B, num_queries, 4] - [cx, cy, w, h] normalizado
        
        batch_size = logits.shape[0]
        num_classes = self.model_config.NUM_CLASSES
        
        # Softmax
        probs = torch.softmax(logits, dim=-1)
        if probs.shape[-1] > num_classes:
            scores, labels = probs[..., :-1].max(dim=-1)
        else:
            scores, labels = probs.max(dim=-1)
        
        all_detections = []
        
        for b in range(batch_size):
            img_id = batch_idx * self.experiment.batch_size + b
            mask = scores[b] > threshold
            
            for i in torch.where(mask)[0]:
                score = float(scores[b, i].cpu())
                label = int(labels[b, i].cpu())
                cx, cy, w, h = boxes[b, i].cpu().numpy()
                
                # Converter [cx, cy, w, h] normalizado -> [x1, y1, x2, y2] em espaço image_size
                cx_px = cx * image_size
                cy_px = cy * image_size
                w_px = w * image_size
                h_px = h * image_size
                x1 = max(0, cx_px - w_px / 2)
                y1 = max(0, cy_px - h_px / 2)
                x2 = min(image_size, cx_px + w_px / 2)
                y2 = min(image_size, cy_px + h_px / 2)
                
                all_detections.append({
                    "image_id": img_id,
                    "category_id": label,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "score": score
                })
        
        return all_detections
    
    def _targets_to_ground_truths(
        self,
        targets: List[Dict],
        batch_idx: int,
        batch_size: int,
        image_size: int
    ) -> List[Dict[str, Any]]:
        """Converte targets do DataLoader em formato ground_truth para métricas."""
        gts = []
        for b, target in enumerate(targets):
            img_id = batch_idx * self.experiment.batch_size + b
            boxes = target["boxes"]  # [N, 4] - [cx, cy, w, h] normalizado
            labels = target["labels"]  # [N]
            
            for i in range(boxes.shape[0]):
                cx, cy, w, h = boxes[i].numpy()
                cx_px = cx * image_size
                cy_px = cy * image_size
                w_px = w * image_size
                h_px = h * image_size
                x1 = max(0, cx_px - w_px / 2)
                y1 = max(0, cy_px - h_px / 2)
                x2 = min(image_size, cx_px + w_px / 2)
                y2 = min(image_size, cy_px + h_px / 2)
                
                gts.append({
                    "image_id": img_id,
                    "category_id": int(labels[i].numpy()),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)]
                })
        return gts
    
    def _evaluate_test_metrics(
        self,
        epoch: int,
        test_loader
    ) -> Optional[Dict[str, float]]:
        """
        Avalia o modelo no conjunto Test e retorna precision, recall e mean IoU.
        Utiliza dados do diretório Test conforme especificação.
        """
        self.model.eval()
        image_size = self.experiment.image_size
        threshold = getattr(self.experiment, 'eval_threshold', 0.5)
        iou_threshold = 0.5
        
        all_predictions: List[Dict[str, Any]] = []
        all_ground_truths: List[Dict[str, Any]] = []
        batch_size = self.experiment.batch_size
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(test_loader):
                images = images.to(self.device)
                
                if self.use_amp:
                    with autocast('cuda', dtype=self.amp_dtype):
                        outputs = self.model(pixel_values=images)
                else:
                    outputs = self.model(pixel_values=images)
                
                preds = self._postprocess_outputs_for_eval(
                    outputs, batch_idx, image_size, threshold
                )
                all_predictions.extend(preds)
                
                gts = self._targets_to_ground_truths(
                    targets, batch_idx, images.shape[0], image_size
                )
                all_ground_truths.extend(gts)
        
        if not all_ground_truths:
            logger.warning("Test set vazio ou sem anotações")
            return None
        
        class_names = self.model_config.FIXED_CLASSES
        per_class = compute_precision_recall_iou_per_class(
            all_predictions,
            all_ground_truths,
            class_names=class_names,
            iou_threshold=iou_threshold
        )
        
        return {"per_class": per_class}
    
    def _save_checkpoint(self, filename: str, epoch: int, loss: float):
        """Salva checkpoint."""
        checkpoint_path = self.exp_dir / filename
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict() if not hasattr(self.model, '_orig_mod') else self.model._orig_mod.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": loss,
            "experiment_config": vars(self.experiment)
        }
        
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint salvo: {checkpoint_path}")
    
    def _init_metrics_file(self, path: Path):
        """Inicializa arquivo CSV de métricas."""
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "epoch", "batch", "loss", "lr",
                "vram_allocated_gb", "iter_time_ms"
            ])
    
    def _write_metrics(
        self, path: Path,
        epoch: int, batch: int, loss: float,
        lr: float, vram: float, iter_time: float
    ):
        """Escreve métricas no CSV."""
        with open(path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                epoch, batch, loss, lr, vram, iter_time
            ])
