"""
Aba de Treinamento.
Sistema DETR - RTX 5080 Optimized
"""
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from typing import Optional, Callable
from pathlib import Path
import time

try:
    import ttkbootstrap as ttkb
    from ttkbootstrap.constants import *
    USE_TTKBOOTSTRAP = True
except ImportError:
    USE_TTKBOOTSTRAP = False
    ttkb = None

from src.core.logger import logger
from src.core.config import Config, ModelConfig
from src.core.device_manager import device_manager
from src.core.session import session
from src.core.user_settings import settings_manager
from src.ui.tooltip import create_tooltip


class TrainingTab:
    """Tab de treinamento."""
    
    def __init__(self, parent: ttk.Frame, dataset_tab):
        self.parent = parent
        self.dataset_tab = dataset_tab
        self.config = Config()
        self.model_config = ModelConfig()
        
        # Variáveis de controle
        self._create_variables()
        self._create_ui()
        self._apply_vram_defaults()
        self._setup_auto_save()
        
        # Thread de treinamento
        self.training_thread: Optional[threading.Thread] = None
        
    def _create_variables(self):
        """Cria variáveis de controle."""
        # Carregar configurações salvas
        saved = settings_manager.training
        defaults = self.config.training
        
        # Modelo
        self.model_var = tk.StringVar(value=saved.model_name)
        
        # Hiperparâmetros - usar valores salvos ou defaults
        self.epochs_var = tk.IntVar(value=saved.epochs)
        self.batch_size_var = tk.IntVar(value=saved.batch_size)
        self.grad_accum_var = tk.IntVar(value=saved.grad_accum)
        self.lr_var = tk.DoubleVar(value=saved.learning_rate)
        self.lr_backbone_var = tk.DoubleVar(value=getattr(saved, 'lr_backbone', saved.learning_rate / 10))
        self.image_size_var = tk.IntVar(value=saved.image_size)
        self.weight_decay_var = tk.DoubleVar(value=saved.weight_decay)
        self.warmup_steps_var = tk.IntVar(value=saved.warmup_steps)
        self.max_grad_norm_var = tk.DoubleVar(value=saved.max_grad_norm)
        self.scheduler_var = tk.StringVar(value=getattr(saved, 'scheduler', 'cosine'))
        
        # Otimizações
        self.precision_var = tk.StringVar(value=saved.precision)
        self.use_tf32_var = tk.BooleanVar(value=saved.use_tf32 and device_manager.gpu_info.supports_tf32)
        self.use_compile_var = tk.BooleanVar(value=saved.use_compile and device_manager.supports_torch_compile())
        
        # DataLoader
        self.num_workers_var = tk.IntVar(value=saved.num_workers)
        self.pin_memory_var = tk.BooleanVar(value=defaults.pin_memory)
        
        # Anti-overfitting (carregar valores salvos ou defaults)
        self.dropout_var = tk.DoubleVar(value=saved.dropout if hasattr(saved, 'dropout') else 0.1)
        self.early_stopping_patience_var = tk.IntVar(value=saved.early_stopping_patience if hasattr(saved, 'early_stopping_patience') else 10)
        self.use_augmentation_var = tk.BooleanVar(value=saved.use_augmentation if hasattr(saved, 'use_augmentation') else False)
        self.label_smoothing_var = tk.DoubleVar(value=getattr(saved, 'label_smoothing', 0.0))
        
        # Data Augmentation - Flags de habilitação (básicas)
        self.enable_hflip_var = tk.BooleanVar(value=True)
        self.enable_vflip_var = tk.BooleanVar(value=getattr(saved, 'enable_vflip', False))
        self.enable_color_jitter_var = tk.BooleanVar(value=True)
        self.enable_gaussian_blur_var = tk.BooleanVar(value=True)
        self.enable_mosaic_var = tk.BooleanVar(value=getattr(saved, 'enable_mosaic', False))  # Roboflow já faz
        self.enable_mixup_var = tk.BooleanVar(value=getattr(saved, 'enable_mixup', True))
        self.enable_cutmix_var = tk.BooleanVar(value=getattr(saved, 'enable_cutmix', False))
        self.enable_copy_paste_var = tk.BooleanVar(value=getattr(saved, 'enable_copy_paste', False))  # Roboflow já faz
        self.enable_random_erasing_var = tk.BooleanVar(value=getattr(saved, 'enable_random_erasing', True))
        self.enable_motion_blur_var = tk.BooleanVar(value=getattr(saved, 'enable_motion_blur', True))
        self.enable_grid_mask_var = tk.BooleanVar(value=getattr(saved, 'enable_grid_mask', True))
        self.enable_cutout_var = tk.BooleanVar(value=getattr(saved, 'enable_cutout', True))
        
        # Data Augmentation - Flags YOLO-style
        self.enable_hsv_var = tk.BooleanVar(value=getattr(saved, 'enable_hsv', True))
        self.enable_scale_var = tk.BooleanVar(value=getattr(saved, 'enable_scale', True))
        self.enable_translate_var = tk.BooleanVar(value=getattr(saved, 'enable_translate', True))
        self.enable_shear_var = tk.BooleanVar(value=getattr(saved, 'enable_shear', False))
        
        # Data Augmentation - Probabilidades básicas
        self.hflip_prob_var = tk.DoubleVar(value=getattr(saved, 'hflip_prob', 0.5))
        self.vflip_prob_var = tk.DoubleVar(value=getattr(saved, 'vflip_prob', 0.0))
        self.color_jitter_prob_var = tk.DoubleVar(value=getattr(saved, 'color_jitter_prob', 0.5))
        self.gaussian_blur_prob_var = tk.DoubleVar(value=getattr(saved, 'gaussian_blur_prob', 0.3))
        
        # Data Augmentation - Probabilidades avançadas
        self.mosaic_prob_var = tk.DoubleVar(value=getattr(saved, 'mosaic_prob', 0.0))  # Desabilitado
        self.mixup_prob_var = tk.DoubleVar(value=getattr(saved, 'mixup_prob', 0.15))
        self.cutmix_prob_var = tk.DoubleVar(value=getattr(saved, 'cutmix_prob', 0.0))
        self.copy_paste_prob_var = tk.DoubleVar(value=getattr(saved, 'copy_paste_prob', 0.0))  # Desabilitado
        self.random_erasing_prob_var = tk.DoubleVar(value=getattr(saved, 'random_erasing_prob', 0.3))
        self.motion_blur_prob_var = tk.DoubleVar(value=getattr(saved, 'motion_blur_prob', 0.2))
        self.grid_mask_prob_var = tk.DoubleVar(value=getattr(saved, 'grid_mask_prob', 0.2))
        self.cutout_prob_var = tk.DoubleVar(value=getattr(saved, 'cutout_prob', 0.3))
        
        # Data Augmentation - Probabilidades YOLO-style
        self.hsv_prob_var = tk.DoubleVar(value=getattr(saved, 'hsv_prob', 0.5))
        self.scale_prob_var = tk.DoubleVar(value=getattr(saved, 'scale_prob', 0.5))
        self.translate_prob_var = tk.DoubleVar(value=getattr(saved, 'translate_prob', 0.3))
        self.shear_prob_var = tk.DoubleVar(value=getattr(saved, 'shear_prob', 0.0))
        
        # Parâmetros HSV
        self.hsv_h_var = tk.DoubleVar(value=getattr(saved, 'hsv_h', 0.015))
        self.hsv_s_var = tk.DoubleVar(value=getattr(saved, 'hsv_s', 0.7))
        self.hsv_v_var = tk.DoubleVar(value=getattr(saved, 'hsv_v', 0.4))
        
        # Parâmetros geométricos
        self.scale_range_min_var = tk.DoubleVar(value=getattr(saved, 'scale_range_min', 0.5))
        self.scale_range_max_var = tk.DoubleVar(value=getattr(saved, 'scale_range_max', 1.5))
        self.translate_range_var = tk.DoubleVar(value=getattr(saved, 'translate_range', 0.1))
        self.shear_range_var = tk.DoubleVar(value=getattr(saved, 'shear_range', 0.0))
        
        # Loss weights (carregar valores salvos ou defaults)
        self.bbox_loss_coefficient_var = tk.DoubleVar(value=getattr(saved, 'bbox_loss_coefficient', 5.0))
        self.giou_loss_coefficient_var = tk.DoubleVar(value=getattr(saved, 'giou_loss_coefficient', 2.0))
        self.class_loss_coefficient_var = tk.DoubleVar(value=getattr(saved, 'class_loss_coefficient', 1.0))
        self.eos_coefficient_var = tk.DoubleVar(value=getattr(saved, 'eos_coefficient', 0.1))
        
        # Experimento
        self.experiment_name_var = tk.StringVar(value="experiment")
        
        # Status
        self.progress_var = tk.DoubleVar(value=0)
        self.status_var = tk.StringVar(value="Pronto para treinar")
        self.epoch_var = tk.StringVar(value="Época: 0/0")
        self.loss_var = tk.StringVar(value="Loss: --")
        self.lr_display_var = tk.StringVar(value="LR: --")
        self.vram_display_var = tk.StringVar(value="VRAM: --")
        self.time_var = tk.StringVar(value="Tempo: --")
        
    def _create_scrollable_frame(self, parent):
        """Cria um frame scrollável usando Canvas."""
        # Container principal
        container = ttk.Frame(parent)
        container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)
        
        # Canvas
        canvas = tk.Canvas(container, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Frame interno (scrollável)
        scrollable_frame = ttk.Frame(canvas)
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        def configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        def configure_canvas_width(event):
            canvas_width = event.width
            canvas.itemconfig(canvas_window, width=canvas_width)
        
        scrollable_frame.bind("<Configure>", configure_scroll_region)
        canvas.bind("<Configure>", configure_canvas_width)
        
        # Bind mousewheel (com proteção contra canvas destruído)
        def on_mousewheel(event):
            try:
                if canvas.winfo_exists():
                    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except Exception:
                pass  # Ignora se o canvas foi destruído
        
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        return scrollable_frame
    
    def _create_ui(self):
        """Cria a interface."""
        # Configurar fundo limpo
        if not USE_TTKBOOTSTRAP:
            self.parent.configure(style='TFrame')
        
        self.parent.columnconfigure(0, weight=1)
        self.parent.columnconfigure(1, weight=1)
        self.parent.rowconfigure(0, weight=1)
        
        # Frame esquerdo - Configuração (com scroll)
        left_container = ttk.Frame(self.parent)
        left_container.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        left_container.columnconfigure(0, weight=1)
        left_container.rowconfigure(0, weight=1)
        
        left_frame = self._create_scrollable_frame(left_container)
        
        # Frame direito - Métricas e controle
        right_frame = ttk.Frame(self.parent)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        
        self._create_model_frame(left_frame)
        self._create_hyperparams_frame(left_frame)
        self._create_optimization_frame(left_frame)
        self._create_dataloader_frame(left_frame)
        self._create_antioverfitting_frame(left_frame)
        # Frame de augmentation (criado mas inicialmente oculto)
        self.augmentation_frame_container = ttk.Frame(left_frame)
        self._create_augmentation_frame(self.augmentation_frame_container)
        self._toggle_augmentation_section()  # Aplicar estado inicial
        self._create_loss_weights_frame(left_frame)
        
        self._create_control_frame(right_frame)
        self._create_metrics_frame(right_frame)
        self._create_progress_frame(right_frame)
        
    def _create_model_frame(self, parent):
        """Frame de seleção de modelo."""
        frame = ttk.LabelFrame(parent, text="🤖 Modelo Base", padding=15)
        frame.pack(fill=tk.X, pady=8)
        
        models = list(self.model_config.AVAILABLE_MODELS.keys())
        
        model_label = ttk.Label(frame, text="Modelo:")
        model_label.pack(anchor=tk.W)
        create_tooltip(model_label, "Selecione o modelo DETR para treinamento. RT-DETR ResNet-101 é recomendado para máxima precisão (mAP 54.3).")
        
        model_combo = ttk.Combobox(
            frame, 
            textvariable=self.model_var,
            values=models,
            state="readonly",
            width=40
        )
        model_combo.pack(fill=tk.X, pady=5)
        create_tooltip(model_combo, "Modelos disponíveis: RT-DETR (State-of-the-Art, mAP 54.3), DINO-DETR, DAB-DETR, Deformable DETR, Conditional DETR, DETR clássico. RT-DETR converge mais rápido (~12 epochs).")
        
        # Info do modelo
        info_label = ttk.Label(
            frame,
            text="RT-DETR ResNet-101 é o modelo State-of-the-Art recomendado.",
            font=("Segoe UI", 9, "italic")
        )
        info_label.pack(anchor=tk.W)
        
    def _create_hyperparams_frame(self, parent):
        """Frame de hiperparâmetros."""
        frame = ttk.LabelFrame(parent, text="⚙️ Hiperparâmetros", padding=15)
        frame.pack(fill=tk.X, pady=8)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(3, weight=1)
        
        # Dicionário de tooltips para hiperparâmetros
        tooltips = {
            "Épocas:": "Número total de épocas de treinamento. RT-DETR converge rápido (~12-20 epochs). Objetos pequenos podem precisar de 50-100 epochs.",
            "Batch Size:": "Número de imagens por batch. Maior = mais estável, mas usa mais VRAM. RTX 5080 suporta 12-16 para RT-DETR ResNet-101.",
            "Grad Accum:": "Acumulação de gradientes. Batch efetivo = batch_size × grad_accum. Útil quando VRAM é limitada.",
            "Image Size:": "Tamanho da imagem de entrada (px). Maior = melhor para objetos pequenos, mas mais lento. 640-800px é comum.",
            "Learning Rate:": "Taxa de aprendizado. 1e-5 é padrão. Muito alto = instável, muito baixo = lento. Use warmup para estabilizar início.",
            "Weight Decay:": "Regularização L2. Previne overfitting. 1e-4 é padrão. Aumente para datasets pequenos.",
            "Warmup Steps:": "Número de steps com LR crescente no início. Estabiliza treinamento. 500-1000 é comum.",
            "Max Grad Norm:": "Gradient clipping. Limita gradientes grandes para estabilidade. 1.0 é padrão. Reduza se loss oscilar."
        }
        
        params = [
            ("Épocas:", self.epochs_var, 0, 0, (1, 200)),
            ("Batch Size:", self.batch_size_var, 0, 2, (1, 64)),
            ("Grad Accum:", self.grad_accum_var, 1, 0, (1, 16)),
            ("Image Size:", self.image_size_var, 1, 2, (320, 1024)),
            ("Learning Rate:", self.lr_var, 2, 0, (1e-7, 1e-2)),
            ("LR Backbone:", self.lr_backbone_var, 2, 2, (1e-7, 1e-2)),
            ("Weight Decay:", self.weight_decay_var, 3, 0, (0, 0.1)),
            ("Warmup Steps:", self.warmup_steps_var, 3, 2, (0, 5000)),
            ("Max Grad Norm:", self.max_grad_norm_var, 4, 0, (0.1, 10.0)),
        ]
        
        # Scheduler
        scheduler_label = ttk.Label(frame, text="Scheduler:")
        scheduler_label.grid(row=4, column=2, sticky="w", pady=2)
        create_tooltip(scheduler_label, "Estratégia de ajuste do Learning Rate: Cosine (recomendado, decai suavemente), Linear (decai linearmente), Polynomial (decai polinomialmente), Constant (mantém LR fixo).")
        
        scheduler_combo = ttk.Combobox(
            frame,
            textvariable=self.scheduler_var,
            values=["cosine", "linear", "polynomial", "constant"],
            state="readonly",
            width=12
        )
        scheduler_combo.grid(row=4, column=3, sticky="w", padx=5, pady=2)
        create_tooltip(scheduler_combo, "Cosine annealing é recomendado para DETR. Reduz LR gradualmente seguindo curva cosseno, melhorando convergência final.")
        
        for label, var, row, col, (min_val, max_val) in params:
            label_widget = ttk.Label(frame, text=label)
            label_widget.grid(row=row, column=col, sticky="w", pady=2)
            create_tooltip(label_widget, tooltips.get(label, ""))
            
            if isinstance(var, tk.DoubleVar):
                entry = ttk.Entry(frame, textvariable=var, width=12)
            else:
                entry = ttk.Spinbox(frame, textvariable=var, from_=min_val, to=max_val, width=10)
            
            entry.grid(row=row, column=col+1, sticky="w", padx=5, pady=2)
            create_tooltip(entry, tooltips.get(label, ""))
        
        # Tooltip para LR Backbone (adicionar ao dicionário de tooltips)
        tooltips["LR Backbone:"] = "Learning rate para o backbone (ResNet). Geralmente 10x menor que o LR do head. Ex: se LR=1e-4, LR Backbone=1e-5"
        
        # Nome do experimento
        exp_label = ttk.Label(frame, text="Nome Experimento:")
        exp_label.grid(row=5, column=0, sticky="w", pady=2)
        create_tooltip(exp_label, "Nome único para identificar este experimento. Será usado como nome da pasta onde o modelo será salvo.")
        
        exp_entry = ttk.Entry(frame, textvariable=self.experiment_name_var, width=30)
        exp_entry.grid(row=5, column=1, columnspan=3, sticky="w", padx=5, pady=2)
        create_tooltip(exp_entry, "Nome do experimento. Ex: 'rtdetr_objetos_pequenos' ou 'treinamento_v1'. Evite caracteres especiais.")
        
        # Botões para aplicar presets
        preset_frame = ttk.Frame(frame)
        preset_frame.grid(row=6, column=0, columnspan=4, pady=10, sticky="ew")
        preset_frame.columnconfigure(0, weight=1)
        preset_frame.columnconfigure(1, weight=1)
        
        # Preset objetos médios/grandes (RECOMENDADO)
        preset_medium_large_btn = ttk.Button(
            preset_frame,
            text="⭐ Preset: Objetos Médios/Grandes [RECOMENDADO]",
            command=self._apply_medium_large_objects_preset
        )
        preset_medium_large_btn.grid(row=0, column=0, columnspan=2, padx=5, sticky="ew")
        create_tooltip(preset_medium_large_btn, "Preset otimizado para objetos médios/grandes considerando Roboflow. Roboflow faz mosaic e crop. Aqui: HSV, Scale, Translation, MixUp, regularização. Valores baseados em papers e YOLO defaults.")
        
        # Preset objetos pequenos
        preset_small_btn = ttk.Button(
            preset_frame,
            text="🎯 Preset: Objetos Pequenos (20-60px)",
            command=self._apply_small_objects_preset
        )
        preset_small_btn.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        create_tooltip(preset_small_btn, "Aplica hiperparâmetros otimizados para objetos pequenos (20-60px): image_size=960, batch_size=8, grad_accum=2, dropout=0.2, label_smoothing=0.1, augmentation agressiva.")
        
        # Preset objetos grandes
        preset_large_btn = ttk.Button(
            preset_frame,
            text="📦 Preset: Objetos Grandes (>100px)",
            command=self._apply_large_objects_preset
        )
        preset_large_btn.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        create_tooltip(preset_large_btn, "Aplica hiperparâmetros otimizados para objetos grandes (>100px): image_size=800, batch_size=12, grad_accum=1, dropout=0.1, augmentation moderada.")
        
        # Preset Especificação Técnica
        preset_spec_btn = ttk.Button(
            preset_frame,
            text="📋 Preset: Especificação Técnica (Testado)",
            command=self._apply_spec_tecnica_preset
        )
        preset_spec_btn.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        create_tooltip(preset_spec_btn, "Aplica configuração da Especificação Técnica que produziu excelentes resultados: facebook/detr-resnet-50, epochs=25, batch_size=2, grad_accum=4, image_size=640")
        
        ttk.Label(
            frame,
            text="Presets otimizados para diferentes tamanhos de objetos (baseados em papers e YOLO)",
            font=("Segoe UI", 8, "italic"),
            foreground="gray"
        ).grid(row=7, column=0, columnspan=4, sticky="w", padx=5)
        
    def _create_optimization_frame(self, parent):
        """Frame de otimizações RTX 5080."""
        frame = ttk.LabelFrame(parent, text="🚀 Otimizações RTX 5080", padding=15)
        frame.pack(fill=tk.X, pady=8)
        
        # Precision
        prec_frame = ttk.Frame(frame)
        prec_frame.pack(fill=tk.X, pady=5)
        
        prec_label = ttk.Label(prec_frame, text="Precisão:")
        prec_label.pack(side=tk.LEFT)
        create_tooltip(prec_label, "Precisão numérica: BF16 (recomendado RTX 5080, 2x mais rápido que FP32, mantém estabilidade), FP16 (mais rápido, pode ter instabilidade), FP32 (mais lento, máxima precisão).")
        
        precisions = ["bf16", "fp16", "fp32"]
        prec_combo = ttk.Combobox(
            prec_frame,
            textvariable=self.precision_var,
            values=precisions,
            state="readonly",
            width=10
        )
        prec_combo.pack(side=tk.LEFT, padx=10)
        create_tooltip(prec_combo, "BF16 é recomendado para RTX 5080 (Blackwell). Oferece melhor equilíbrio velocidade/precisão. FP32 é mais lento mas mais estável.")
        
        # Checkboxes
        check_frame = ttk.Frame(frame)
        check_frame.pack(fill=tk.X, pady=5)
        
        tf32_check = ttk.Checkbutton(
            check_frame,
            text="TF32 (Tensor Cores)",
            variable=self.use_tf32_var
        )
        tf32_check.pack(side=tk.LEFT, padx=5)
        create_tooltip(tf32_check, "TensorFloat-32: Acelera operações de matriz em Tensor Cores (RTX 5080 suporta). Mantém precisão FP32 mas usa TF32 internamente. Sempre habilitar se disponível.")
        
        if not device_manager.gpu_info.supports_tf32:
            tf32_check.config(state=tk.DISABLED)
        
        compile_check = ttk.Checkbutton(
            check_frame,
            text="torch.compile (PyTorch 2.x)",
            variable=self.use_compile_var
        )
        compile_check.pack(side=tk.LEFT, padx=5)
        create_tooltip(compile_check, "torch.compile: Compila o modelo para acelerar (PyTorch 2.0+). Pode dar 20-30% de ganho, mas primeiro treino é mais lento (compilação). Desabilitado no Windows por padrão.")
        
        if not device_manager.supports_torch_compile():
            compile_check.config(state=tk.DISABLED)
            self.use_compile_var.set(False)  # Desabilitar por padrão se não suportado
        
        # Info
        ttk.Label(
            frame,
            text=f"VRAM Tier: {device_manager.get_vram_tier().upper()} | "
                 f"Precisão ótima: {device_manager.get_optimal_precision().upper()}",
            font=("Segoe UI", 9, "italic")
        ).pack(anchor=tk.W, pady=5)
        
    def _create_dataloader_frame(self, parent):
        """Frame de configuração do DataLoader."""
        frame = ttk.LabelFrame(parent, text="📦 DataLoader", padding=15)
        frame.pack(fill=tk.X, pady=8)
        
        dl_frame = ttk.Frame(frame)
        dl_frame.pack(fill=tk.X)
        
        workers_label = ttk.Label(dl_frame, text="Workers:")
        workers_label.pack(side=tk.LEFT)
        create_tooltip(workers_label, "Número de processos paralelos para carregar dados. 8 é recomendado para RTX 5080. Mais workers = I/O mais rápido, mas usa mais CPU/RAM.")
        
        workers_spin = ttk.Spinbox(
            dl_frame,
            textvariable=self.num_workers_var,
            from_=0, to=16,
            width=5
        )
        workers_spin.pack(side=tk.LEFT, padx=5)
        create_tooltip(workers_spin, "Workers: 0 = main thread (lento), 4-8 = bom equilíbrio, 12-16 = máximo (pode saturar CPU). Use 8 para RTX 5080.")
        
        pin_mem_check = ttk.Checkbutton(
            dl_frame,
            text="Pin Memory",
            variable=self.pin_memory_var
        )
        pin_mem_check.pack(side=tk.LEFT, padx=20)
        create_tooltip(pin_mem_check, "Pin Memory: Mantém dados na RAM paginável, acelerando transferência CPU→GPU. Sempre habilitar se tiver VRAM suficiente. Reduz latência de I/O.")
    
    def _create_antioverfitting_frame(self, parent):
        """Frame Anti-Overfitting."""
        antiover_frame = ttk.LabelFrame(parent, text="🛡️ Anti-Overfitting", padding=15)
        antiover_frame.pack(fill=tk.X, pady=8)
        antiover_frame.columnconfigure(1, weight=1)
        antiover_frame.columnconfigure(3, weight=1)
        
        # Dropout
        dropout_label = ttk.Label(antiover_frame, text="Dropout:")
        dropout_label.grid(row=0, column=0, sticky="w", pady=2)
        create_tooltip(dropout_label, "Dropout: Probabilidade de desativar neurônios aleatoriamente durante treino (regularização). 0.1-0.2 é comum. Aumente para datasets pequenos ou overfitting.")
        
        dropout_spin = ttk.Spinbox(
            antiover_frame,
            textvariable=self.dropout_var,
            from_=0.0, to=0.5,
            increment=0.05,
            width=8,
            format="%.2f"
        )
        dropout_spin.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        create_tooltip(dropout_spin, "Dropout: 0.0 = sem dropout, 0.1 = padrão, 0.2 = mais regularização (objetos pequenos), 0.3+ = muito agressivo.")
        
        # Early Stopping Patience
        early_label = ttk.Label(antiover_frame, text="Early Stop:")
        early_label.grid(row=0, column=2, sticky="w", pady=2)
        create_tooltip(early_label, "Early Stopping Patience: Número de épocas sem melhoria no validation loss antes de parar. 10 é padrão. Reduza para 5-7 se overfitting, aumente para 15-20 se underfitting.")
        
        # Usar Entry sem textvariable para permitir digitação livre
        def validate_early_stop_input(value):
            """Valida entrada - permite apenas dígitos."""
            if value == "":
                return True
            return value.isdigit()
        
        def on_focusout_early_stop(event):
            """Garante valor válido ao perder foco e atualiza variável."""
            try:
                current_text = early_entry.get().strip()
                if not current_text:
                    val = 10  # Valor padrão
                else:
                    val = int(current_text)
                    if val < 3:
                        val = 3
                    elif val > 50:
                        val = 50
                
                # Atualizar Entry e variável
                early_entry.delete(0, tk.END)
                early_entry.insert(0, str(val))
                self.early_stopping_patience_var.set(val)
            except (ValueError, tk.TclError):
                early_entry.delete(0, tk.END)
                early_entry.insert(0, "10")
                self.early_stopping_patience_var.set(10)
        
        def on_key_release_early_stop(event):
            """Atualiza variável em tempo real se valor válido."""
            try:
                current_text = early_entry.get().strip()
                if current_text.isdigit():
                    val = int(current_text)
                    if 3 <= val <= 50:
                        self.early_stopping_patience_var.set(val)
            except (ValueError, tk.TclError):
                pass
        
        # Criar Entry sem textvariable para permitir digitação livre
        vcmd = (antiover_frame.register(validate_early_stop_input), '%P')
        early_entry = ttk.Entry(
            antiover_frame,
            width=8,
            validate="key",
            validatecommand=vcmd
        )
        early_entry.grid(row=0, column=3, sticky="w", padx=5, pady=2)
        
        # Configurar valor inicial
        initial_value = self.early_stopping_patience_var.get()
        early_entry.insert(0, str(initial_value))
        
        # Bind eventos
        early_entry.bind('<FocusOut>', on_focusout_early_stop)
        early_entry.bind('<KeyRelease>', on_key_release_early_stop)
        
        # Guardar referência para leitura direta quando necessário
        self.early_stopping_entry = early_entry
        
        # Adicionar tooltip/info
        ttk.Label(antiover_frame, text="épocas (3-50)").grid(row=0, column=4, sticky="w")
        
        # Label Smoothing
        label_smooth_label = ttk.Label(antiover_frame, text="Label Smoothing:")
        label_smooth_label.grid(row=1, column=0, sticky="w", pady=2)
        create_tooltip(label_smooth_label, "Label Smoothing: Suaviza labels one-hot (ex: [1,0,0] → [0.9,0.05,0.05]). Previne overconfidence. 0.0 = sem smoothing, 0.1 = recomendado para objetos pequenos, 0.2+ = muito agressivo.")
        
        label_smoothing_spin = ttk.Spinbox(
            antiover_frame,
            textvariable=self.label_smoothing_var,
            from_=0.0, to=0.5,
            increment=0.01,
            width=8,
            format="%.2f"
        )
        label_smoothing_spin.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        create_tooltip(label_smoothing_spin, "Label Smoothing: 0.0 = padrão (sem smoothing), 0.05-0.1 = leve regularização, 0.2+ = forte regularização.")
        ttk.Label(antiover_frame, text="(0.0-0.5)").grid(row=1, column=2, sticky="w")
        
        # Augmentation checkbox principal
        aug_check = ttk.Checkbutton(
            antiover_frame,
            text="🔧 Data Augmentation (habilitar/desabilitar todas)",
            variable=self.use_augmentation_var,
            command=self._toggle_augmentation_section
        )
        aug_check.grid(row=2, column=0, columnspan=5, sticky="w", pady=5)
        create_tooltip(aug_check, "Data Augmentation: Habilita/desabilita todas as técnicas de augmentation. Desmarque para desabilitar completamente.")
        
        ttk.Label(
            antiover_frame,
            text="Dropout: regularização no modelo | Early Stop: para treino se val_loss não melhorar | Label Smoothing: suaviza labels",
            font=("Segoe UI", 8, "italic"),
            foreground="gray"
        ).grid(row=3, column=0, columnspan=5, sticky="w")
    
    def _create_augmentation_frame(self, parent):
        """Frame expandido de Data Augmentation com controles detalhados."""
        frame = ttk.LabelFrame(parent, text="🔧 Data Augmentation Avançada", padding=15)
        frame.pack(fill=tk.X, pady=8)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(3, weight=1)
        
        row = 0
        
        # Título e descrição
        desc_label = ttk.Label(
            frame,
            text="Configure técnicas de augmentation state-of-the-art para object detection.\n"
                 "Roboflow já aplica: mosaic e crop. Configure aqui as demais técnicas.",
            font=("Segoe UI", 9, "italic"),
            foreground="gray"
        )
        desc_label.grid(row=row, column=0, columnspan=4, sticky="w", pady=(0, 10))
        row += 1
        
        # ============ SEÇÃO: Augmentações Básicas ============
        basic_label = ttk.Label(frame, text="📌 Augmentações Básicas:", font=("Segoe UI", 9, "bold"))
        basic_label.grid(row=row, column=0, columnspan=4, sticky="w", pady=(5, 5))
        row += 1
        
        # Horizontal Flip
        self._create_augmentation_control(frame, row, "Horizontal Flip:", self.enable_hflip_var, 
                                         self.hflip_prob_var, "Espelhamento horizontal da imagem")
        row += 1
        
        # Vertical Flip
        self._create_augmentation_control(frame, row, "Vertical Flip:", self.enable_vflip_var,
                                         self.vflip_prob_var, "Espelhamento vertical (use com cautela em objetos naturais)")
        row += 1
        
        # Color Jitter
        self._create_augmentation_control(frame, row, "Color Jitter:", self.enable_color_jitter_var,
                                         self.color_jitter_prob_var, "Variação de brilho, contraste, saturação e matiz")
        row += 1
        
        # Gaussian Blur
        self._create_augmentation_control(frame, row, "Gaussian Blur:", self.enable_gaussian_blur_var,
                                         self.gaussian_blur_prob_var, "Desfoque gaussiano para simular foco")
        row += 1
        
        # Separador
        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=4, sticky="ew", pady=10)
        row += 1
        
        # ============ SEÇÃO: Augmentações YOLO-style ============
        yolo_label = ttk.Label(frame, text="🎯 Augmentações YOLO-style (Recomendadas):", font=("Segoe UI", 9, "bold"))
        yolo_label.grid(row=row, column=0, columnspan=4, sticky="w", pady=(5, 5))
        row += 1
        
        # HSV
        self._create_augmentation_control(frame, row, "HSV:", self.enable_hsv_var,
                                         self.hsv_prob_var, "Variação de Hue, Saturation, Value (melhor que Color Jitter)")
        row += 1
        
        # HSV Parameters
        hsv_params_frame = ttk.Frame(frame)
        hsv_params_frame.grid(row=row, column=0, columnspan=4, sticky="w", padx=20, pady=2)
        ttk.Label(hsv_params_frame, text="H:", font=("Segoe UI", 8)).pack(side=tk.LEFT)
        ttk.Entry(hsv_params_frame, textvariable=self.hsv_h_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(hsv_params_frame, text="S:", font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Entry(hsv_params_frame, textvariable=self.hsv_s_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(hsv_params_frame, text="V:", font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Entry(hsv_params_frame, textvariable=self.hsv_v_var, width=6).pack(side=tk.LEFT, padx=2)
        row += 1
        
        # Scale
        self._create_augmentation_control(frame, row, "Scale:", self.enable_scale_var,
                                         self.scale_prob_var, "Redimensionamento aleatório (zoom in/out)")
        row += 1
        
        # Scale Parameters
        scale_params_frame = ttk.Frame(frame)
        scale_params_frame.grid(row=row, column=0, columnspan=4, sticky="w", padx=20, pady=2)
        ttk.Label(scale_params_frame, text="Min:", font=("Segoe UI", 8)).pack(side=tk.LEFT)
        ttk.Entry(scale_params_frame, textvariable=self.scale_range_min_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(scale_params_frame, text="Max:", font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Entry(scale_params_frame, textvariable=self.scale_range_max_var, width=6).pack(side=tk.LEFT, padx=2)
        row += 1
        
        # Translation
        self._create_augmentation_control(frame, row, "Translation:", self.enable_translate_var,
                                         self.translate_prob_var, "Deslocamento horizontal/vertical")
        row += 1
        
        # Translate Parameters
        translate_params_frame = ttk.Frame(frame)
        translate_params_frame.grid(row=row, column=0, columnspan=4, sticky="w", padx=20, pady=2)
        ttk.Label(translate_params_frame, text="Range:", font=("Segoe UI", 8)).pack(side=tk.LEFT)
        ttk.Entry(translate_params_frame, textvariable=self.translate_range_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(translate_params_frame, text="(fração da imagem)", font=("Segoe UI", 8), foreground="gray").pack(side=tk.LEFT, padx=5)
        row += 1
        
        # Shear
        self._create_augmentation_control(frame, row, "Shear:", self.enable_shear_var,
                                         self.shear_prob_var, "Distorção de cisalhamento (use com cautela)")
        row += 1
        
        # Separador
        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=4, sticky="ew", pady=10)
        row += 1
        
        # ============ SEÇÃO: Augmentações Avançadas ============
        advanced_label = ttk.Label(frame, text="🚀 Augmentações Avançadas:", font=("Segoe UI", 9, "bold"))
        advanced_label.grid(row=row, column=0, columnspan=4, sticky="w", pady=(5, 5))
        row += 1
        
        # Mosaic (desabilitado - Roboflow já faz)
        self._create_augmentation_control(frame, row, "Mosaic:", self.enable_mosaic_var,
                                         self.mosaic_prob_var, "⚠️ Desabilite se Roboflow já aplica mosaic")
        row += 1
        
        # MixUp
        self._create_augmentation_control(frame, row, "MixUp:", self.enable_mixup_var,
                                         self.mixup_prob_var, "Combina duas imagens com blend suave")
        row += 1
        
        # CutMix
        self._create_augmentation_control(frame, row, "CutMix:", self.enable_cutmix_var,
                                         self.cutmix_prob_var, "Recorta região de outra imagem e cola")
        row += 1
        
        # Copy-Paste (desabilitado - Roboflow já faz)
        self._create_augmentation_control(frame, row, "Copy-Paste:", self.enable_copy_paste_var,
                                         self.copy_paste_prob_var, "⚠️ Desabilite se Roboflow já aplica")
        row += 1
        
        # Separador
        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=4, sticky="ew", pady=10)
        row += 1
        
        # ============ SEÇÃO: Regularização ============
        reg_label = ttk.Label(frame, text="🛡️ Regularização:", font=("Segoe UI", 9, "bold"))
        reg_label.grid(row=row, column=0, columnspan=4, sticky="w", pady=(5, 5))
        row += 1
        
        # Random Erasing
        self._create_augmentation_control(frame, row, "Random Erasing:", self.enable_random_erasing_var,
                                         self.random_erasing_prob_var, "Apaga regiões aleatórias")
        row += 1
        
        # Motion Blur
        self._create_augmentation_control(frame, row, "Motion Blur:", self.enable_motion_blur_var,
                                         self.motion_blur_prob_var, "Simula movimento da câmera")
        row += 1
        
        # Grid Mask
        self._create_augmentation_control(frame, row, "Grid Mask:", self.enable_grid_mask_var,
                                         self.grid_mask_prob_var, "Remove blocos em grade")
        row += 1
        
        # Cutout
        self._create_augmentation_control(frame, row, "Cutout:", self.enable_cutout_var,
                                         self.cutout_prob_var, "Remove regiões retangulares")
        row += 1
        
        # Nota final
        note_label = ttk.Label(
            frame,
            text="💡 Dica: Para objetos médios/grandes, HSV + Scale + Translation são as mais eficazes.\n"
                 "Desabilite Mosaic e Copy-Paste se Roboflow já aplica estas técnicas.",
            font=("Segoe UI", 8, "italic"),
            foreground="blue"
        )
        note_label.grid(row=row, column=0, columnspan=4, sticky="w", pady=(10, 0))
    
    def _toggle_augmentation_section(self):
        """Mostra/oculta a seção de augmentation baseado no checkbox principal."""
        if self.use_augmentation_var.get():
            self.augmentation_frame_container.pack(fill=tk.X, pady=8)
        else:
            self.augmentation_frame_container.pack_forget()
    
    def _create_augmentation_control(self, parent, row, label_text, enable_var, prob_var, tooltip_text):
        """Cria um controle de augmentation (checkbox + slider de probabilidade)."""
        # Label
        label = ttk.Label(parent, text=label_text)
        label.grid(row=row, column=0, sticky="w", padx=(0, 5), pady=2)
        create_tooltip(label, tooltip_text)
        
        # Checkbox de habilitação
        check = ttk.Checkbutton(parent, variable=enable_var, text="")
        check.grid(row=row, column=1, sticky="w", padx=5, pady=2)
        
        # Label de probabilidade
        prob_label = ttk.Label(parent, text="Prob:")
        prob_label.grid(row=row, column=2, sticky="w", padx=(10, 5), pady=2)
        
        # Spinbox de probabilidade
        prob_spin = ttk.Spinbox(
            parent,
            textvariable=prob_var,
            from_=0.0, to=1.0,
            increment=0.05,
            width=6,
            format="%.2f"
        )
        prob_spin.grid(row=row, column=3, sticky="w", padx=5, pady=2)
        create_tooltip(prob_spin, f"{tooltip_text}\nProbabilidade: 0.0 = nunca aplica, 1.0 = sempre aplica")
        
        # Desabilitar spinbox se augmentation desabilitada
        def update_prob_state(*args):
            if enable_var.get():
                prob_spin.config(state="normal")
            else:
                prob_spin.config(state="disabled")
        
        enable_var.trace_add("write", lambda *args: update_prob_state())
        update_prob_state()
    
    def _create_loss_weights_frame(self, parent):
        """Frame de Loss Weights."""
        frame = ttk.LabelFrame(parent, text="⚖️ Loss Weights", padding=15)
        frame.pack(fill=tk.X, pady=8)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(3, weight=1)
        
        # Bbox Loss Coefficient
        bbox_label = ttk.Label(frame, text="Bbox Loss:")
        bbox_label.grid(row=0, column=0, sticky="w", pady=2)
        create_tooltip(bbox_label, "Bbox Loss Coefficient: Peso da perda L1 para localização de bounding boxes. Valores maiores = mais foco em localização precisa. Padrão: 5.0. Aumente para 8.0 em objetos pequenos.")
        
        bbox_entry = ttk.Entry(frame, textvariable=self.bbox_loss_coefficient_var, width=10)
        bbox_entry.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        create_tooltip(bbox_entry, "Bbox Loss: 5.0 = padrão, 8.0 = objetos pequenos (mais peso em localização), 3.0 = objetos grandes.")
        ttk.Label(frame, text="(localização)").grid(row=0, column=2, sticky="w", padx=5)
        
        # GIoU Loss Coefficient
        giou_label = ttk.Label(frame, text="GIoU Loss:")
        giou_label.grid(row=0, column=3, sticky="w", pady=2)
        create_tooltip(giou_label, "GIoU Loss Coefficient: Peso da perda Generalized IoU (mede sobreposição de boxes). Valores maiores = mais foco em IoU alto. Padrão: 2.0. Aumente para 2.5 em objetos pequenos.")
        
        giou_entry = ttk.Entry(frame, textvariable=self.giou_loss_coefficient_var, width=10)
        giou_entry.grid(row=0, column=4, sticky="w", padx=5, pady=2)
        create_tooltip(giou_entry, "GIoU Loss: 2.0 = padrão, 2.5 = objetos pequenos, 1.5 = objetos grandes.")
        ttk.Label(frame, text="(IoU)").grid(row=0, column=5, sticky="w", padx=5)
        
        # Class Loss Coefficient
        class_label = ttk.Label(frame, text="Class Loss:")
        class_label.grid(row=1, column=0, sticky="w", pady=2)
        create_tooltip(class_label, "Class Loss Coefficient: Peso da perda de classificação (cross-entropy). Padrão: 1.0. Geralmente não precisa alterar.")
        
        class_entry = ttk.Entry(frame, textvariable=self.class_loss_coefficient_var, width=10)
        class_entry.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        create_tooltip(class_entry, "Class Loss: 1.0 = padrão. Mantenha em 1.0 na maioria dos casos.")
        ttk.Label(frame, text="(classificação)").grid(row=1, column=2, sticky="w", padx=5)
        
        # EOS Coefficient
        eos_label = ttk.Label(frame, text="EOS Coeff:")
        eos_label.grid(row=1, column=3, sticky="w", pady=2)
        create_tooltip(eos_label, "EOS (End of Sequence) Coefficient: Peso da perda para tokens 'no object'. Padrão: 0.1. Valores menores = menos penalização para detecções vazias.")
        
        eos_entry = ttk.Entry(frame, textvariable=self.eos_coefficient_var, width=10)
        eos_entry.grid(row=1, column=4, sticky="w", padx=5, pady=2)
        create_tooltip(eos_entry, "EOS Coeff: 0.1 = padrão. Reduza para 0.05 se muitas detecções falsas, aumente para 0.2 se muitas detecções perdidas.")
        ttk.Label(frame, text="(no object)").grid(row=1, column=5, sticky="w", padx=5)
        
        ttk.Label(
            frame,
            text="Ajuste os pesos das funções de perda. Valores maiores = mais importância para essa métrica",
            font=("Segoe UI", 8, "italic"),
            foreground="gray"
        ).grid(row=2, column=0, columnspan=6, sticky="w", pady=5)
        
    def _create_control_frame(self, parent):
        """Frame de controle do treinamento."""
        frame = ttk.LabelFrame(parent, text="🎮 Controle", padding=15)
        frame.pack(fill=tk.X, pady=8)
        
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        if USE_TTKBOOTSTRAP:
            self.start_btn = ttkb.Button(
                btn_frame,
                text="▶️ Iniciar Treinamento",
                bootstyle="success",
                command=self._start_training
            )
            self.stop_btn = ttkb.Button(
                btn_frame,
                text="⏹️ Parar",
                bootstyle="danger",
                command=self._stop_training,
                state=tk.DISABLED
            )
        else:
            self.start_btn = ttk.Button(btn_frame, text="▶️ Iniciar Treinamento", command=self._start_training)
            self.stop_btn = ttk.Button(btn_frame, text="⏹️ Parar", command=self._stop_training, state=tk.DISABLED)
        
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        create_tooltip(self.start_btn, "Inicia o treinamento com os hiperparâmetros configurados. Verifique se o dataset está válido antes de iniciar. O treinamento pode ser interrompido a qualquer momento.")
        create_tooltip(self.stop_btn, "Interrompe o treinamento atual. O modelo será salvo no último checkpoint válido.")
        
        # Status
        status_label = ttk.Label(frame, textvariable=self.status_var, font=("Segoe UI", 10, "bold"))
        status_label.pack(pady=10)
        
    def _create_metrics_frame(self, parent):
        """Frame de métricas em tempo real."""
        frame = ttk.LabelFrame(parent, text="📊 Métricas", padding=15)
        frame.pack(fill=tk.X, pady=8)
        
        metrics = [
            ("epoch", self.epoch_var),
            ("loss", self.loss_var),
            ("lr", self.lr_display_var),
            ("vram", self.vram_display_var),
            ("time", self.time_var)
        ]
        
        for name, var in metrics:
            label = ttk.Label(frame, textvariable=var, font=("Consolas", 10))
            label.pack(anchor=tk.W, pady=2)
        
    def _create_progress_frame(self, parent):
        """Frame de progresso."""
        frame = ttk.LabelFrame(parent, text="📈 Progresso", padding=15)
        frame.pack(fill=tk.X, pady=8)
        
        self.progress_bar = ttk.Progressbar(
            frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate'
        )
        self.progress_bar.pack(fill=tk.X, pady=10)
        
    def _get_early_stopping_value(self):
        """Lê o valor de early stopping diretamente do Entry."""
        try:
            if hasattr(self, 'early_stopping_entry'):
                text = self.early_stopping_entry.get().strip()
                if text:
                    val = int(text)
                    # Garantir que está no range válido
                    if val < 3:
                        return 3
                    elif val > 50:
                        return 50
                    return val
        except (ValueError, AttributeError):
            pass
        # Fallback para variável
        return self.early_stopping_patience_var.get()
    
    def _apply_vram_defaults(self):
        """Aplica defaults baseados na VRAM apenas se não houver config salva."""
        # Se já tem configuração salva, não sobrescrever
        saved = settings_manager.training
        if saved.batch_size != 8 or saved.image_size != 800:
            logger.info("Usando configuracoes salvas do usuario")
            return
        
        tier = device_manager.get_vram_tier()
        defaults = self.config.training.VRAM_THRESHOLDS.get(tier, {})
        
        if defaults:
            self.batch_size_var.set(defaults.get("batch_size", 8))
            self.image_size_var.set(defaults.get("image_size", 800))
            self.grad_accum_var.set(defaults.get("grad_accum", 1))
            
            logger.info(f"Defaults aplicados para VRAM tier '{tier}':")
            logger.info(f"  Batch size: {defaults.get('batch_size')}")
            logger.info(f"  Image size: {defaults.get('image_size')}")
    
    def _setup_auto_save(self):
        """Configura salvamento automático quando configurações mudam."""
        # Adicionar trace para salvar automaticamente
        vars_to_trace = [
            self.model_var, self.epochs_var, self.batch_size_var,
            self.grad_accum_var, self.lr_var, self.lr_backbone_var, self.image_size_var,
            self.scheduler_var, self.weight_decay_var, self.warmup_steps_var, self.max_grad_norm_var,
            self.precision_var, self.use_tf32_var, self.use_compile_var,
            self.num_workers_var,
            # Anti-overfitting
            self.dropout_var, self.early_stopping_patience_var, self.use_augmentation_var,
            self.label_smoothing_var,
            # Augmentation flags - básicas
            self.enable_mosaic_var, self.enable_mixup_var, self.enable_cutmix_var,
            self.enable_copy_paste_var, self.enable_random_erasing_var, self.enable_motion_blur_var,
            self.enable_grid_mask_var, self.enable_cutout_var, self.enable_vflip_var,
            # Augmentation flags - YOLO-style
            self.enable_hsv_var, self.enable_scale_var, self.enable_translate_var, self.enable_shear_var,
            # Augmentation probabilities - básicas
            self.hflip_prob_var, self.vflip_prob_var, self.color_jitter_prob_var, self.gaussian_blur_prob_var,
            # Augmentation probabilities - avançadas
            self.mosaic_prob_var, self.mixup_prob_var, self.cutmix_prob_var, self.copy_paste_prob_var,
            self.random_erasing_prob_var, self.motion_blur_prob_var,
            self.grid_mask_prob_var, self.cutout_prob_var,
            # Augmentation probabilities - YOLO-style
            self.hsv_prob_var, self.scale_prob_var, self.translate_prob_var, self.shear_prob_var,
            # Parâmetros HSV
            self.hsv_h_var, self.hsv_s_var, self.hsv_v_var,
            # Parâmetros geométricos
            self.scale_range_min_var, self.scale_range_max_var, self.translate_range_var, self.shear_range_var,
            # Loss weights
            self.bbox_loss_coefficient_var, self.giou_loss_coefficient_var,
            self.class_loss_coefficient_var, self.eos_coefficient_var
        ]
        
        for var in vars_to_trace:
            var.trace_add("write", lambda *args: self._save_settings_auto())
    
    def _save_settings_auto(self):
        """Salva configurações automaticamente."""
        try:
            aug_active = self.use_augmentation_var.get()
            settings_manager.update_section(
                "training",
                model_name=self.model_var.get(),
                epochs=self.epochs_var.get(),
                batch_size=self.batch_size_var.get(),
                learning_rate=self.lr_var.get(),
                lr_backbone=self.lr_backbone_var.get(),
                image_size=self.image_size_var.get(),
                grad_accum=self.grad_accum_var.get(),
                scheduler=self.scheduler_var.get(),
                weight_decay=self.weight_decay_var.get(),
                warmup_steps=self.warmup_steps_var.get(),
                max_grad_norm=self.max_grad_norm_var.get(),
                precision=self.precision_var.get(),
                use_tf32=self.use_tf32_var.get(),
                use_compile=self.use_compile_var.get(),
                num_workers=self.num_workers_var.get(),
                # Anti-overfitting
                dropout=self.dropout_var.get(),
                early_stopping_patience=self._get_early_stopping_value(),
                use_augmentation=aug_active,
                label_smoothing=self.label_smoothing_var.get(),
                # Augmentation flags - básicas
                enable_mosaic=self.enable_mosaic_var.get() and aug_active,
                enable_mixup=self.enable_mixup_var.get() and aug_active,
                enable_cutmix=self.enable_cutmix_var.get() and aug_active,
                enable_copy_paste=self.enable_copy_paste_var.get() and aug_active,
                enable_random_erasing=self.enable_random_erasing_var.get() and aug_active,
                enable_motion_blur=self.enable_motion_blur_var.get() and aug_active,
                enable_grid_mask=self.enable_grid_mask_var.get() and aug_active,
                enable_cutout=self.enable_cutout_var.get() and aug_active,
                enable_vflip=self.enable_vflip_var.get() and aug_active,
                # Augmentation flags - YOLO-style
                enable_hsv=self.enable_hsv_var.get() and aug_active,
                enable_scale=self.enable_scale_var.get() and aug_active,
                enable_translate=self.enable_translate_var.get() and aug_active,
                enable_shear=self.enable_shear_var.get() and aug_active,
                # Augmentation probabilities - básicas
                hflip_prob=self.hflip_prob_var.get(),
                vflip_prob=self.vflip_prob_var.get(),
                color_jitter_prob=self.color_jitter_prob_var.get(),
                gaussian_blur_prob=self.gaussian_blur_prob_var.get(),
                # Augmentation probabilities - avançadas
                mosaic_prob=self.mosaic_prob_var.get(),
                mixup_prob=self.mixup_prob_var.get(),
                cutmix_prob=self.cutmix_prob_var.get(),
                copy_paste_prob=self.copy_paste_prob_var.get(),
                random_erasing_prob=self.random_erasing_prob_var.get(),
                motion_blur_prob=self.motion_blur_prob_var.get(),
                grid_mask_prob=self.grid_mask_prob_var.get(),
                cutout_prob=self.cutout_prob_var.get(),
                # Augmentation probabilities - YOLO-style
                hsv_prob=self.hsv_prob_var.get(),
                scale_prob=self.scale_prob_var.get(),
                translate_prob=self.translate_prob_var.get(),
                shear_prob=self.shear_prob_var.get(),
                # Parâmetros HSV
                hsv_h=self.hsv_h_var.get(),
                hsv_s=self.hsv_s_var.get(),
                hsv_v=self.hsv_v_var.get(),
                # Parâmetros geométricos
                scale_range_min=self.scale_range_min_var.get(),
                scale_range_max=self.scale_range_max_var.get(),
                translate_range=self.translate_range_var.get(),
                shear_range=self.shear_range_var.get(),
                # Loss weights
                bbox_loss_coefficient=self.bbox_loss_coefficient_var.get(),
                giou_loss_coefficient=self.giou_loss_coefficient_var.get(),
                class_loss_coefficient=self.class_loss_coefficient_var.get(),
                eos_coefficient=self.eos_coefficient_var.get()
            )
        except Exception:
            pass  # Silenciar erros durante trace
    
    def _validate_config(self) -> bool:
        """Valida configuração antes do treino."""
        # Verificar dataset
        if not self.dataset_tab.is_dataset_valid():
            messagebox.showerror("Erro", "Selecione e valide um dataset primeiro!")
            return False
        
        # Verificar nome do experimento
        if not self.experiment_name_var.get().strip():
            messagebox.showerror("Erro", "Digite um nome para o experimento!")
            return False
        
        # Verificar GPU
        if not device_manager.gpu_info.is_available:
            if not messagebox.askyesno(
                "Aviso",
                "GPU não detectada. Treinar na CPU será muito lento.\nDeseja continuar?"
            ):
                return False
        
        return True
    
    def _start_training(self):
        """Inicia o treinamento."""
        if not self._validate_config():
            return
        
        # Diálogo de seleção de modelo DETR ResNet
        model_choice = self._show_model_selection_dialog()
        if model_choice is None:
            return  # Usuário cancelou
        
        # Usar modelo selecionado no diálogo
        if model_choice in self.model_config.AVAILABLE_MODELS:
            model_name = model_choice
            model_id = self.model_config.AVAILABLE_MODELS[model_name]
            # Atualizar variável da UI para refletir a escolha
            self.model_var.set(model_name)
            logger.info(f"Modelo selecionado: {model_name} ({model_id})")
        else:
            # Fallback: usar modelo selecionado na UI
            model_name = self.model_var.get()
            model_id = self.model_config.AVAILABLE_MODELS.get(model_name, self.model_config.DEFAULT_MODEL)
            logger.info(f"Usando modelo da UI: {model_name} ({model_id})")
        
        # Detectar qual preset está sendo usado
        # Objetos pequenos: image_size >= 960 e grad_accum >= 2
        is_small_objects = (
            self.image_size_var.get() >= 960 and 
            self.grad_accum_var.get() >= 2
        )
        
        # Objetos grandes: batch_size >= 12 e image_size <= 800
        is_large_objects = (
            self.batch_size_var.get() >= 12 and 
            self.image_size_var.get() <= 800 and
            self.grad_accum_var.get() == 1
        )
        
        # Preparar kwargs básicos
        experiment_kwargs = {
            "name": self.experiment_name_var.get(),
            "model_name": model_name,
            "model_id": model_id,
            "epochs": self.epochs_var.get(),
            "batch_size": self.batch_size_var.get(),
            "grad_accum": self.grad_accum_var.get(),
            "learning_rate": self.lr_var.get(),
            "lr_backbone": self.lr_backbone_var.get(),
            "image_size": self.image_size_var.get(),
            "scheduler": self.scheduler_var.get(),
            "weight_decay": self.weight_decay_var.get(),
            "warmup_steps": self.warmup_steps_var.get(),
            "max_grad_norm": self.max_grad_norm_var.get(),
            "precision": self.precision_var.get(),
            "use_tf32": self.use_tf32_var.get(),
            "use_torch_compile": self.use_compile_var.get(),
            "num_workers": self.num_workers_var.get(),
            "pin_memory": self.pin_memory_var.get(),
            "dataset_path": str(self.dataset_tab.get_dataset_path()),
            "dropout": self.dropout_var.get(),
            "early_stopping_patience": self._get_early_stopping_value(),
            "use_augmentation": self.use_augmentation_var.get(),
            "label_smoothing": self.label_smoothing_var.get(),
            # Augmentation flags - básicas
            "enable_mosaic": self.enable_mosaic_var.get(),
            "enable_mixup": self.enable_mixup_var.get(),
            "enable_cutmix": self.enable_cutmix_var.get(),
            "enable_copy_paste": self.enable_copy_paste_var.get(),
            "enable_random_erasing": self.enable_random_erasing_var.get(),
            "enable_motion_blur": self.enable_motion_blur_var.get(),
            "enable_grid_mask": self.enable_grid_mask_var.get(),
            "enable_cutout": self.enable_cutout_var.get(),
            "enable_vflip": self.enable_vflip_var.get(),
            # Augmentation flags - YOLO-style
            "enable_hsv": self.enable_hsv_var.get(),
            "enable_scale": self.enable_scale_var.get(),
            "enable_translate": self.enable_translate_var.get(),
            "enable_shear": self.enable_shear_var.get(),
            # Augmentation probabilities - básicas
            "hflip_prob": self.hflip_prob_var.get(),
            "vflip_prob": self.vflip_prob_var.get(),
            "color_jitter_prob": self.color_jitter_prob_var.get(),
            "gaussian_blur_prob": self.gaussian_blur_prob_var.get(),
            # Augmentation probabilities - avançadas
            "mosaic_prob": self.mosaic_prob_var.get(),
            "mixup_prob": self.mixup_prob_var.get(),
            "cutmix_prob": self.cutmix_prob_var.get(),
            "copy_paste_prob": self.copy_paste_prob_var.get(),
            "random_erasing_prob": self.random_erasing_prob_var.get(),
            "motion_blur_prob": self.motion_blur_prob_var.get(),
            "grid_mask_prob": self.grid_mask_prob_var.get(),
            "cutout_prob": self.cutout_prob_var.get(),
            # Augmentation probabilities - YOLO-style
            "hsv_prob": self.hsv_prob_var.get(),
            "scale_prob": self.scale_prob_var.get(),
            "translate_prob": self.translate_prob_var.get(),
            "shear_prob": self.shear_prob_var.get(),
            # Parâmetros HSV
            "hsv_h": self.hsv_h_var.get(),
            "hsv_s": self.hsv_s_var.get(),
            "hsv_v": self.hsv_v_var.get(),
            # Parâmetros geométricos
            "scale_range_min": self.scale_range_min_var.get(),
            "scale_range_max": self.scale_range_max_var.get(),
            "translate_range": self.translate_range_var.get(),
            "shear_range": self.shear_range_var.get(),
            # Loss weights
            "bbox_loss_coefficient": self.bbox_loss_coefficient_var.get(),
            "giou_loss_coefficient": self.giou_loss_coefficient_var.get(),
            "class_loss_coefficient": self.class_loss_coefficient_var.get(),
            "eos_coefficient": self.eos_coefficient_var.get(),
            # Logging e salvamento (conforme especificação técnica)
            "log_steps": 50,
            "eval_steps": 500,
            "save_steps": 1000,
        }
        
        # Log dos loss weights sendo usados
        logger.info(f"Loss weights: bbox={experiment_kwargs['bbox_loss_coefficient']}, "
                   f"giou={experiment_kwargs['giou_loss_coefficient']}, "
                   f"class={experiment_kwargs['class_loss_coefficient']}, "
                   f"eos={experiment_kwargs['eos_coefficient']}")
        
        # Aplicar loss weights otimizados baseado no preset detectado (apenas se não foram alterados manualmente)
        # Se o usuário não alterou os valores padrão, aplicar preset
        default_bbox = 5.0
        default_giou = 2.0
        default_class = 1.0
        default_eos = 0.1
        
        if is_small_objects and (
            abs(self.bbox_loss_coefficient_var.get() - default_bbox) < 0.01 and
            abs(self.giou_loss_coefficient_var.get() - default_giou) < 0.01
        ):
            from src.core.config import SmallObjectTrainingDefaults
            preset = SmallObjectTrainingDefaults()
            experiment_kwargs.update({
                "bbox_loss_coefficient": preset.bbox_loss_coefficient,
                "giou_loss_coefficient": preset.giou_loss_coefficient,
                "class_loss_coefficient": preset.class_loss_coefficient,
                "eos_coefficient": preset.eos_coefficient,
            })
            logger.info("Aplicando loss weights otimizados para objetos pequenos")
        elif is_large_objects and (
            abs(self.bbox_loss_coefficient_var.get() - default_bbox) < 0.01 and
            abs(self.giou_loss_coefficient_var.get() - default_giou) < 0.01
        ):
            from src.core.config import LargeObjectTrainingDefaults
            preset = LargeObjectTrainingDefaults()
            experiment_kwargs.update({
                "bbox_loss_coefficient": preset.bbox_loss_coefficient,
                "giou_loss_coefficient": preset.giou_loss_coefficient,
                "class_loss_coefficient": preset.class_loss_coefficient,
                "eos_coefficient": preset.eos_coefficient,
            })
            logger.info("Aplicando loss weights otimizados para objetos grandes")
        
        experiment = session.create_experiment(**experiment_kwargs)
        
        # Atualizar UI
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("🟡 Iniciando treinamento...")
        
        session.clear_stop_request()
        session.set_training_active(True)
        
        # Iniciar thread de treinamento
        self.training_thread = threading.Thread(
            target=self._training_loop,
            args=(experiment,),
            daemon=True
        )
        self.training_thread.start()
        
        # Iniciar atualização de UI
        self._schedule_ui_update()
    
    def _stop_training(self):
        """Solicita parada do treinamento."""
        session.request_stop()
        self.status_var.set("🟠 Parando treinamento...")
        self.stop_btn.config(state=tk.DISABLED)
        logger.info("Parada do treinamento solicitada")
    
    def _training_loop(self, experiment):
        """Loop principal de treinamento (roda em thread separada)."""
        try:
            from src.models.detr_trainer import DETRTrainer
            
            trainer = DETRTrainer(
                experiment=experiment,
                progress_callback=self._update_progress,
                metrics_callback=self._update_metrics
            )
            
            trainer.train()
            
            if session.is_stop_requested():
                session.update_experiment_status("cancelled")
                self._update_status("⏹️ Treinamento cancelado")
            else:
                session.update_experiment_status("completed")
                self._update_status("✅ Treinamento concluído!")
                
        except Exception as e:
            logger.error(f"Erro no treinamento: {e}")
            session.update_experiment_status("failed")
            self._update_status(f"❌ Erro: {e}")
            
        finally:
            session.set_training_active(False)
            self._training_finished()
    
    def _training_finished(self):
        """Chamado quando treinamento termina."""
        def update():
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.progress_var.set(0)
        
        self.parent.after(0, update)
    
    def _update_progress(self, value: float):
        """Atualiza barra de progresso."""
        def update():
            self.progress_var.set(value)
        self.parent.after(0, update)
    
    def _update_metrics(self, metrics: dict):
        """Atualiza métricas na UI."""
        def update():
            self.epoch_var.set(f"Época: {metrics.get('epoch', 0)}/{metrics.get('total_epochs', 0)}")
            self.loss_var.set(f"Loss: {metrics.get('loss', 0):.4f}")
            self.lr_display_var.set(f"LR: {metrics.get('lr', 0):.2e}")
            self.vram_display_var.set(f"VRAM: {metrics.get('vram_gb', 0):.1f} GB")
            self.time_var.set(f"Tempo/iter: {metrics.get('iter_time_ms', 0):.0f} ms")
        
        self.parent.after(0, update)
    
    def _apply_medium_large_objects_preset(self):
        """Aplica preset otimizado para objetos médios/grandes.
        
        Considera que Roboflow já faz: mosaic, crop, flip, rotação.
        Aqui aplicamos: HSV, Scale, Translation, MixUp, regularização.
        Valores baseados em papers DETR/RT-DETR e YOLO defaults.
        """
        # Modelo recomendado
        self.model_var.set("RT-DETR ResNet-101 [RECOMENDADO]")
        
        # Treinamento
        self.epochs_var.set(50)
        self.batch_size_var.set(8)
        self.grad_accum_var.set(2)  # Batch efetivo = 16
        self.lr_var.set(2e-4)  # LR recomendado para DETR fine-tuning
        self.lr_backbone_var.set(2e-5)  # 10x menor que head (padrão DETR)
        self.image_size_var.set(800)  # Tamanho padrão para objetos médios/grandes
        self.scheduler_var.set("cosine")
        self.weight_decay_var.set(1e-4)
        self.warmup_steps_var.set(500)
        self.max_grad_norm_var.set(0.1)  # Conforme paper DETR
        
        # Workers
        self.num_workers_var.set(8)
        self.precision_var.set("bf16")
        self.use_tf32_var.set(True)
        self.use_compile_var.set(False)
        
        # Anti-overfitting
        self.dropout_var.set(0.1)
        self.early_stopping_patience_var.set(10)
        if hasattr(self, 'early_stopping_entry'):
            self.early_stopping_entry.delete(0, tk.END)
            self.early_stopping_entry.insert(0, "10")
        self.use_augmentation_var.set(True)
        self.label_smoothing_var.set(0.0)
        
        # Loss weights (conforme paper DETR)
        self.bbox_loss_coefficient_var.set(5.0)
        self.giou_loss_coefficient_var.set(2.0)
        self.class_loss_coefficient_var.set(1.0)
        self.eos_coefficient_var.set(0.1)
        
        # Augmentation - Desabilitar Mosaic e Copy-Paste (Roboflow já faz)
        self.enable_mosaic_var.set(False)
        self.mosaic_prob_var.set(0.0)
        self.enable_copy_paste_var.set(False)
        self.copy_paste_prob_var.set(0.0)
        
        # Augmentation - Habilitar YOLO-style (recomendado)
        self.enable_hsv_var.set(True)
        self.hsv_prob_var.set(0.5)
        self.hsv_h_var.set(0.015)  # YOLO default
        self.hsv_s_var.set(0.7)    # YOLO default
        self.hsv_v_var.set(0.4)    # YOLO default
        
        self.enable_scale_var.set(True)
        self.scale_prob_var.set(0.5)
        self.scale_range_min_var.set(0.5)  # YOLO default
        self.scale_range_max_var.set(1.5)
        
        self.enable_translate_var.set(True)
        self.translate_prob_var.set(0.3)
        self.translate_range_var.set(0.1)  # YOLO default
        
        self.enable_shear_var.set(False)
        self.shear_prob_var.set(0.0)
        
        # Augmentation - MixUp (bom para generalização)
        self.enable_mixup_var.set(True)
        self.mixup_prob_var.set(0.15)
        
        self.enable_cutmix_var.set(False)
        self.cutmix_prob_var.set(0.0)
        
        # Augmentation - Regularização
        self.enable_random_erasing_var.set(True)
        self.random_erasing_prob_var.set(0.3)
        
        self.enable_motion_blur_var.set(True)
        self.motion_blur_prob_var.set(0.2)
        
        self.enable_grid_mask_var.set(True)
        self.grid_mask_prob_var.set(0.2)
        
        self.enable_cutout_var.set(True)
        self.cutout_prob_var.set(0.3)
        
        # Augmentation - Básicas
        self.hflip_prob_var.set(0.5)
        self.enable_vflip_var.set(False)
        self.vflip_prob_var.set(0.0)
        self.color_jitter_prob_var.set(0.3)  # Reduzido pois HSV já cobre cor
        self.gaussian_blur_prob_var.set(0.2)
        
        # Toggle augmentation section
        self._toggle_augmentation_section()
        
        logger.info("Preset 'Objetos Médios/Grandes [RECOMENDADO]' aplicado com sucesso!")
        messagebox.showinfo(
            "Preset Aplicado",
            "Configuração otimizada para objetos médios/grandes aplicada!\n\n"
            "Considera que Roboflow já faz: mosaic, crop, flip, rotação.\n\n"
            "• Modelo: RT-DETR ResNet-101 (mAP 54.3)\n"
            "• Image Size: 800px\n"
            "• Batch Size: 8 (efetivo: 16 com grad_accum=2)\n"
            "• Learning Rate: 2e-4 (head), 2e-5 (backbone)\n"
            "• Max Grad Norm: 0.1 (conforme paper DETR)\n"
            "• Augmentation: HSV, Scale, Translation, MixUp\n"
            "• Regularização: Random Erasing, Motion Blur, Grid Mask, Cutout\n"
            "• Mosaic e Copy-Paste: DESABILITADOS (Roboflow já aplica)"
        )
    
    def _apply_small_objects_preset(self):
        """Aplica preset otimizado para objetos retangulares pequenos (20-60px)."""
        from src.core.config import SmallObjectTrainingDefaults
        
        preset = SmallObjectTrainingDefaults()
        
        # Aplicar valores nas variáveis da UI
        self.epochs_var.set(preset.epochs)
        self.batch_size_var.set(preset.batch_size)
        self.grad_accum_var.set(preset.grad_accum)
        self.lr_var.set(preset.learning_rate)
        self.image_size_var.set(preset.image_size)
        self.scheduler_var.set(preset.scheduler)
        self.weight_decay_var.set(preset.weight_decay)
        self.warmup_steps_var.set(preset.warmup_steps)
        self.max_grad_norm_var.set(preset.max_grad_norm)
        self.num_workers_var.set(preset.num_workers)
        self.precision_var.set(preset.precision)
        self.use_tf32_var.set(preset.use_tf32)
        self.dropout_var.set(preset.dropout)
        self.early_stopping_patience_var.set(preset.early_stopping_patience)
        # Atualizar Entry também
        if hasattr(self, 'early_stopping_entry'):
            self.early_stopping_entry.delete(0, tk.END)
            self.early_stopping_entry.insert(0, str(preset.early_stopping_patience))
        self.use_augmentation_var.set(preset.use_augmentation)
        self.label_smoothing_var.set(preset.label_smoothing)
        # Loss weights
        self.bbox_loss_coefficient_var.set(preset.bbox_loss_coefficient)
        self.giou_loss_coefficient_var.set(preset.giou_loss_coefficient)
        self.class_loss_coefficient_var.set(preset.class_loss_coefficient)
        self.eos_coefficient_var.set(preset.eos_coefficient)
        
        # Nota: o auto-save é acionado automaticamente pelos traces das variáveis
        
        logger.info("Preset 'Objetos Retangulares (20-60px)' aplicado com sucesso!")
        messagebox.showinfo(
            "Preset Aplicado",
            "Configuração otimizada para objetos retangulares (20-60px) aplicada!\n\n"
            f"• Image Size: {preset.image_size}px\n"
            f"• Batch Size: {preset.batch_size} (efetivo: {preset.batch_size * preset.grad_accum})\n"
            f"• Learning Rate: {preset.learning_rate}\n"
            f"• Weight Decay: {preset.weight_decay} (anti-overfitting)\n"
            f"• Early Stopping: {preset.early_stopping_patience} épocas\n"
            f"• Dropout: {preset.dropout}\n"
            f"• Augmentation: Ativada"
        )
    
    def _apply_large_objects_preset(self):
        """Aplica preset otimizado para objetos grandes (>100px)."""
        from src.core.config import LargeObjectTrainingDefaults
        
        preset = LargeObjectTrainingDefaults()
        
        # Aplicar valores nas variáveis da UI
        self.epochs_var.set(preset.epochs)
        self.batch_size_var.set(preset.batch_size)
        self.grad_accum_var.set(preset.grad_accum)
        self.lr_var.set(preset.learning_rate)
        self.image_size_var.set(preset.image_size)
        self.scheduler_var.set(preset.scheduler)
        self.weight_decay_var.set(preset.weight_decay)
        self.warmup_steps_var.set(preset.warmup_steps)
        self.max_grad_norm_var.set(preset.max_grad_norm)
        self.num_workers_var.set(preset.num_workers)
        self.precision_var.set(preset.precision)
        self.use_tf32_var.set(preset.use_tf32)
        self.dropout_var.set(preset.dropout)
        self.early_stopping_patience_var.set(preset.early_stopping_patience)
        # Atualizar Entry também
        if hasattr(self, 'early_stopping_entry'):
            self.early_stopping_entry.delete(0, tk.END)
            self.early_stopping_entry.insert(0, str(preset.early_stopping_patience))
        self.use_augmentation_var.set(preset.use_augmentation)
        self.label_smoothing_var.set(preset.label_smoothing)
        # Loss weights
        self.bbox_loss_coefficient_var.set(preset.bbox_loss_coefficient)
        self.giou_loss_coefficient_var.set(preset.giou_loss_coefficient)
        self.class_loss_coefficient_var.set(preset.class_loss_coefficient)
        self.eos_coefficient_var.set(preset.eos_coefficient)
        
        # Nota: o auto-save é acionado automaticamente pelos traces das variáveis
        
        logger.info("Preset 'Objetos Grandes (>100px)' aplicado com sucesso!")
        messagebox.showinfo(
            "Preset Aplicado",
            "Configuração otimizada para objetos grandes (>100px) aplicada!\n\n"
            f"• Image Size: {preset.image_size}px\n"
            f"• Batch Size: {preset.batch_size}\n"
            f"• Learning Rate: {preset.learning_rate}\n"
            f"• Weight Decay: {preset.weight_decay}\n"
            f"• Dropout: {preset.dropout}\n"
            f"• Early Stopping: {preset.early_stopping_patience} épocas"
        )
    
    def _apply_spec_tecnica_preset(self):
        """Aplica preset da Especificação Técnica (configuração testada e validada)."""
        # Configuração conforme especificação técnica
        self.model_var.set("DETR ResNet-50")
        self.epochs_var.set(25)
        self.batch_size_var.set(2)
        self.grad_accum_var.set(4)
        self.lr_var.set(0.0001)  # 1e-4
        self.lr_backbone_var.set(0.00001)  # 1e-5
        self.image_size_var.set(640)
        self.scheduler_var.set("cosine")
        self.weight_decay_var.set(0.0001)
        self.warmup_steps_var.set(500)
        self.max_grad_norm_var.set(0.1)
        self.num_workers_var.set(4)
        self.precision_var.set("fp16")
        self.use_tf32_var.set(True)
        self.use_compile_var.set(False)
        self.dropout_var.set(0.1)
        self.early_stopping_patience_var.set(7)
        if hasattr(self, 'early_stopping_entry'):
            self.early_stopping_entry.delete(0, tk.END)
            self.early_stopping_entry.insert(0, "7")
        self.use_augmentation_var.set(True)
        self.label_smoothing_var.set(0.0)
        # Loss weights conforme especificação
        self.bbox_loss_coefficient_var.set(5.0)
        self.giou_loss_coefficient_var.set(2.0)
        self.class_loss_coefficient_var.set(1.0)
        self.eos_coefficient_var.set(0.1)
        
        logger.info("Preset 'Especificação Técnica' aplicado com sucesso!")
        messagebox.showinfo(
            "Preset Aplicado",
            "Configuração da Especificação Técnica aplicada!\n\n"
            "Esta configuração foi testada e produziu excelentes resultados:\n\n"
            "• Modelo: facebook/detr-resnet-50\n"
            "• Épocas: 25\n"
            "• Batch Size: 2 (efetivo: 8 com grad_accum=4)\n"
            "• Image Size: 640px\n"
            "• Learning Rate: 1e-4 (head), 1e-5 (backbone)\n"
            "• Precision: FP16\n"
            "• Augmentation: Completa (hflip, color, scale, crop)\n"
            "• Early Stopping: 7 épocas\n"
            "• Max Grad Norm: 0.1"
        )
    
    def _show_model_selection_dialog(self):
        """Mostra diálogo para seleção de modelo DETR antes do treinamento."""
        # Detectar se está usando preset de objetos grandes
        is_large_objects = (
            self.batch_size_var.get() >= 12 and 
            self.image_size_var.get() <= 800 and
            self.grad_accum_var.get() == 1
        )
        
        dialog = tk.Toplevel(self.parent)
        dialog.title("Selecionar Modelo DETR")
        dialog.geometry("700x600")  # Tamanho maior para acomodar todos os modelos
        dialog.transient(self.parent)
        dialog.grab_set()
        
        # Centralizar na tela
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        # Variável para resultado
        current_model = self.model_var.get()
        
        # Detectar preset de objetos pequenos
        is_small_objects = (
            self.image_size_var.get() >= 900 and
            self.grad_accum_var.get() >= 2
        )
        
        # Selecionar modelos baseado no preset
        if is_large_objects:
            # Mostrar os 4 melhores modelos para objetos grandes
            available_models = self.model_config.BEST_LARGE_OBJECT_MODELS
            if current_model not in available_models:
                initial_value = "RT-DETR ResNet-101 [RECOMENDADO]"
            else:
                initial_value = current_model
            title_text = "Selecione o Modelo (Top 4 para Objetos Grandes)"
            desc_text = "Modelos otimizados para objetos grandes (>100px):"
        elif is_small_objects:
            # Mostrar os 4 melhores modelos para objetos pequenos
            available_models = self.model_config.BEST_SMALL_OBJECT_MODELS
            if current_model not in available_models:
                initial_value = "RT-DETR ResNet-101 [RECOMENDADO]"
            else:
                initial_value = current_model
            title_text = "Selecione o Modelo (Top 4 para Objetos Pequenos)"
            desc_text = "Modelos otimizados para objetos pequenos (20-60px):"
        else:
            # Mostrar todos os modelos disponíveis (ordenados por precisao)
            available_models = self.model_config.AVAILABLE_MODELS
            if current_model not in available_models:
                initial_value = "RT-DETR ResNet-101 [RECOMENDADO]"
            else:
                initial_value = current_model
            title_text = "Selecione o Modelo DETR para Treinamento"
            desc_text = "Modelos ordenados por precisao (mAP). RT-DETR e DINO sao State-of-the-Art:"
        
        selected_model = tk.StringVar(value=initial_value)
        result = [initial_value]
        
        # Frame principal
        main_frame = ttk.Frame(dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        ttk.Label(
            main_frame,
            text=title_text,
            font=("Segoe UI", 12, "bold")
        ).pack(pady=(20, 10))
        
        # Descrição
        ttk.Label(
            main_frame,
            text=desc_text,
            font=("Segoe UI", 9),
            wraplength=550
        ).pack(pady=(0, 20))
        
        # Container com scroll para lista de modelos
        canvas_container = ttk.Frame(main_frame)
        canvas_container.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(canvas_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_container, orient="vertical", command=canvas.yview)
        
        models_frame = ttk.Frame(canvas)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        canvas_window = canvas.create_window((0, 0), window=models_frame, anchor="nw")
        
        def configure_scroll(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(canvas_window, width=event.width)
        
        models_frame.bind("<Configure>", configure_scroll)
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))
        
        # Bind mouse wheel (com proteção contra canvas destruído)
        def on_mousewheel(event):
            try:
                if canvas.winfo_exists():
                    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except Exception:
                pass  # Ignora se o canvas foi destruído
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # Informações detalhadas dos modelos (ordenados por qualidade)
        all_models_info = {
            # === TIER 1: State-of-the-Art (RT-DETR) ===
            "RT-DETR ResNet-101 [RECOMENDADO]": {
                "desc": "State-of-the-Art! mAP 54.3, convergencia ultra-rapida (~12 epochs)",
                "speed": "Muito rapido",
                "vram": "~12GB",
                "best_for": "MAXIMA PRECISAO - Recomendado para todos os casos"
            },
            "RT-DETR ResNet-50": {
                "desc": "State-of-the-Art! mAP 53.1, eficiente em VRAM",
                "speed": "Ultra rapido",
                "vram": "~8GB",
                "best_for": "Alta precisao com menor uso de VRAM"
            },
            # === TIER 2: Alta Precisao ===
            "Deformable DETR": {
                "desc": "Deformable attention, mAP 46.2, multi-escala",
                "speed": "Rapido",
                "vram": "~9GB",
                "best_for": "Objetos pequenos e variaveis"
            },
            "DAB-DETR ResNet-50": {
                "desc": "Dynamic Anchor Boxes, mAP 45.7, muito estavel",
                "speed": "Rapido",
                "vram": "~8GB",
                "best_for": "Equilibrio precisao/estabilidade"
            },
            "Conditional DETR ResNet-50": {
                "desc": "Convergencia 6-10x mais rapida, mAP 43.8",
                "speed": "Muito rapido",
                "vram": "~8GB",
                "best_for": "Treinamento rapido, boa precisao"
            },
            # === TIER 3: Classicos ===
            "DETR ResNet-101": {
                "desc": "Modelo classico robusto, mAP 44.9",
                "speed": "Lento",
                "vram": "~12GB",
                "best_for": "Objetos grandes, muito estavel"
            },
            "DETR ResNet-50": {
                "desc": "Modelo base original, mAP 42.0",
                "speed": "Moderado",
                "vram": "~8GB",
                "best_for": "Baseline, menor VRAM"
            }
        }
        
        # Filtrar modelos baseado no preset
        models_info = {k: all_models_info.get(k, {
            "desc": "Modelo DETR",
            "speed": "⚡",
            "vram": "💾",
            "best_for": "Uso geral"
        }) for k in available_models.keys()}
        
        for i, (model_name, info) in enumerate(models_info.items()):
            frame = ttk.LabelFrame(models_frame, text=model_name, padding=10)
            frame.pack(fill=tk.X, pady=5, padx=10)
            
            # Radio button
            radio = ttk.Radiobutton(
                frame,
                text="Selecionar",
                variable=selected_model,
                value=model_name,
                command=lambda m=model_name: result.__setitem__(0, m)
            )
            radio.grid(row=0, column=0, sticky="w", padx=5)
            
            # Info
            info_frame = ttk.Frame(frame)
            info_frame.grid(row=0, column=1, sticky="w", padx=10)
            
            ttk.Label(
                info_frame,
                text=info["desc"],
                font=("Segoe UI", 8),
                foreground="gray"
            ).pack(anchor=tk.W)
            
            ttk.Label(
                info_frame,
                text=f"{info['speed']} | {info['vram']}",
                font=("Segoe UI", 8),
                foreground="blue"
            ).pack(anchor=tk.W)
            
            ttk.Label(
                info_frame,
                text=f"Melhor para: {info['best_for']}",
                font=("Segoe UI", 8, "italic"),
                foreground="green"
            ).pack(anchor=tk.W)
        
        # Botões
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(pady=20)
        
        def on_ok():
            result[0] = selected_model.get()
            dialog.destroy()
        
        def on_cancel():
            result[0] = None
            dialog.destroy()
        
        if USE_TTKBOOTSTRAP:
            ttkb.Button(
                buttons_frame,
                text="Confirmar",
                bootstyle="primary",
                command=on_ok
            ).pack(side=tk.LEFT, padx=5)
            ttkb.Button(
                buttons_frame,
                text="Cancelar",
                bootstyle="secondary",
                command=on_cancel
            ).pack(side=tk.LEFT, padx=5)
        else:
            ttk.Button(buttons_frame, text="Confirmar", command=on_ok).pack(side=tk.LEFT, padx=5)
            ttk.Button(buttons_frame, text="Cancelar", command=on_cancel).pack(side=tk.LEFT, padx=5)
        
        # Aguardar fechamento do diálogo
        dialog.wait_window()
        
        return result[0]
    
    def _update_status(self, status: str):
        """Atualiza status de forma thread-safe."""
        def update():
            self.status_var.set(status)
        self.parent.after(0, update)
    
    def _schedule_ui_update(self):
        """Agenda atualizações periódicas da UI durante treino."""
        if session.is_training_active():
            # Atualizar VRAM
            if device_manager.gpu_info.is_available:
                _, used, _ = device_manager.get_vram_status()
                self.vram_display_var.set(f"VRAM: {used:.1f} GB")
            
            self.parent.after(1000, self._schedule_ui_update)
