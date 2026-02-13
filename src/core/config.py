"""
Configurações globais e paths do sistema.
Sistema DETR - RTX 5080 Optimized
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List


# Base path do projeto
BASE_DIR = Path(__file__).parent.parent.parent.resolve()


@dataclass
class Paths:
    """Estrutura de diretórios do projeto."""
    base: Path = BASE_DIR
    
    # Data directories
    data: Path = field(default_factory=lambda: BASE_DIR / "data")
    datasets: Path = field(default_factory=lambda: BASE_DIR / "data" / "datasets")
    models: Path = field(default_factory=lambda: BASE_DIR / "data" / "models")
    experiments: Path = field(default_factory=lambda: BASE_DIR / "data" / "models" / "experiments")
    outputs: Path = field(default_factory=lambda: BASE_DIR / "data" / "outputs")
    predictions: Path = field(default_factory=lambda: BASE_DIR / "data" / "outputs" / "predictions")
    logs: Path = field(default_factory=lambda: BASE_DIR / "data" / "logs")
    reports: Path = field(default_factory=lambda: BASE_DIR / "data" / "reports")
    
    # TensorBoard runs
    runs: Path = field(default_factory=lambda: BASE_DIR / "runs")
    
    def create_all(self):
        """Cria todos os diretórios necessários."""
        for attr_name in dir(self):
            if not attr_name.startswith('_'):
                attr = getattr(self, attr_name)
                if isinstance(attr, Path) and attr_name != 'base':
                    attr.mkdir(parents=True, exist_ok=True)


# Instância global de paths
PATHS = Paths()


@dataclass
class ModelConfig:
    """Configuração dos modelos DETR disponíveis.
    
    Organizados por qualidade de treinamento (precisão, recall, estabilidade).
    Otimizados para RTX 5080 (16GB VRAM).
    
    Nota: Apenas modelos públicos do Hugging Face são incluídos.
    """
    
    # ========== TODOS OS MODELOS DISPONÍVEIS ==========
    # Ordenados por prioridade de treinamento (mAP, estabilidade, convergência)
    # Nota: DINO original não está disponível no HuggingFace Transformers
    AVAILABLE_MODELS: Dict[str, str] = field(default_factory=lambda: {
        # === TIER 1: State-of-the-Art (Máxima Precisão) ===
        "RT-DETR ResNet-101 [RECOMENDADO]": "PekingU/rtdetr_r101vd",      # mAP 54.3, treino rápido
        "RT-DETR ResNet-50": "PekingU/rtdetr_r50vd",                       # mAP 53.1, eficiente
        
        # === TIER 2: Alta Precisão (Melhorias sobre DETR) ===
        "Deformable DETR + Box Refine": "SenseTime/deformable-detr-with-box-refine",  # mAP 46.8, refinamento iterativo
        "Deformable DETR Two-Stage": "SenseTime/deformable-detr-with-box-refine-two-stage",  # mAP 47.0, two-stage
        "Deformable DETR": "SenseTime/deformable-detr",                    # mAP 46.2, objetos pequenos
        "DAB-DETR ResNet-50 DC5": "IDEA-Research/dab-detr-resnet-50-dc5",  # mAP 45.8, dilated C5
        "DAB-DETR ResNet-50": "IDEA-Research/dab-detr-resnet-50",          # mAP 45.7, estável
        "Conditional DETR ResNet-50": "microsoft/conditional-detr-resnet-50",  # mAP 43.8, rápida convergência
        
        # === TIER 3: Clássicos (Muito Estáveis) ===
        "DETR ResNet-101 DC5": "facebook/detr-resnet-101-dc5",             # mAP 44.9, dilated backbone
        "DETR ResNet-101": "facebook/detr-resnet-101",                     # mAP 44.9, robusto
        "DETR ResNet-50 DC5": "facebook/detr-resnet-50-dc5",               # mAP 43.3, dilated backbone
        "DETR ResNet-50": "facebook/detr-resnet-50",                       # mAP 42.0, baseline (Especificação Técnica)
    })
    
    # ========== MODELOS RECOMENDADOS POR CASO DE USO ==========
    
    # Top 5 para máxima precisão
    BEST_PRECISION_MODELS: Dict[str, str] = field(default_factory=lambda: {
        "RT-DETR ResNet-101 [RECOMENDADO]": "PekingU/rtdetr_r101vd",
        "RT-DETR ResNet-50": "PekingU/rtdetr_r50vd",
        "Deformable DETR Two-Stage": "SenseTime/deformable-detr-with-box-refine-two-stage",
        "Deformable DETR + Box Refine": "SenseTime/deformable-detr-with-box-refine",
        "DAB-DETR ResNet-50 DC5": "IDEA-Research/dab-detr-resnet-50-dc5",
    })
    
    # Top 5 para objetos grandes (>100px)
    BEST_LARGE_OBJECT_MODELS: Dict[str, str] = field(default_factory=lambda: {
        "RT-DETR ResNet-101 [RECOMENDADO]": "PekingU/rtdetr_r101vd",
        "DETR ResNet-101 DC5": "facebook/detr-resnet-101-dc5",
        "DETR ResNet-101": "facebook/detr-resnet-101",
        "RT-DETR ResNet-50": "PekingU/rtdetr_r50vd",
        "DAB-DETR ResNet-50 DC5": "IDEA-Research/dab-detr-resnet-50-dc5",
    })
    
    # Top 5 para objetos pequenos (20-60px)
    BEST_SMALL_OBJECT_MODELS: Dict[str, str] = field(default_factory=lambda: {
        "RT-DETR ResNet-101 [RECOMENDADO]": "PekingU/rtdetr_r101vd",
        "Deformable DETR Two-Stage": "SenseTime/deformable-detr-with-box-refine-two-stage",
        "Deformable DETR + Box Refine": "SenseTime/deformable-detr-with-box-refine",
        "Deformable DETR": "SenseTime/deformable-detr",
        "RT-DETR ResNet-50": "PekingU/rtdetr_r50vd",
    })
    
    # Top 5 para estabilidade de treinamento
    BEST_STABLE_MODELS: Dict[str, str] = field(default_factory=lambda: {
        "DAB-DETR ResNet-50": "IDEA-Research/dab-detr-resnet-50",
        "DAB-DETR ResNet-50 DC5": "IDEA-Research/dab-detr-resnet-50-dc5",
        "DETR ResNet-101": "facebook/detr-resnet-101",
        "Conditional DETR ResNet-50": "microsoft/conditional-detr-resnet-50",
        "DETR ResNet-50": "facebook/detr-resnet-50",
    })
    
    # Modelo padrão (melhor equilíbrio precisão/estabilidade/VRAM)
    DEFAULT_MODEL: str = "PekingU/rtdetr_r101vd"
    
    # Classes fixas para detecção (conforme dataset)
    FIXED_CLASSES: List[str] = field(default_factory=lambda: [
        "ok",
        "ref", 
        "vazio"
    ])
    
    # Número de classes (incluindo background para alguns modelos)
    NUM_CLASSES: int = 3


@dataclass
class RoboflowDefaults:
    """Configurações padrão do Roboflow."""
    # URL direta do dataset
    DATASET_URL: str = "https://app.roboflow.com/ds/28uCRbc4wr?key=q7qHpXbRBo"
    API_KEY: str = "q7qHpXbRBo"
    VERSION: str = "5"


@dataclass
class TrainingDefaults:
    """Defaults de treinamento otimizados para RTX 5080."""
    epochs: int = 50
    batch_size: int = 8  # Auto-ajuste por VRAM
    grad_accum: int = 1
    learning_rate: float = 1e-5
    image_size: int = 800  # Auto-ajuste por VRAM
    scheduler: str = "cosine"
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    
    # Mixed precision
    precision: str = "bf16"  # bf16 preferido, fallback fp16
    use_tf32: bool = True
    use_torch_compile: bool = False  # Desabilitado por padrão (Triton não suporta Windows)
    
    # DataLoader
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    cudnn_benchmark: bool = True
    
    # Anti-overfitting
    dropout: float = 0.1
    early_stopping_patience: int = 10
    label_smoothing: float = 0.0
    
    # VRAM thresholds para auto-tuning (em GB)
    VRAM_THRESHOLDS: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "high": {  # >= 16GB
            "batch_size": 16,
            "image_size": 800,
            "grad_accum": 1
        },
        "medium": {  # 8-16GB
            "batch_size": 8,
            "image_size": 800,
            "grad_accum": 2
        },
        "low": {  # < 8GB
            "batch_size": 4,
            "image_size": 640,
            "grad_accum": 4
        }
    })


@dataclass
class SmallObjectTrainingDefaults:
    """Defaults otimizados para objetos retangulares pequenos (20-60px) em RTX 5080."""
    # Treinamento básico
    epochs: int = 100  # Mais épocas para objetos pequenos
    batch_size: int = 8  # Batch size otimizado para objetos 20-60px
    grad_accum: int = 2  # Batch efetivo = 16
    learning_rate: float = 5e-6  # LR menor para treinamento mais estável
    image_size: int = 960  # Resolução otimizada para objetos 20-60px
    scheduler: str = "cosine"
    weight_decay: float = 5e-4  # Aumentado para regularização
    warmup_steps: int = 1000  # Warmup mais longo
    max_grad_norm: float = 0.5  # Gradient clipping mais agressivo
    
    # Mixed precision
    precision: str = "bf16"
    use_tf32: bool = True
    use_torch_compile: bool = False
    
    # DataLoader
    num_workers: int = 10  # Mais workers para I/O
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 3  # Mais prefetch
    cudnn_benchmark: bool = True
    
    # Anti-overfitting (agressivo)
    dropout: float = 0.2  # Dropout aumentado
    early_stopping_patience: int = 7  # Early stopping mais agressivo
    label_smoothing: float = 0.1  # Label smoothing para regularização
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation_hflip_prob: float = 0.5
    augmentation_color_jitter_prob: float = 0.6
    augmentation_gaussian_blur_prob: float = 0.3
    augmentation_cutout_prob: float = 0.3
    augmentation_cutout_size: tuple = (20, 60)  # Tamanho de cutout ajustado para objetos 20-60px
    
    # Loss weights (ajustados para objetos retangulares 20-60px)
    bbox_loss_coefficient: float = 8.0  # Aumentado para melhor localização
    giou_loss_coefficient: float = 2.5  # Aumentado para melhor IoU
    class_loss_coefficient: float = 1.0
    eos_coefficient: float = 0.1


@dataclass
class LargeObjectTrainingDefaults:
    """Defaults otimizados para objetos grandes (>100px) em RTX 5080."""
    # Treinamento básico
    epochs: int = 50  # Menos épocas necessárias para objetos grandes
    batch_size: int = 12  # Batch size maior possível para objetos grandes
    grad_accum: int = 1  # Não precisa de acumulação (batch já é grande)
    learning_rate: float = 1e-5  # LR padrão adequado para objetos grandes
    image_size: int = 800  # Resolução padrão suficiente para objetos grandes
    scheduler: str = "cosine"
    weight_decay: float = 1e-4  # Regularização padrão
    warmup_steps: int = 500  # Warmup padrão
    max_grad_norm: float = 1.0  # Gradient clipping padrão
    
    # Mixed precision
    precision: str = "bf16"
    use_tf32: bool = True
    use_torch_compile: bool = False
    
    # DataLoader
    num_workers: int = 8  # Workers padrão
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2  # Prefetch padrão
    cudnn_benchmark: bool = True
    
    # Anti-overfitting (moderado para objetos grandes)
    dropout: float = 0.1  # Dropout padrão
    early_stopping_patience: int = 10  # Early stopping padrão
    label_smoothing: float = 0.0  # Sem label smoothing para objetos grandes
    
    # Data augmentation (moderada)
    use_augmentation: bool = True
    augmentation_hflip_prob: float = 0.5
    augmentation_color_jitter_prob: float = 0.4  # Menos agressivo
    augmentation_gaussian_blur_prob: float = 0.2  # Menos blur
    augmentation_cutout_prob: float = 0.2  # Menos cutout
    augmentation_cutout_size: tuple = (50, 150)  # Cutout maior para objetos grandes
    
    # Loss weights (ajustados para objetos grandes - menos peso em localização)
    bbox_loss_coefficient: float = 5.0  # Padrão para objetos grandes
    giou_loss_coefficient: float = 2.0  # Padrão para objetos grandes
    class_loss_coefficient: float = 1.0
    eos_coefficient: float = 0.1


@dataclass
class InferenceDefaults:
    """Defaults de inferência."""
    threshold: float = 0.05  # 5%
    preview_fps_limit: int = 30
    fast_mode_image_size: int = 640


class Config:
    """Configuração central do sistema."""
    
    def __init__(self):
        self.paths = PATHS
        self.model = ModelConfig()
        self.training = TrainingDefaults()
        self.inference = InferenceDefaults()
        self.roboflow = RoboflowDefaults()
        
        # Criar diretórios
        self.paths.create_all()
    
    def get_experiment_dir(self, experiment_name: str) -> Path:
        """Retorna o diretório de um experimento específico."""
        exp_dir = self.paths.experiments / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte configuração para dicionário."""
        return {
            "paths": {k: str(v) for k, v in vars(self.paths).items() if isinstance(v, Path)},
            "model": vars(self.model),
            "training": vars(self.training),
            "inference": vars(self.inference)
        }


# Instância global
config = Config()
