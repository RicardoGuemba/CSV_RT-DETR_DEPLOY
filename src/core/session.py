"""
Gerenciamento de sessão e experimentos.
Sistema DETR - RTX 5080 Optimized
"""
import json
import shutil
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
import threading

from .config import PATHS, Config
from .logger import logger
from .device_manager import device_manager


@dataclass
class ExperimentConfig:
    """Configuração de um experimento."""
    name: str
    model_name: str
    model_id: str
    
    # Hiperparâmetros
    epochs: int = 50
    batch_size: int = 8
    grad_accum: int = 1
    learning_rate: float = 1e-5
    lr_backbone: float = 1e-5  # Learning rate separado para backbone (10x menor que head)
    image_size: int = 800
    scheduler: str = "cosine"
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    
    # Otimizações
    precision: str = "bf16"
    use_tf32: bool = True
    use_torch_compile: bool = True
    
    # DataLoader
    num_workers: int = 8
    pin_memory: bool = True
    
    # Dataset
    dataset_path: str = ""
    
    # Anti-overfitting
    dropout: float = 0.1
    early_stopping_patience: int = 10
    use_augmentation: bool = False
    label_smoothing: float = 0.0
    
    # Data Augmentation - Flags de habilitação (básicas)
    enable_mosaic: bool = False  # Desabilitado: Roboflow já faz mosaic
    enable_mixup: bool = True
    enable_cutmix: bool = False  # CutMix desabilitado por padrão
    enable_copy_paste: bool = False  # Desabilitado: Roboflow já faz
    enable_random_erasing: bool = True
    enable_motion_blur: bool = True
    enable_grid_mask: bool = True
    enable_cutout: bool = True
    
    # Data Augmentation - Flags YOLO-style (desabilitadas por padrão)
    enable_hsv: bool = True  # HSV habilitado (boa para generalização)
    enable_scale: bool = True  # Scale habilitado
    enable_translate: bool = True  # Translation habilitado
    enable_shear: bool = False  # Shear desabilitado (pode distorcer objetos)
    enable_vflip: bool = False  # Vertical flip desabilitado (objetos naturais)
    
    # Data Augmentation - Probabilidades básicas
    hflip_prob: float = 0.5
    vflip_prob: float = 0.0
    color_jitter_prob: float = 0.5
    gaussian_blur_prob: float = 0.3
    
    # Data Augmentation - Probabilidades avançadas
    mosaic_prob: float = 0.0  # Desabilitado: Roboflow já faz
    mixup_prob: float = 0.15  # Reduzido (já tem outras augmentações)
    cutmix_prob: float = 0.0
    copy_paste_prob: float = 0.0  # Desabilitado: Roboflow já faz
    random_erasing_prob: float = 0.3
    motion_blur_prob: float = 0.2
    grid_mask_prob: float = 0.2
    cutout_prob: float = 0.3
    
    # Data Augmentation - Probabilidades YOLO-style
    hsv_prob: float = 0.5  # HSV com 50% prob
    scale_prob: float = 0.5  # Scale com 50% prob
    translate_prob: float = 0.3  # Translation com 30% prob
    shear_prob: float = 0.0
    
    # Parâmetros HSV (YOLO defaults)
    hsv_h: float = 0.015  # Hue variation
    hsv_s: float = 0.7    # Saturation variation
    hsv_v: float = 0.4    # Value variation
    
    # Parâmetros geométricos
    scale_range_min: float = 0.5
    scale_range_max: float = 1.5
    translate_range: float = 0.1  # Fração da imagem
    shear_range: float = 0.0  # Graus
    
    # Loss weights (para melhor acurácia, especialmente objetos pequenos)
    bbox_loss_coefficient: float = 5.0
    giou_loss_coefficient: float = 2.0
    class_loss_coefficient: float = 1.0
    eos_coefficient: float = 0.1
    
    # Logging e salvamento
    log_steps: int = 50  # Intervalo de logging (conforme especificação técnica)
    eval_steps: int = 500  # Intervalo de avaliação
    save_steps: int = 1000  # Intervalo de salvamento
    
    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Status
    status: str = "created"  # created, training, completed, failed, cancelled


@dataclass
class TrainingMetrics:
    """Métricas de treinamento."""
    epoch: int = 0
    step: int = 0
    loss: float = 0.0
    loss_ce: float = 0.0
    loss_bbox: float = 0.0
    loss_giou: float = 0.0
    learning_rate: float = 0.0
    vram_allocated_gb: float = 0.0
    vram_reserved_gb: float = 0.0
    iter_time_ms: float = 0.0
    
    # Validation metrics
    val_loss: Optional[float] = None
    mAP: Optional[float] = None
    AP50: Optional[float] = None
    AP75: Optional[float] = None


class Session:
    """Gerencia a sessão atual e experimentos."""
    
    _instance: Optional['Session'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.current_experiment: Optional[ExperimentConfig] = None
        self.training_active = threading.Event()
        self.stop_requested = threading.Event()
        self._config = Config()
        
        logger.info("Session manager inicializado")
    
    def create_experiment(self, name: str, model_name: str, model_id: str, 
                         **kwargs) -> ExperimentConfig:
        """Cria um novo experimento.
        
        O nome do experimento segue o formato: {nome_modelo}_{data}
        Ex: RTDETR_ResNet101_20260124_143052
        """
        # Gerar nome único baseado no nome do modelo e data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Extrair nome curto do modelo para usar no nome do experimento
        # Remove caracteres especiais e espaços
        model_short_name = model_name.replace("[RECOMENDADO]", "").strip()
        model_short_name = model_short_name.replace(" ", "_").replace("-", "_")
        model_short_name = model_short_name.replace("__", "_").strip("_")
        
        unique_name = f"{model_short_name}_{timestamp}"
        
        # Criar diretório do experimento
        exp_dir = PATHS.experiments / unique_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Aplicar defaults baseados em VRAM
        vram_tier = device_manager.get_vram_tier()
        defaults = self._config.training.VRAM_THRESHOLDS.get(vram_tier, {})
        
        # Criar configuração
        config_data = {
            "name": unique_name,
            "model_name": model_name,
            "model_id": model_id,
            "batch_size": kwargs.get("batch_size", defaults.get("batch_size", 8)),
            "image_size": kwargs.get("image_size", defaults.get("image_size", 800)),
            "grad_accum": kwargs.get("grad_accum", defaults.get("grad_accum", 1)),
            **{k: v for k, v in kwargs.items() if k not in ["batch_size", "image_size", "grad_accum"]}
        }
        
        experiment = ExperimentConfig(**config_data)
        
        # Salvar configuração
        self._save_experiment_config(experiment, exp_dir)
        
        # Salvar hardware environment
        device_manager.save_hardware_env(exp_dir / "hardware_env.json")
        
        self.current_experiment = experiment
        logger.info(f"Experimento criado: {unique_name}")
        logger.info(f"  VRAM tier: {vram_tier}")
        logger.info(f"  Batch size: {experiment.batch_size}")
        logger.info(f"  Image size: {experiment.image_size}")
        
        return experiment
    
    def _save_experiment_config(self, experiment: ExperimentConfig, exp_dir: Path):
        """Salva configuração do experimento."""
        config_path = exp_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(experiment), f, indent=2, ensure_ascii=False)
    
    def load_experiment(self, name: str) -> Optional[ExperimentConfig]:
        """Carrega um experimento existente."""
        exp_dir = PATHS.experiments / name
        config_path = exp_dir / "config.json"
        
        if not config_path.exists():
            logger.error(f"Experimento não encontrado: {name}")
            return None
        
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        experiment = ExperimentConfig(**data)
        self.current_experiment = experiment
        logger.info(f"Experimento carregado: {name}")
        
        return experiment
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """Lista todos os experimentos."""
        experiments = []
        
        if not PATHS.experiments.exists():
            return experiments
        
        for exp_dir in PATHS.experiments.iterdir():
            if exp_dir.is_dir():
                config_path = exp_dir / "config.json"
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    experiments.append({
                        "name": data.get("name", exp_dir.name),
                        "model": data.get("model_name", "N/A"),
                        "status": data.get("status", "unknown"),
                        "created": data.get("created_at", "N/A"),
                        "path": str(exp_dir)
                    })
        
        # Ordenar por data de criação (mais recente primeiro)
        experiments.sort(key=lambda x: x["created"], reverse=True)
        
        return experiments
    
    def delete_experiment(self, name: str) -> bool:
        """Deleta um experimento."""
        exp_dir = PATHS.experiments / name
        
        if not exp_dir.exists():
            logger.error(f"Experimento não encontrado: {name}")
            return False
        
        try:
            shutil.rmtree(exp_dir)
            logger.info(f"Experimento deletado: {name}")
            
            if self.current_experiment and self.current_experiment.name == name:
                self.current_experiment = None
            
            return True
        except PermissionError as e:
            logger.error(f"Erro de permissão ao deletar: {e}")
            return False
        except Exception as e:
            logger.error(f"Erro ao deletar experimento: {e}")
            return False
    
    def update_experiment_status(self, status: str):
        """Atualiza o status do experimento atual."""
        if self.current_experiment is None:
            return
        
        self.current_experiment.status = status
        self.current_experiment.updated_at = datetime.now().isoformat()
        
        exp_dir = PATHS.experiments / self.current_experiment.name
        self._save_experiment_config(self.current_experiment, exp_dir)
        
        logger.info(f"Status do experimento atualizado: {status}")
    
    def get_experiment_dir(self) -> Optional[Path]:
        """Retorna o diretório do experimento atual."""
        if self.current_experiment is None:
            return None
        return PATHS.experiments / self.current_experiment.name
    
    def request_stop(self):
        """Solicita parada do treinamento."""
        self.stop_requested.set()
        logger.info("Parada solicitada")
    
    def clear_stop_request(self):
        """Limpa a solicitação de parada."""
        self.stop_requested.clear()
    
    def is_stop_requested(self) -> bool:
        """Verifica se parada foi solicitada."""
        return self.stop_requested.is_set()
    
    def set_training_active(self, active: bool):
        """Define se treinamento está ativo."""
        if active:
            self.training_active.set()
        else:
            self.training_active.clear()
    
    def is_training_active(self) -> bool:
        """Verifica se treinamento está ativo."""
        return self.training_active.is_set()


# Instância global
session = Session()
