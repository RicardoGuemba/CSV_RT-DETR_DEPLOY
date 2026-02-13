"""
Gerenciamento de configurações do usuário.
Persiste configurações entre sessões.
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict
import threading

from .logger import logger


# Caminho padrão para arquivo de configurações
DEFAULT_SETTINGS_PATH = Path("config/user_settings.json")


@dataclass
class DatasetSettings:
    """Configurações da aba Dataset."""
    dataset_path: str = ""
    roboflow_url: str = "https://app.roboflow.com/ds/28uCRbc4wr?key=q7qHpXbRBo"
    roboflow_version: str = "5"


@dataclass
class TrainingSettings:
    """Configurações da aba Treinamento."""
    model_name: str = "DETR ResNet-50"
    epochs: int = 25
    batch_size: int = 2
    learning_rate: float = 1e-4
    lr_backbone: float = 1e-5  # Learning rate separado para backbone (10x menor)
    image_size: int = 640
    grad_accum: int = 4
    scheduler: str = "cosine"
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    max_grad_norm: float = 0.1
    precision: str = "bf16"
    use_tf32: bool = True
    use_compile: bool = False
    num_workers: int = 4
    # Anti-overfitting
    dropout: float = 0.1
    early_stopping_patience: int = 7
    use_augmentation: bool = True
    label_smoothing: float = 0.0
    
    # Data Augmentation - Flags de habilitação (básicas)
    enable_mosaic: bool = False  # Desabilitado: Roboflow já faz
    enable_mixup: bool = True
    enable_cutmix: bool = False
    enable_copy_paste: bool = False  # Desabilitado: Roboflow já faz
    enable_random_erasing: bool = True
    enable_motion_blur: bool = True
    enable_grid_mask: bool = True
    enable_cutout: bool = True
    enable_vflip: bool = False
    
    # Data Augmentation - Flags YOLO-style
    enable_hsv: bool = True
    enable_scale: bool = True
    enable_translate: bool = True
    enable_shear: bool = False
    
    # Data Augmentation - Probabilidades básicas
    hflip_prob: float = 0.5
    vflip_prob: float = 0.0
    color_jitter_prob: float = 0.3
    gaussian_blur_prob: float = 0.2
    
    # Data Augmentation - Probabilidades avançadas
    mosaic_prob: float = 0.0  # Desabilitado
    mixup_prob: float = 0.15
    cutmix_prob: float = 0.0
    copy_paste_prob: float = 0.0  # Desabilitado
    random_erasing_prob: float = 0.3
    motion_blur_prob: float = 0.2
    grid_mask_prob: float = 0.2
    cutout_prob: float = 0.3
    
    # Data Augmentation - Probabilidades YOLO-style
    hsv_prob: float = 0.5
    scale_prob: float = 0.5
    translate_prob: float = 0.3
    shear_prob: float = 0.0
    
    # Parâmetros HSV (YOLO defaults)
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    
    # Parâmetros geométricos
    scale_range_min: float = 0.5
    scale_range_max: float = 1.5
    translate_range: float = 0.1
    shear_range: float = 0.0
    
    # Loss weights (conforme paper DETR)
    bbox_loss_coefficient: float = 5.0
    giou_loss_coefficient: float = 2.0
    class_loss_coefficient: float = 1.0
    eos_coefficient: float = 0.1


@dataclass
class PredictionSettings:
    """Configurações da aba Predição."""
    model_path: str = ""
    video_path: str = ""
    threshold: float = 0.05
    fps_limit: int = 30
    natural_speed: bool = True
    show_preview: bool = True
    fast_mode: bool = False
    # ROI - Region of Interest (percentual de margem interna)
    roi_enabled: bool = True
    roi_x_percent: float = 25.0  # Margem X: 25% significa ROI de 25% a 75% da largura
    roi_y_percent: float = 25.0  # Margem Y: 25% significa ROI de 25% a 75% da altura


@dataclass
class EvaluationSettings:
    """Configurações da aba Avaliação."""
    model_path: str = ""
    dataset_path: str = ""
    split: str = "test"


@dataclass
class UserSettings:
    """Todas as configurações do usuário."""
    dataset: DatasetSettings = None
    training: TrainingSettings = None
    prediction: PredictionSettings = None
    evaluation: EvaluationSettings = None
    
    def __post_init__(self):
        if self.dataset is None:
            self.dataset = DatasetSettings()
        if self.training is None:
            self.training = TrainingSettings()
        if self.prediction is None:
            self.prediction = PredictionSettings()
        if self.evaluation is None:
            self.evaluation = EvaluationSettings()


class SettingsManager:
    """Gerencia persistência de configurações do usuário."""
    
    _instance: Optional['SettingsManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, settings_path: Path = DEFAULT_SETTINGS_PATH):
        if self._initialized:
            return
        
        self._initialized = True
        self.settings_path = Path(settings_path)
        self.settings = UserSettings()
        
        # Garantir que diretório existe
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Carregar configurações existentes
        self.load()
        
        logger.info(f"SettingsManager inicializado: {self.settings_path}")
    
    def load(self) -> bool:
        """Carrega configurações do arquivo."""
        if not self.settings_path.exists():
            logger.info("Arquivo de configuracoes nao existe, usando defaults")
            return False
        
        try:
            with open(self.settings_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Carregar cada seção (filtrar apenas campos válidos)
            if "dataset" in data:
                dataset_data = {k: v for k, v in data["dataset"].items() if hasattr(DatasetSettings, k)}
                self.settings.dataset = DatasetSettings(**dataset_data)
            if "training" in data:
                training_data = {k: v for k, v in data["training"].items() if hasattr(TrainingSettings, k)}
                self.settings.training = TrainingSettings(**training_data)
            if "prediction" in data:
                prediction_data = {k: v for k, v in data["prediction"].items() if hasattr(PredictionSettings, k)}
                self.settings.prediction = PredictionSettings(**prediction_data)
            if "evaluation" in data:
                evaluation_data = {k: v for k, v in data["evaluation"].items() if hasattr(EvaluationSettings, k)}
                self.settings.evaluation = EvaluationSettings(**evaluation_data)
            
            logger.info("Configuracoes carregadas com sucesso")
            return True
            
        except Exception as e:
            logger.warning(f"Erro ao carregar configuracoes: {e}")
            return False
    
    def save(self) -> bool:
        """Salva configurações no arquivo."""
        try:
            data = {
                "dataset": asdict(self.settings.dataset),
                "training": asdict(self.settings.training),
                "prediction": asdict(self.settings.prediction),
                "evaluation": asdict(self.settings.evaluation),
            }
            
            with open(self.settings_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info("Configuracoes salvas com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar configuracoes: {e}")
            return False
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Obtém valor de configuração."""
        section_obj = getattr(self.settings, section, None)
        if section_obj is None:
            return default
        return getattr(section_obj, key, default)
    
    def set(self, section: str, key: str, value: Any) -> bool:
        """Define valor de configuração."""
        section_obj = getattr(self.settings, section, None)
        if section_obj is None:
            return False
        
        if hasattr(section_obj, key):
            setattr(section_obj, key, value)
            return True
        return False
    
    def update_section(self, section: str, **kwargs) -> bool:
        """Atualiza múltiplos valores de uma seção."""
        for key, value in kwargs.items():
            self.set(section, key, value)
        return self.save()
    
    # Atalhos para cada seção
    @property
    def dataset(self) -> DatasetSettings:
        return self.settings.dataset
    
    @property
    def training(self) -> TrainingSettings:
        return self.settings.training
    
    @property
    def prediction(self) -> PredictionSettings:
        return self.settings.prediction
    
    @property
    def evaluation(self) -> EvaluationSettings:
        return self.settings.evaluation


# Instância global
settings_manager = SettingsManager()
