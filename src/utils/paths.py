"""
Gerenciador de caminhos do projeto.
Carrega configuração de config/paths.json e gerencia estrutura de diretórios.
"""
import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


class PathManager:
    """Gerencia caminhos do projeto baseado em configuração JSON."""
    
    def __init__(self, config_path: str = "config/paths.json"):
        """
        Args:
            config_path: Caminho para arquivo de configuração de paths
        """
        self.base_dir = Path(__file__).parent.parent.parent.resolve()
        self.config_path = self.base_dir / config_path
        
        # Carregar configuração
        self.config = self._load_config()
        
        # Resolver paths relativos ao base_dir
        self.datasets_dir = self.base_dir / self.config.get("datasets_dir", "data/datasets")
        self.models_dir = self.base_dir / self.config.get("models_dir", "data/models/experiments")
        self.outputs_dir = self.base_dir / self.config.get("outputs_dir", "data/outputs")
        self.predictions_dir = self.base_dir / self.config.get("predictions_dir", "data/outputs/predictions")
        self.logs_dir = self.base_dir / self.config.get("logs_dir", "data/logs")
        
        # Subdiretórios
        self.checkpoints_subdir = self.config.get("checkpoints_subdir", "checkpoints")
        self.model_best_subdir = self.config.get("model_best_subdir", "model_best")
        self.model_final_subdir = self.config.get("model_final_subdir", "model_final")
    
    def _load_config(self) -> Dict:
        """Carrega configuração de paths.json."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Erro ao carregar config/paths.json: {e}")
        
        # Retornar defaults se arquivo não existir
        return {
            "datasets_dir": "data/datasets",
            "models_dir": "data/models/experiments",
            "outputs_dir": "data/outputs",
            "predictions_dir": "data/outputs/predictions",
            "logs_dir": "data/logs",
            "checkpoints_subdir": "checkpoints",
            "model_best_subdir": "model_best",
            "model_final_subdir": "model_final"
        }
    
    def get_dataset_path(self, dataset_name: str) -> Path:
        """Retorna caminho para um dataset."""
        return self.datasets_dir / dataset_name
    
    def get_models_dir(self) -> Path:
        """Retorna diretório de modelos."""
        return self.models_dir
    
    def generate_experiment_name(self, prefix: str = "detr") -> str:
        """
        Gera nome único para experimento.
        
        Args:
            prefix: Prefixo do nome (ex: "detr")
        
        Returns:
            Nome no formato {prefix}_{YYYYMMDD}{HHMMSS}
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}"
    
    def create_experiment_dirs(self, experiment_name: str) -> Dict[str, Path]:
        """
        Cria estrutura de diretórios para um experimento.
        
        Args:
            experiment_name: Nome do experimento
        
        Returns:
            Dicionário com paths criados
        """
        exp_dir = self.models_dir / experiment_name
        checkpoints_dir = exp_dir / self.checkpoints_subdir
        model_best_dir = exp_dir / self.model_best_subdir
        model_final_dir = exp_dir / self.model_final_subdir
        
        # Criar diretórios
        exp_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        model_best_dir.mkdir(parents=True, exist_ok=True)
        model_final_dir.mkdir(parents=True, exist_ok=True)
        
        return {
            "experiment": exp_dir,
            "checkpoints": checkpoints_dir,
            "model_best": model_best_dir,
            "model_final": model_final_dir
        }
    
    def get_prediction_output_path(self, video_path: str, model_name: str) -> Path:
        """
        Gera caminho de saída para vídeo processado.
        
        Args:
            video_path: Caminho do vídeo original
            model_name: Nome do modelo usado
        
        Returns:
            Path para vídeo de saída
        """
        video_name = Path(video_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{video_name}_{model_name}_{timestamp}.mp4"
        return self.predictions_dir / output_name
