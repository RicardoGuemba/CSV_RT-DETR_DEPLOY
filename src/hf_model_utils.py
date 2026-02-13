"""
Utilitários para carregamento e salvamento de modelos Hugging Face DETR.
Conforme PRD - Sistema de Detecção de Objetos com DETR
"""
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import torch
from transformers import (
    DetrForObjectDetection,
    DetrImageProcessor,
    AutoModelForObjectDetection,
    AutoImageProcessor
)


def load_detr_model_for_training(
    model_name: str,
    num_classes: int,
    id2label: Dict[int, str],
    device: torch.device,
) -> Tuple[DetrForObjectDetection, DetrImageProcessor]:
    """
    Carrega modelo DETR para treinamento.
    
    Args:
        model_name: Nome do modelo (ex: "facebook/detr-resnet-50")
        num_classes: Número de classes (sem background)
        id2label: Mapeamento ID -> Label
        device: Dispositivo (cuda/cpu)
    
    Returns:
        Tupla (model, processor)
    """
    # Normalizar id2label (garantir chaves inteiras)
    id2label_normalized = {int(k): str(v) for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label_normalized.items()}
    
    # Tentar carregar como AutoModel primeiro
    try:
        model = AutoModelForObjectDetection.from_pretrained(
            model_name,
            num_labels=num_classes,
            id2label=id2label_normalized,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
        processor = AutoImageProcessor.from_pretrained(model_name)
    except Exception:
        # Fallback para DETR padrão
        model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=num_classes,
            id2label=id2label_normalized,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    
    model.to(device)
    return model, processor


def load_detr_model_for_inference(
    model_path: str,
    device: torch.device,
) -> Tuple[DetrForObjectDetection, DetrImageProcessor, Dict[int, str]]:
    """
    Carrega modelo DETR treinado para inferência.
    
    Args:
        model_path: Caminho para checkpoint ou diretório do modelo
        device: Dispositivo (cuda/cpu)
    
    Returns:
        Tupla (model, processor, id2label)
    
    IMPORTANTE: Normaliza id2label para ter chaves inteiras.
    Converte strings para int: {'0': 'ok'} → {0: 'ok'}
    """
    model_path = Path(model_path)
    
    # Encontrar checkpoint
    if model_path.is_file():
        checkpoint_path = model_path
        model_dir = model_path.parent
    else:
        # É um diretório - procurar best_model.pt ou qualquer .pt
        model_dir = model_path
        checkpoint_path = model_dir / "best_model.pt"
        if not checkpoint_path.exists():
            pt_files = list(model_dir.glob("*.pt"))
            if pt_files:
                checkpoint_path = pt_files[0]
            else:
                raise FileNotFoundError(f"Nenhum modelo encontrado em {model_dir}")
    
    # Carregar checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Recuperar configuração
    exp_config = checkpoint.get("experiment_config", {})
    model_id = exp_config.get("model_id", "facebook/detr-resnet-50")
    num_classes = exp_config.get("num_classes", 3)
    
    # Recuperar id2label do checkpoint ou config
    id2label_raw = checkpoint.get("id2label", {})
    if not id2label_raw:
        # Tentar do config.json
        config_path = model_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                id2label_raw = config_data.get("id2label", {})
    
    # NORMALIZAR: Converter chaves string para int
    id2label = {int(k): str(v) for k, v in id2label_raw.items()}
    label2id = {v: k for k, v in id2label.items()}
    
    # Carregar modelo
    try:
        model = AutoModelForObjectDetection.from_pretrained(
            model_id,
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
        processor = AutoImageProcessor.from_pretrained(model_id)
    except Exception:
        model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    
    # Carregar pesos do checkpoint
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.to(device)
    model.eval()
    
    return model, processor, id2label


def save_model(
    model: DetrForObjectDetection,
    image_processor: DetrImageProcessor,
    output_dir: str,
    id2label: Dict[int, str],
    training_config: Optional[Dict] = None,
) -> None:
    """
    Salva modelo treinado.
    
    Args:
        model: Modelo DETR treinado
        image_processor: Processador de imagens
        output_dir: Diretório de saída
        id2label: Mapeamento ID -> Label
        training_config: Configuração de treinamento (opcional)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Normalizar id2label
    id2label_normalized = {int(k): str(v) for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label_normalized.items()}
    
    # Salvar modelo e processor
    model.save_pretrained(output_dir)
    image_processor.save_pretrained(output_dir)
    
    # Salvar configuração adicional
    config_data = {
        "id2label": id2label_normalized,
        "label2id": label2id,
        "num_classes": len(id2label_normalized)
    }
    
    if training_config:
        config_data["training_config"] = training_config
    
    config_path = output_dir / "config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
