"""
Dataset COCO para DETR.
Sistema DETR - RTX 5080 Optimized
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from src.core.logger import logger
from src.core.config import ModelConfig


class COCODataset(Dataset):
    """Dataset COCO para treinamento DETR."""
    
    def __init__(
        self,
        root_dir: Path,
        split: str = "train",
        image_size: int = 800,
        transforms: Optional[Callable] = None,
        class_mapping: Optional[Dict[str, int]] = None
    ):
        """
        Args:
            root_dir: Diretório raiz do dataset
            split: Split do dataset (train, valid, val, test)
            image_size: Tamanho da imagem para resize
            transforms: Transformações adicionais
            class_mapping: Mapeamento de classes do dataset para classes fixas
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.transforms = transforms
        
        # Classes fixas do sistema
        self.model_config = ModelConfig()
        self.fixed_classes = self.model_config.FIXED_CLASSES
        self.num_classes = self.model_config.NUM_CLASSES
        
        # Encontrar diretório do split
        self.split_dir = self._find_split_dir()
        
        # Carregar anotações
        self.annotations_file = self._find_annotations_file()
        self.coco_data = self._load_annotations()
        
        # Criar índices
        self.images = self.coco_data.get("images", [])
        self.annotations = self._index_annotations()
        
        # Criar mapeamento de classes
        self.class_mapping = class_mapping or self._create_class_mapping()
        
        logger.info(f"Dataset {split} carregado: {len(self.images)} imagens")
    
    def _find_split_dir(self) -> Path:
        """Encontra diretório do split."""
        # Tentar diferentes nomes
        possible_names = [self.split]
        if self.split == "valid":
            possible_names.extend(["val", "validation"])
        elif self.split == "val":
            possible_names.extend(["valid", "validation"])
        
        for name in possible_names:
            split_dir = self.root_dir / name
            if split_dir.exists():
                return split_dir
        
        raise FileNotFoundError(f"Split '{self.split}' não encontrado em {self.root_dir}")
    
    def _find_annotations_file(self) -> Path:
        """Encontra arquivo de anotações."""
        possible_names = [
            "_annotations.coco.json",
            "annotations.json",
            f"{self.split}.json",
            "instances.json"
        ]
        
        for name in possible_names:
            ann_file = self.split_dir / name
            if ann_file.exists():
                return ann_file
        
        raise FileNotFoundError(f"Arquivo de anotações não encontrado em {self.split_dir}")
    
    def _load_annotations(self) -> Dict[str, Any]:
        """Carrega arquivo de anotações."""
        with open(self.annotations_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _index_annotations(self) -> Dict[int, List[Dict]]:
        """Indexa anotações por image_id."""
        indexed = {}
        for ann in self.coco_data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in indexed:
                indexed[img_id] = []
            indexed[img_id].append(ann)
        return indexed
    
    def _create_class_mapping(self) -> Dict[int, int]:
        """
        Cria mapeamento de classes do dataset para classes fixas do sistema.
        
        IMPORTANTE: O dataset do Roboflow tem IDs de categoria que NÃO correspondem
        aos IDs do sistema. O mapeamento é feito pelo NOME da categoria, não pelo ID.
        
        Exemplo do dataset Roboflow:
        - Dataset ID 0: 'objects-69mY-ok-b5Hm' -> Sistema ID 0 (ok)
        - Dataset ID 1: 'Ref' -> Sistema ID 1 (ref)
        - Dataset ID 2: 'ok' -> Sistema ID 0 (ok)
        - Dataset ID 3: 'vazio' -> Sistema ID 2 (vazio)
        
        Classes do sistema:
        - Sistema ID 0: ok
        - Sistema ID 1: ref  
        - Sistema ID 2: vazio
        """
        mapping = {}
        
        # Keywords para cada classe do sistema (case insensitive)
        # Sistema: 0=ok, 1=ref, 2=vazio
        class_keywords = {
            0: ["ok", "good", "normal", "pass", "objects", "b5hm"],  # ok
            1: ["ref", "reference", "defeito", "defect", "nok"],     # ref
            2: ["vazio", "empty", "null", "none", "blank"]           # vazio
        }
        
        categories = self.coco_data.get("categories", [])
        logger.info(f"Criando mapeamento de {len(categories)} categorias COCO -> {self.num_classes} classes do sistema")
        
        for cat in categories:
            cat_id = cat["id"]
            cat_name_original = cat["name"]
            cat_name = cat_name_original.lower().strip()
            
            system_id = None
            matched_by = None
            
            # Prioridade 1: Mapeamento direto por nome exato
            if cat_name == "ok":
                system_id = 0
                matched_by = "nome exato"
            elif cat_name == "ref":
                system_id = 1
                matched_by = "nome exato"
            elif cat_name == "vazio":
                system_id = 2
                matched_by = "nome exato"
            else:
                # Prioridade 2: Mapeamento por keywords
                for sys_id, keywords in class_keywords.items():
                    if any(kw in cat_name for kw in keywords):
                        system_id = sys_id
                        matched_by = f"keyword"
                        break
            
            # Prioridade 3: Fallback para ok (ID 0)
            if system_id is None:
                system_id = 0
                matched_by = "fallback"
                logger.warning(f"Categoria '{cat_name_original}' nao reconhecida, usando fallback 'ok'")
            
            mapping[cat_id] = system_id
            logger.info(f"  Dataset ID {cat_id} ('{cat_name_original}') -> Sistema ID {system_id} ({self.fixed_classes[system_id]}) [{matched_by}]")
        
        # Validar que todas as classes do sistema estão representadas
        mapped_system_ids = set(mapping.values())
        expected_system_ids = set(range(self.num_classes))
        
        if mapped_system_ids != expected_system_ids:
            missing = expected_system_ids - mapped_system_ids
            logger.warning(f"Classes do sistema nao mapeadas: {[self.fixed_classes[i] for i in missing]}")
        
        logger.info(f"Mapeamento final: {mapping}")
        return mapping
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Retorna imagem e targets."""
        img_info = self.images[idx]
        img_id = img_info["id"]
        
        # Carregar imagem
        img_path = self.split_dir / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")
        
        # Dimensões originais
        orig_w, orig_h = image.size
        
        # Carregar anotações antes do resize
        annotations_raw = self.annotations.get(img_id, [])
        
        # Aplicar augmentation ANTES do resize (em imagem PIL)
        if self.transforms:
            # Criar estrutura de anotações para augmentation
            ann_dict = {"annotations": annotations_raw}
            try:
                image, ann_dict = self.transforms(image, ann_dict)
                annotations_raw = ann_dict.get("annotations", annotations_raw)
            except Exception:
                # Em caso de erro, continuar sem augmentation
                pass
        
        # Resize mantendo aspect ratio (DEPOIS da augmentation)
        image, scale = self._resize_image(image)
        
        # Processar anotações para formato DETR
        targets = self._process_annotations(annotations_raw, orig_w, orig_h, scale)
        
        # Converter para tensor
        image_tensor = self._to_tensor(image)
        
        return image_tensor, targets
    
    def _resize_image(self, image: Image.Image) -> Tuple[Image.Image, float]:
        """Redimensiona imagem mantendo aspect ratio."""
        orig_w, orig_h = image.size
        
        # Calcular escala
        scale = self.image_size / max(orig_w, orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        image = image.resize((new_w, new_h), Image.BILINEAR)
        
        # Padding para tamanho fixo
        padded = Image.new("RGB", (self.image_size, self.image_size), (128, 128, 128))
        padded.paste(image, (0, 0))
        
        return padded, scale
    
    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Converte imagem para tensor normalizado."""
        # Converter para numpy
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Normalização ImageNet
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        
        # HWC -> CHW
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        
        return img_tensor
    
    def _process_annotations(
        self, 
        annotations: List[Dict],
        orig_w: int,
        orig_h: int,
        scale: float
    ) -> Dict[str, torch.Tensor]:
        """Processa anotações para formato DETR."""
        boxes = []
        labels = []
        
        for ann in annotations:
            # Skip anotações sem categoria válida
            cat_id = ann.get("category_id")
            if cat_id not in self.class_mapping:
                continue
            
            # Bbox no formato [x, y, width, height]
            bbox = ann["bbox"]
            x, y, w, h = bbox
            
            # Aplicar escala
            x *= scale
            y *= scale
            w *= scale
            h *= scale
            
            # Converter para formato DETR [cx, cy, w, h] normalizado
            cx = (x + w / 2) / self.image_size
            cy = (y + h / 2) / self.image_size
            w_norm = w / self.image_size
            h_norm = h / self.image_size
            
            # Validar bbox
            if w_norm > 0 and h_norm > 0 and cx >= 0 and cy >= 0:
                boxes.append([cx, cy, w_norm, h_norm])
                labels.append(self.class_mapping[cat_id])
        
        # Criar tensores
        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)
        
        return {
            "boxes": boxes_tensor,
            "labels": labels_tensor
        }


def collate_fn(batch):
    """Função de collate para DataLoader."""
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    # Stack imagens
    images = torch.stack(images, dim=0)
    
    return images, targets


def create_dataloaders(
    dataset_path: Path,
    batch_size: int = 8,
    image_size: int = 800,
    num_workers: int = 8,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    use_augmentation: bool = False,
    # Augmentation flags - básicas
    enable_mosaic: bool = False,  # Desabilitado: Roboflow já faz
    enable_mixup: bool = True,
    enable_cutmix: bool = False,
    enable_copy_paste: bool = False,  # Desabilitado: Roboflow já faz
    enable_random_erasing: bool = True,
    enable_motion_blur: bool = True,
    enable_grid_mask: bool = True,
    enable_cutout: bool = True,
    # Augmentation flags - YOLO-style
    enable_hsv: bool = True,
    enable_scale: bool = True,
    enable_translate: bool = True,
    enable_shear: bool = False,
    enable_vflip: bool = False,
    # Augmentation probabilities - básicas
    hflip_prob: float = 0.5,
    vflip_prob: float = 0.0,
    color_jitter_prob: float = 0.5,
    gaussian_blur_prob: float = 0.3,
    # Augmentation probabilities - avançadas
    mosaic_prob: float = 0.0,
    mixup_prob: float = 0.15,
    cutmix_prob: float = 0.0,
    copy_paste_prob: float = 0.0,
    random_erasing_prob: float = 0.3,
    motion_blur_prob: float = 0.2,
    grid_mask_prob: float = 0.2,
    cutout_prob: float = 0.3,
    # Augmentation probabilities - YOLO-style
    hsv_prob: float = 0.5,
    scale_prob: float = 0.5,
    translate_prob: float = 0.3,
    shear_prob: float = 0.0,
    # Parâmetros HSV
    hsv_h: float = 0.015,
    hsv_s: float = 0.7,
    hsv_v: float = 0.4,
    # Parâmetros geométricos
    scale_range_min: float = 0.5,
    scale_range_max: float = 1.5,
    translate_range: float = 0.1,
    shear_range: float = 0.0,
    small_objects: bool = False  # False para objetos médios/grandes
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Cria DataLoaders otimizados para treinamento.
    
    Args:
        dataset_path: Caminho do dataset COCO
        batch_size: Tamanho do batch
        image_size: Tamanho das imagens
        num_workers: Número de workers
        pin_memory: Usar pin memory
        persistent_workers: Manter workers persistentes
        prefetch_factor: Fator de prefetch
        use_augmentation: Se deve usar augmentation (apenas para train)
        enable_*: Flags para habilitar/desabilitar cada augmentation
        *_prob: Probabilidades de cada augmentation
        small_objects: Se detecta objetos pequenos (afeta configuração padrão)
    
    Returns:
        Tuple de (train_loader, val_loader, test_loader ou None)
    """
    dataset_path = Path(dataset_path)
    
    # Augmentation apenas para treino
    augmentation = None
    if use_augmentation:
        try:
            from src.data.augmentation import create_augmentation_pipeline
            augmentation = create_augmentation_pipeline(
                image_size=image_size,
                training=True,
                small_objects=small_objects,
                # Flags básicas
                enable_mosaic=enable_mosaic,
                enable_mixup=enable_mixup,
                enable_cutmix=enable_cutmix,
                enable_copy_paste=enable_copy_paste,
                enable_random_erasing=enable_random_erasing,
                enable_motion_blur=enable_motion_blur,
                enable_grid_mask=enable_grid_mask,
                enable_cutout=enable_cutout,
                # Flags YOLO-style
                enable_hsv=enable_hsv,
                enable_scale=enable_scale,
                enable_translate=enable_translate,
                enable_shear=enable_shear,
                enable_vflip=enable_vflip,
                # Probabilidades básicas
                hflip_prob=hflip_prob,
                vflip_prob=vflip_prob,
                color_jitter_prob=color_jitter_prob,
                gaussian_blur_prob=gaussian_blur_prob,
                # Probabilidades avançadas
                mosaic_prob=mosaic_prob,
                mixup_prob=mixup_prob,
                cutmix_prob=cutmix_prob,
                copy_paste_prob=copy_paste_prob,
                random_erasing_prob=random_erasing_prob,
                motion_blur_prob=motion_blur_prob,
                grid_mask_prob=grid_mask_prob,
                cutout_prob=cutout_prob,
                # Probabilidades YOLO-style
                hsv_prob=hsv_prob,
                scale_prob=scale_prob,
                translate_prob=translate_prob,
                shear_prob=shear_prob,
                # Parâmetros HSV
                hsv_h=hsv_h,
                hsv_s=hsv_s,
                hsv_v=hsv_v,
                # Parâmetros geométricos
                scale_range=(scale_range_min, scale_range_max),
                translate_range=translate_range,
                shear_range=shear_range,
            )
            enabled_augs = []
            if enable_mosaic and mosaic_prob > 0: enabled_augs.append("Mosaic")
            if enable_mixup and mixup_prob > 0: enabled_augs.append("MixUp")
            if enable_cutmix and cutmix_prob > 0: enabled_augs.append("CutMix")
            if enable_copy_paste and copy_paste_prob > 0: enabled_augs.append("Copy-Paste")
            if enable_hsv and hsv_prob > 0: enabled_augs.append("HSV")
            if enable_scale and scale_prob > 0: enabled_augs.append("Scale")
            if enable_translate and translate_prob > 0: enabled_augs.append("Translate")
            if enable_shear and shear_prob > 0: enabled_augs.append("Shear")
            if enable_random_erasing and random_erasing_prob > 0: enabled_augs.append("Random Erasing")
            if enable_motion_blur and motion_blur_prob > 0: enabled_augs.append("Motion Blur")
            if enable_grid_mask and grid_mask_prob > 0: enabled_augs.append("Grid Mask")
            if enable_cutout and cutout_prob > 0: enabled_augs.append("Cutout")
            if enable_vflip and vflip_prob > 0: enabled_augs.append("V-Flip")
            logger.info(f"Augmentation habilitada: {', '.join(enabled_augs) if enabled_augs else 'Básicas (H-Flip, Color Jitter, Gaussian Blur)'}")
        except Exception as e:
            logger.warning(f"Falha ao carregar augmentation: {e}")
    
    # Dataset de treino
    train_dataset = COCODataset(
        root_dir=dataset_path,
        split="train",
        image_size=image_size,
        transforms=augmentation
    )
    
    # Dataset de validação
    try:
        val_dataset = COCODataset(
            root_dir=dataset_path,
            split="valid",
            image_size=image_size,
            class_mapping=train_dataset.class_mapping
        )
    except FileNotFoundError:
        try:
            val_dataset = COCODataset(
                root_dir=dataset_path,
                split="val",
                image_size=image_size,
                class_mapping=train_dataset.class_mapping
            )
        except FileNotFoundError:
            logger.warning("Dataset de validação não encontrado!")
            val_dataset = None
    
    # Dataset de teste (opcional)
    try:
        test_dataset = COCODataset(
            root_dir=dataset_path,
            split="test",
            image_size=image_size,
            class_mapping=train_dataset.class_mapping
        )
    except FileNotFoundError:
        test_dataset = None
    
    # Configurações do DataLoader
    loader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": collate_fn,
        "pin_memory": pin_memory,
        "drop_last": False
    }
    
    # Ajustar workers para Windows
    if num_workers > 0:
        loader_kwargs["num_workers"] = num_workers
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    else:
        loader_kwargs["num_workers"] = 0
    
    # Train loader
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs
    )
    
    # Val loader
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **loader_kwargs
        )
    
    # Test loader
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            **loader_kwargs
        )
    
    logger.info(f"DataLoaders criados:")
    logger.info(f"  Train: {len(train_dataset)} imagens, {len(train_loader)} batches")
    if val_loader:
        logger.info(f"  Val: {len(val_dataset)} imagens, {len(val_loader)} batches")
    if test_loader:
        logger.info(f"  Test: {len(test_dataset)} imagens, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader
