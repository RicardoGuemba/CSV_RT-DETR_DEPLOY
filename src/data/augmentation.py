"""
Data Augmentation state-of-the-art para Object Detection.
Técnicas para melhorar detecção de objetos e evitar overfitting.
Inclui: MixUp, Mosaic, Copy-Paste, Random Erasing, Motion Blur, Grid Mask, 
HSV, Scale, Translation, CutMix, Shear, e outras (similar a YOLO).
"""
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import random
import math

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class AdvancedAugmentation:
    """Augmentation avançada com técnicas state-of-the-art para object detection.
    
    Inclui técnicas similares ao YOLO: HSV, Scale, Translation, CutMix, Shear.
    Otimizado para modelos DETR/RT-DETR.
    """
    
    def __init__(
        self,
        image_size: int = 1024,
        # Augmentações básicas
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.0,  # Flip vertical (geralmente 0 para objetos naturais)
        color_jitter_prob: float = 0.5,
        gaussian_blur_prob: float = 0.3,
        # Parâmetros de cor (Color Jitter)
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        saturation_range: Tuple[float, float] = (0.8, 1.2),
        hue_range: Tuple[float, float] = (-0.1, 0.1),
        # HSV augmentation (similar YOLO) - mais controle que color_jitter
        hsv_prob: float = 0.0,  # Probabilidade de aplicar HSV
        hsv_h: float = 0.015,  # Hue variation (fração, YOLO default: 0.015)
        hsv_s: float = 0.7,    # Saturation variation (fração, YOLO default: 0.7)
        hsv_v: float = 0.4,    # Value variation (fração, YOLO default: 0.4)
        # Augmentações geométricas (similar YOLO)
        scale_prob: float = 0.0,  # Probabilidade de aplicar scale
        scale_range: Tuple[float, float] = (0.5, 1.5),  # YOLO default: 0.5
        translate_prob: float = 0.0,  # Probabilidade de aplicar translation
        translate_range: float = 0.1,  # Fração da imagem (YOLO default: 0.1)
        shear_prob: float = 0.0,  # Probabilidade de aplicar shear
        shear_range: float = 0.0,  # Graus (YOLO default: 0.0)
        # Augmentações avançadas
        mosaic_prob: float = 0.3,
        mixup_prob: float = 0.2,
        cutmix_prob: float = 0.0,  # CutMix (YOLO uses this)
        copy_paste_prob: float = 0.1,
        random_erasing_prob: float = 0.3,
        motion_blur_prob: float = 0.2,
        grid_mask_prob: float = 0.2,
        cutout_prob: float = 0.3,
        cutout_size: Tuple[int, int] = (20, 60),
        # Parâmetros específicos
        mosaic_scale_range: Tuple[float, float] = (0.5, 1.5),
        mixup_alpha: float = 1.5,
        cutmix_alpha: float = 1.0,  # Alpha para CutMix
        copy_paste_scale_range: Tuple[float, float] = (0.5, 1.5),
        erasing_area_ratio: Tuple[float, float] = (0.02, 0.33),
        erasing_aspect_ratio: Tuple[float, float] = (0.3, 3.3),
        motion_blur_kernel_size: Tuple[int, int] = (3, 7),
        grid_mask_grid_size: Tuple[int, int] = (8, 16),
        grid_mask_ratio: Tuple[float, float] = (0.3, 0.6),
        # Flags de habilitação
        enable_mosaic: bool = True,
        enable_mixup: bool = True,
        enable_cutmix: bool = False,  # CutMix desabilitado por padrão
        enable_copy_paste: bool = True,
        enable_random_erasing: bool = True,
        enable_motion_blur: bool = True,
        enable_grid_mask: bool = True,
        enable_cutout: bool = True,
        enable_hsv: bool = False,  # HSV desabilitado por padrão
        enable_scale: bool = False,  # Scale desabilitado por padrão
        enable_translate: bool = False,  # Translate desabilitado por padrão
        enable_shear: bool = False,  # Shear desabilitado por padrão
        enable_vflip: bool = False,  # Vertical flip desabilitado por padrão
    ):
        """
        Args:
            image_size: Tamanho da imagem
            hflip_prob: Probabilidade de flip horizontal
            vflip_prob: Probabilidade de flip vertical
            color_jitter_prob: Probabilidade de color jitter
            gaussian_blur_prob: Probabilidade de blur gaussiano
            brightness_range: Range de brilho
            contrast_range: Range de contraste
            saturation_range: Range de saturação
            hue_range: Range de hue
            hsv_prob: Probabilidade de HSV augmentation (similar YOLO)
            hsv_h/s/v: Parâmetros de HSV (frações)
            scale_prob: Probabilidade de scale augmentation
            scale_range: Range de escala
            translate_prob: Probabilidade de translation augmentation
            translate_range: Range de translation (fração)
            shear_prob: Probabilidade de shear augmentation
            shear_range: Range de shear em graus
            mosaic_prob: Probabilidade de mosaic augmentation
            mixup_prob: Probabilidade de mixup augmentation
            cutmix_prob: Probabilidade de cutmix augmentation
            copy_paste_prob: Probabilidade de copy-paste augmentation
            random_erasing_prob: Probabilidade de random erasing
            motion_blur_prob: Probabilidade de motion blur
            grid_mask_prob: Probabilidade de grid mask
            cutout_prob: Probabilidade de cutout
            cutout_size: Tamanho do cutout (min, max)
            mosaic_scale_range: Range de escala para mosaic
            mixup_alpha: Parâmetro alpha para mixup (distribuição Beta)
            cutmix_alpha: Parâmetro alpha para cutmix (distribuição Beta)
            copy_paste_scale_range: Range de escala para copy-paste
            erasing_area_ratio: Range de área para random erasing
            erasing_aspect_ratio: Range de aspect ratio para random erasing
            motion_blur_kernel_size: Tamanho do kernel para motion blur
            grid_mask_grid_size: Tamanho da grade para grid mask
            grid_mask_ratio: Range de ratio para grid mask
            enable_*: Flags para habilitar/desabilitar cada augmentation
        """
        self.image_size = image_size
        # Flips
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.enable_vflip = enable_vflip
        # Color
        self.color_jitter_prob = color_jitter_prob
        self.gaussian_blur_prob = gaussian_blur_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        # HSV (YOLO-style)
        self.hsv_prob = hsv_prob
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v
        self.enable_hsv = enable_hsv
        # Geometric
        self.scale_prob = scale_prob
        self.scale_range = scale_range
        self.enable_scale = enable_scale
        self.translate_prob = translate_prob
        self.translate_range = translate_range
        self.enable_translate = enable_translate
        self.shear_prob = shear_prob
        self.shear_range = shear_range
        self.enable_shear = enable_shear
        # Advanced
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
        self.cutmix_alpha = cutmix_alpha
        self.enable_cutmix = enable_cutmix
        self.copy_paste_prob = copy_paste_prob
        self.random_erasing_prob = random_erasing_prob
        self.motion_blur_prob = motion_blur_prob
        self.grid_mask_prob = grid_mask_prob
        self.cutout_prob = cutout_prob
        self.cutout_size = cutout_size
        self.mosaic_scale_range = mosaic_scale_range
        self.mixup_alpha = mixup_alpha
        self.copy_paste_scale_range = copy_paste_scale_range
        self.erasing_area_ratio = erasing_area_ratio
        self.erasing_aspect_ratio = erasing_aspect_ratio
        self.motion_blur_kernel_size = motion_blur_kernel_size
        self.grid_mask_grid_size = grid_mask_grid_size
        self.grid_mask_ratio = grid_mask_ratio
        self.enable_mosaic = enable_mosaic
        self.enable_mixup = enable_mixup
        self.enable_copy_paste = enable_copy_paste
        self.enable_random_erasing = enable_random_erasing
        self.enable_motion_blur = enable_motion_blur
        self.enable_grid_mask = enable_grid_mask
        self.enable_cutout = enable_cutout
        
        # Cache para imagens e anotações (para MixUp, CutMix e Copy-Paste)
        self._image_cache: List[Tuple[Image.Image, Dict[str, Any]]] = []
        self._cache_size = 50
    
    def __call__(self, image: Image.Image, annotations: Dict[str, Any], 
                 dataset_cache: Optional[List[Tuple[Image.Image, Dict[str, Any]]]] = None) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Aplica augmentations state-of-the-art.
        
        Args:
            image: Imagem PIL
            annotations: Anotações no formato COCO
            dataset_cache: Cache opcional de outras imagens do dataset (para MixUp, Mosaic, CutMix, Copy-Paste)
        
        Returns:
            Imagem e anotações aumentadas
        """
        # Atualizar cache
        if dataset_cache is not None:
            self._image_cache = dataset_cache[:self._cache_size]
        
        # ============ AUGMENTAÇÕES QUE COMBINAM MÚLTIPLAS IMAGENS ============
        # Nota: Mosaic e CutMix são mutuamente exclusivos (aplicar apenas um)
        if self.enable_mosaic and random.random() < self.mosaic_prob and len(self._image_cache) >= 3:
            image, annotations = self._mosaic(image, annotations)
        elif self.enable_cutmix and random.random() < self.cutmix_prob and len(self._image_cache) >= 1:
            image, annotations = self._cutmix(image, annotations)
        
        if self.enable_mixup and random.random() < self.mixup_prob and len(self._image_cache) >= 1:
            image, annotations = self._mixup(image, annotations)
        
        if self.enable_copy_paste and random.random() < self.copy_paste_prob and len(self._image_cache) >= 1:
            image, annotations = self._copy_paste(image, annotations)
        
        # ============ AUGMENTAÇÕES GEOMÉTRICAS ============
        # Flip horizontal
        if self.hflip_prob > 0 and random.random() < self.hflip_prob:
            image, annotations = self._horizontal_flip(image, annotations)
        
        # Flip vertical (novo)
        if self.enable_vflip and self.vflip_prob > 0 and random.random() < self.vflip_prob:
            image, annotations = self._vertical_flip(image, annotations)
        
        # Scale (novo - similar YOLO)
        if self.enable_scale and self.scale_prob > 0 and random.random() < self.scale_prob:
            image, annotations = self._scale(image, annotations)
        
        # Translation (novo - similar YOLO)
        if self.enable_translate and self.translate_prob > 0 and random.random() < self.translate_prob:
            image, annotations = self._translate(image, annotations)
        
        # Shear (novo - similar YOLO)
        if self.enable_shear and self.shear_prob > 0 and random.random() < self.shear_prob:
            image, annotations = self._shear(image, annotations)
        
        # ============ AUGMENTAÇÕES DE COR ============
        # HSV augmentation (novo - similar YOLO)
        if self.enable_hsv and self.hsv_prob > 0 and random.random() < self.hsv_prob:
            image = self._hsv_augmentation(image)
        
        # Color jitter tradicional
        if self.color_jitter_prob > 0 and random.random() < self.color_jitter_prob:
            image = self._color_jitter(image)
        
        # ============ AUGMENTAÇÕES DE BLUR ============
        if self.gaussian_blur_prob > 0 and random.random() < self.gaussian_blur_prob:
            image = self._gaussian_blur(image)
        
        if self.enable_motion_blur and random.random() < self.motion_blur_prob:
            image = self._motion_blur(image)
        
        # ============ AUGMENTAÇÕES DE REGULARIZAÇÃO ============
        if self.enable_random_erasing and random.random() < self.random_erasing_prob:
            image = self._random_erasing(image)
        
        if self.enable_grid_mask and random.random() < self.grid_mask_prob:
            image = self._grid_mask(image)
        
        if self.enable_cutout and random.random() < self.cutout_prob:
            image = self._cutout(image)
        
        return image, annotations
    
    def _horizontal_flip(self, image: Image.Image, annotations: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, Any]]:
        """Flip horizontal."""
        w, h = image.size
        image = TF.hflip(image)
        
        # Atualizar bboxes
        new_annotations = annotations.copy()
        if "annotations" in new_annotations:
            for ann in new_annotations["annotations"]:
                bbox = ann["bbox"]  # [x, y, w, h]
                x, y, width, height = bbox
                # Flip: x' = w - x - width
                new_x = w - x - width
                ann["bbox"] = [new_x, y, width, height]
        
        return image, new_annotations
    
    def _vertical_flip(self, image: Image.Image, annotations: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, Any]]:
        """Flip vertical."""
        w, h = image.size
        image = TF.vflip(image)
        
        # Atualizar bboxes
        new_annotations = annotations.copy()
        if "annotations" in new_annotations:
            for ann in new_annotations["annotations"]:
                bbox = ann["bbox"]  # [x, y, w, h]
                x, y, width, height = bbox
                # Flip vertical: y' = h - y - height
                new_y = h - y - height
                ann["bbox"] = [x, new_y, width, height]
        
        return image, new_annotations
    
    def _hsv_augmentation(self, image: Image.Image) -> Image.Image:
        """HSV augmentation similar ao YOLO."""
        if not HAS_CV2:
            # Fallback para color jitter se cv2 não disponível
            return self._color_jitter(image)
        
        img_array = np.array(image)
        
        # Converter RGB para HSV
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Aplicar variações HSV
        h_delta = random.uniform(-self.hsv_h, self.hsv_h) * 180  # Hue em graus [0, 180] no OpenCV
        s_delta = random.uniform(1 - self.hsv_s, 1 + self.hsv_s)
        v_delta = random.uniform(1 - self.hsv_v, 1 + self.hsv_v)
        
        hsv[:, :, 0] = (hsv[:, :, 0] + h_delta) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_delta, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_delta, 0, 255)
        
        # Converter de volta para RGB
        img_array = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        return Image.fromarray(img_array)
    
    def _scale(self, image: Image.Image, annotations: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, Any]]:
        """Scale augmentation similar ao YOLO."""
        w, h = image.size
        
        # Fator de escala aleatório
        scale_factor = random.uniform(*self.scale_range)
        
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        
        # Redimensionar imagem
        scaled_image = image.resize((new_w, new_h), Image.BILINEAR)
        
        # Criar nova imagem com padding ou crop
        result = Image.new("RGB", (w, h), (128, 128, 128))
        
        if scale_factor < 1:
            # Padding: centralizar imagem menor
            paste_x = (w - new_w) // 2
            paste_y = (h - new_h) // 2
            result.paste(scaled_image, (paste_x, paste_y))
            
            # Ajustar bboxes
            new_annotations = annotations.copy()
            if "annotations" in new_annotations:
                new_annotations["annotations"] = []
                for ann in annotations.get("annotations", []):
                    bbox = ann["bbox"]  # [x, y, w, h]
                    x, y, bw, bh = bbox
                    new_bbox = [
                        x * scale_factor + paste_x,
                        y * scale_factor + paste_y,
                        bw * scale_factor,
                        bh * scale_factor
                    ]
                    new_ann = ann.copy()
                    new_ann["bbox"] = new_bbox
                    new_annotations["annotations"].append(new_ann)
        else:
            # Crop: cortar do centro
            crop_x = (new_w - w) // 2
            crop_y = (new_h - h) // 2
            result = scaled_image.crop((crop_x, crop_y, crop_x + w, crop_y + h))
            
            # Ajustar bboxes
            new_annotations = annotations.copy()
            if "annotations" in new_annotations:
                new_annotations["annotations"] = []
                for ann in annotations.get("annotations", []):
                    bbox = ann["bbox"]  # [x, y, w, h]
                    x, y, bw, bh = bbox
                    new_x = x * scale_factor - crop_x
                    new_y = y * scale_factor - crop_y
                    new_bbox = [new_x, new_y, bw * scale_factor, bh * scale_factor]
                    
                    # Verificar se bbox ainda está visível
                    if new_x + new_bbox[2] > 0 and new_y + new_bbox[3] > 0 and new_x < w and new_y < h:
                        new_ann = ann.copy()
                        new_ann["bbox"] = new_bbox
                        new_annotations["annotations"].append(new_ann)
        
        return result, new_annotations
    
    def _translate(self, image: Image.Image, annotations: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, Any]]:
        """Translation augmentation similar ao YOLO."""
        w, h = image.size
        
        # Deslocamento aleatório
        tx = int(w * random.uniform(-self.translate_range, self.translate_range))
        ty = int(h * random.uniform(-self.translate_range, self.translate_range))
        
        # Criar nova imagem com deslocamento
        result = Image.new("RGB", (w, h), (128, 128, 128))
        result.paste(image, (tx, ty))
        
        # Ajustar bboxes
        new_annotations = annotations.copy()
        if "annotations" in new_annotations:
            new_annotations["annotations"] = []
            for ann in annotations.get("annotations", []):
                bbox = ann["bbox"]  # [x, y, w, h]
                x, y, bw, bh = bbox
                new_x = x + tx
                new_y = y + ty
                
                # Verificar se bbox ainda está visível
                if new_x + bw > 0 and new_y + bh > 0 and new_x < w and new_y < h:
                    new_ann = ann.copy()
                    new_ann["bbox"] = [new_x, new_y, bw, bh]
                    new_annotations["annotations"].append(new_ann)
        
        return result, new_annotations
    
    def _shear(self, image: Image.Image, annotations: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, Any]]:
        """Shear augmentation similar ao YOLO."""
        if self.shear_range == 0:
            return image, annotations
        
        w, h = image.size
        
        # Ângulo de shear aleatório
        shear_x = math.radians(random.uniform(-self.shear_range, self.shear_range))
        shear_y = math.radians(random.uniform(-self.shear_range, self.shear_range))
        
        # Matriz de transformação affine para shear
        # [1, shear_x, 0]
        # [shear_y, 1, 0]
        coeffs = (1, math.tan(shear_x), 0, math.tan(shear_y), 1, 0)
        
        # Aplicar transformação
        image = image.transform((w, h), Image.AFFINE, coeffs, Image.BILINEAR, fillcolor=(128, 128, 128))
        
        # Ajustar bboxes (aproximação simples)
        new_annotations = annotations.copy()
        if "annotations" in new_annotations:
            new_annotations["annotations"] = []
            for ann in annotations.get("annotations", []):
                bbox = ann["bbox"]  # [x, y, w, h]
                x, y, bw, bh = bbox
                
                # Ajustar centro
                cx = x + bw / 2
                cy = y + bh / 2
                new_cx = cx + cy * math.tan(shear_x)
                new_cy = cy + cx * math.tan(shear_y)
                
                # Manter tamanho (aproximação)
                new_x = new_cx - bw / 2
                new_y = new_cy - bh / 2
                
                # Verificar se bbox ainda está visível
                if new_x + bw > 0 and new_y + bh > 0 and new_x < w and new_y < h:
                    new_ann = ann.copy()
                    new_ann["bbox"] = [new_x, new_y, bw, bh]
                    new_annotations["annotations"].append(new_ann)
        
        return image, new_annotations
    
    def _cutmix(self, image: Image.Image, annotations: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, Any]]:
        """CutMix augmentation: recorta região de outra imagem e cola."""
        if len(self._image_cache) < 1:
            return image, annotations
        
        # Selecionar imagem aleatória do cache
        other_image, other_annotations = random.choice(self._image_cache)
        
        # Redimensionar se necessário
        if other_image.size != image.size:
            other_image = other_image.resize(image.size, Image.BILINEAR)
        
        w, h = image.size
        
        # Lambda da distribuição Beta
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        
        # Calcular tamanho da região de corte
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)
        
        # Posição aleatória do centro do corte
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        
        # Calcular coordenadas do corte (clipar para limites da imagem)
        x1 = max(0, cx - cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        x2 = min(w, cx + cut_w // 2)
        y2 = min(h, cy + cut_h // 2)
        
        # Converter para arrays
        img1 = np.array(image)
        img2 = np.array(other_image)
        
        # Colar região da segunda imagem na primeira
        img1[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
        
        mixed_image = Image.fromarray(img1)
        
        # Combinar anotações
        new_annotations = {"annotations": []}
        
        # Anotações da primeira imagem (fora da região de corte)
        if "annotations" in annotations:
            for ann in annotations["annotations"]:
                bbox = ann["bbox"]  # [x, y, w, h]
                bx, by, bw, bh = bbox
                
                # Verificar se bbox está fora da região de corte
                if bx + bw < x1 or bx > x2 or by + bh < y1 or by > y2:
                    new_annotations["annotations"].append(ann.copy())
        
        # Anotações da segunda imagem (dentro da região de corte)
        if "annotations" in other_annotations:
            for ann in other_annotations["annotations"]:
                bbox = ann["bbox"]  # [x, y, w, h]
                bx, by, bw, bh = bbox
                
                # Verificar se bbox está dentro da região de corte
                if bx >= x1 and bx + bw <= x2 and by >= y1 and by + bh <= y2:
                    new_annotations["annotations"].append(ann.copy())
        
        return mixed_image, new_annotations
    
    def _rotate(self, image: Image.Image, annotations: Dict[str, Any], angle: float) -> Tuple[Image.Image, Dict[str, Any]]:
        """Rotação leve (pequenos objetos são sensíveis a rotações grandes)."""
        # Rotação leve apenas
        if abs(angle) < 1.0:
            return image, annotations
        
        w, h = image.size
        center = (w // 2, h // 2)
        
        # Rotacionar imagem
        image = TF.rotate(image, angle, center=center)
        
        # Rotacionar bboxes (aproximação simples para rotações pequenas)
        # Para rotações maiores, seria necessário transformação mais complexa
        if abs(angle) < 5.0 and "annotations" in annotations:
            new_annotations = annotations.copy()
            # Para rotações pequenas, manter bboxes (aproximação)
            # Em produção, implementar transformação de coordenadas completa
            pass
        
        return image, annotations
    
    def _color_jitter(self, image: Image.Image) -> Image.Image:
        """Color jitter."""
        brightness = random.uniform(*self.brightness_range)
        contrast = random.uniform(*self.contrast_range)
        saturation = random.uniform(*self.saturation_range)
        hue = random.uniform(*self.hue_range)
        
        image = TF.adjust_brightness(image, brightness)
        image = TF.adjust_contrast(image, contrast)
        image = TF.adjust_saturation(image, saturation)
        image = TF.adjust_hue(image, hue)
        
        return image
    
    def _gaussian_blur(self, image: Image.Image) -> Image.Image:
        """Blur gaussiano leve."""
        sigma = random.uniform(0.5, 1.5)
        kernel_size = int(2 * sigma * 2) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Converter para tensor, aplicar blur, converter de volta
        img_tensor = TF.to_tensor(image)
        blur_transform = T.GaussianBlur(kernel_size=kernel_size, sigma=(sigma, sigma))
        img_tensor = blur_transform(img_tensor)
        image = TF.to_pil_image(img_tensor)
        
        return image
    
    def _cutout(self, image: Image.Image) -> Image.Image:
        """Cutout para regularização."""
        w, h = image.size
        img_array = np.array(image)
        
        # Número de cutouts
        num_cutouts = random.randint(1, 3)
        
        for _ in range(num_cutouts):
            # Tamanho do cutout
            cutout_w = random.randint(*self.cutout_size)
            cutout_h = random.randint(*self.cutout_size)
            
            # Posição aleatória
            x = random.randint(0, max(1, w - cutout_w))
            y = random.randint(0, max(1, h - cutout_h))
            
            # Preencher com valor médio
            mean_color = img_array.mean(axis=(0, 1)).astype(np.uint8)
            img_array[y:y+cutout_h, x:x+cutout_w] = mean_color
        
        return Image.fromarray(img_array)
    
    def _mosaic(self, image: Image.Image, annotations: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, Any]]:
        """Mosaic augmentation: combina 4 imagens em uma grade 2x2."""
        if len(self._image_cache) < 3:
            return image, annotations
        
        w, h = image.size
        # Selecionar 3 imagens aleatórias do cache
        other_images = random.sample(self._image_cache, 3)
        
        # Criar canvas para mosaic
        mosaic_w, mosaic_h = w * 2, h * 2
        mosaic = Image.new("RGB", (mosaic_w, mosaic_h), (128, 128, 128))
        
        # Posições na grade 2x2
        positions = [(0, 0), (w, 0), (0, h), (w, h)]
        images_to_place = [image] + [img for img, _ in other_images]
        annotations_list = [annotations] + [ann for _, ann in other_images]
        
        new_annotations = {"annotations": []}
        
        for idx, (pos_x, pos_y) in enumerate(positions):
            img = images_to_place[idx]
            ann = annotations_list[idx]
            
            # Redimensionar imagem aleatoriamente
            scale = random.uniform(*self.mosaic_scale_range)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_resized = img.resize((new_w, new_h), Image.BILINEAR)
            
            # Colar na posição
            mosaic.paste(img_resized, (pos_x, pos_y))
            
            # Ajustar anotações
            if "annotations" in ann:
                for bbox_ann in ann["annotations"]:
                    bbox = bbox_ann["bbox"].copy()  # [x, y, w, h]
                    # Ajustar coordenadas para posição no mosaic
                    bbox[0] = bbox[0] * scale + pos_x
                    bbox[1] = bbox[1] * scale + pos_y
                    bbox[2] *= scale
                    bbox[3] *= scale
                    new_ann = bbox_ann.copy()
                    new_ann["bbox"] = bbox
                    new_annotations["annotations"].append(new_ann)
        
        # Redimensionar mosaic de volta para tamanho original
        mosaic = mosaic.resize((w, h), Image.BILINEAR)
        
        # Ajustar anotações para novo tamanho
        scale_x = w / mosaic_w
        scale_y = h / mosaic_h
        for ann in new_annotations["annotations"]:
            bbox = ann["bbox"]
            bbox[0] *= scale_x
            bbox[1] *= scale_y
            bbox[2] *= scale_x
            bbox[3] *= scale_y
        
        return mosaic, new_annotations
    
    def _mixup(self, image: Image.Image, annotations: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, Any]]:
        """MixUp augmentation: combina duas imagens com lambda da distribuição Beta."""
        if len(self._image_cache) < 1:
            return image, annotations
        
        # Selecionar imagem aleatória do cache
        other_image, other_annotations = random.choice(self._image_cache)
        
        # Lambda da distribuição Beta
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        
        # Redimensionar outras imagens se necessário
        if other_image.size != image.size:
            other_image = other_image.resize(image.size, Image.BILINEAR)
        
        # Converter para arrays
        img1 = np.array(image, dtype=np.float32)
        img2 = np.array(other_image, dtype=np.float32)
        
        # Mixup
        mixed_img = (lam * img1 + (1 - lam) * img2).astype(np.uint8)
        mixed_image = Image.fromarray(mixed_img)
        
        # Combinar anotações (manter ambas com pesos)
        new_annotations = {"annotations": []}
        
        # Anotações da primeira imagem (peso lambda)
        if "annotations" in annotations:
            for ann in annotations["annotations"]:
                new_annotations["annotations"].append(ann.copy())
        
        # Anotações da segunda imagem (peso 1-lambda)
        # Nota: Em object detection, geralmente mantemos todas as anotações
        if "annotations" in other_annotations:
            for ann in other_annotations["annotations"]:
                new_annotations["annotations"].append(ann.copy())
        
        return mixed_image, new_annotations
    
    def _copy_paste(self, image: Image.Image, annotations: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, Any]]:
        """Copy-Paste augmentation: copia objetos de outra imagem."""
        if len(self._image_cache) < 1:
            return image, annotations
        
        # Selecionar imagem aleatória do cache
        other_image, other_annotations = random.choice(self._image_cache)
        
        # Redimensionar outras imagens se necessário
        if other_image.size != image.size:
            other_image = other_image.resize(image.size, Image.BILINEAR)
        
        # Converter para arrays
        img1 = np.array(image)
        img2 = np.array(other_image)
        
        # Copiar objetos aleatórios da segunda imagem
        if "annotations" in other_annotations and len(other_annotations["annotations"]) > 0:
            # Selecionar anotação aleatória
            ann_to_copy = random.choice(other_annotations["annotations"])
            bbox = ann_to_copy["bbox"]  # [x, y, w, h]
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # Garantir coordenadas válidas
            x = max(0, min(x, img2.shape[1] - 1))
            y = max(0, min(y, img2.shape[0] - 1))
            w = max(1, min(w, img2.shape[1] - x))
            h = max(1, min(h, img2.shape[0] - y))
            
            # Posição aleatória na primeira imagem
            paste_x = random.randint(0, max(1, img1.shape[1] - w))
            paste_y = random.randint(0, max(1, img1.shape[0] - h))
            
            # Copiar região
            img1[paste_y:paste_y+h, paste_x:paste_x+w] = img2[y:y+h, x:x+w]
            
            # Adicionar anotação
            new_ann = ann_to_copy.copy()
            new_ann["bbox"] = [paste_x, paste_y, w, h]
            
            new_annotations = annotations.copy()
            if "annotations" not in new_annotations:
                new_annotations["annotations"] = []
            new_annotations["annotations"].append(new_ann)
            
            return Image.fromarray(img1), new_annotations
        
        return image, annotations
    
    def _random_erasing(self, image: Image.Image) -> Image.Image:
        """Random Erasing: apaga regiões aleatórias da imagem."""
        w, h = image.size
        img_array = np.array(image)
        
        # Área a ser apagada
        area = w * h
        erase_area = area * random.uniform(*self.erasing_area_ratio)
        aspect_ratio = random.uniform(*self.erasing_aspect_ratio)
        
        # Calcular dimensões
        erase_h = int(np.sqrt(erase_area * aspect_ratio))
        erase_w = int(np.sqrt(erase_area / aspect_ratio))
        
        # Garantir que cabe na imagem
        erase_h = min(erase_h, h)
        erase_w = min(erase_w, w)
        
        # Posição aleatória
        x = random.randint(0, max(1, w - erase_w))
        y = random.randint(0, max(1, h - erase_h))
        
        # Preencher com valor aleatório ou médio
        if random.random() < 0.5:
            # Valor aleatório
            fill_value = tuple(random.randint(0, 255) for _ in range(3))
        else:
            # Valor médio da imagem
            fill_value = tuple(img_array.mean(axis=(0, 1)).astype(np.uint8))
        
        img_array[y:y+erase_h, x:x+erase_w] = fill_value
        
        return Image.fromarray(img_array)
    
    def _motion_blur(self, image: Image.Image) -> Image.Image:
        """Motion Blur: simula movimento da câmera."""
        if not HAS_CV2:
            # Fallback: usar Gaussian blur se cv2 não estiver disponível
            return self._gaussian_blur(image)
        
        img_array = np.array(image)
        
        # Tamanho do kernel
        kernel_size = random.randint(*self.motion_blur_kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Direção do movimento (0-180 graus)
        angle = random.uniform(0, 180)
        
        # Criar kernel de motion blur
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        
        # Rotacionar kernel
        M = cv2.getRotationMatrix2D((kernel_size/2, kernel_size/2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
        
        # Aplicar blur
        blurred = cv2.filter2D(img_array, -1, kernel)
        
        return Image.fromarray(blurred)
    
    def _grid_mask(self, image: Image.Image) -> Image.Image:
        """Grid Mask: remove blocos aleatórios em grade para regularização."""
        w, h = image.size
        img_array = np.array(image)
        
        # Tamanho da grade
        grid_size = random.randint(*self.grid_mask_grid_size)
        ratio = random.uniform(*self.grid_mask_ratio)
        
        # Número de blocos a remover
        num_blocks = int((w // grid_size) * (h // grid_size) * ratio)
        
        # Selecionar blocos aleatórios
        blocks_x = w // grid_size
        blocks_y = h // grid_size
        blocks_to_mask = random.sample(
            [(i, j) for i in range(blocks_x) for j in range(blocks_y)],
            min(num_blocks, blocks_x * blocks_y)
        )
        
        # Preencher blocos selecionados
        mean_color = img_array.mean(axis=(0, 1)).astype(np.uint8)
        for bx, by in blocks_to_mask:
            x1 = bx * grid_size
            y1 = by * grid_size
            x2 = min(x1 + grid_size, w)
            y2 = min(y1 + grid_size, h)
            img_array[y1:y2, x1:x2] = mean_color
        
        return Image.fromarray(img_array)


def create_augmentation_pipeline(
    image_size: int = 1024,
    training: bool = True,
    small_objects: bool = True,
    # Flags de habilitação
    enable_mosaic: bool = True,
    enable_mixup: bool = True,
    enable_cutmix: bool = False,
    enable_copy_paste: bool = True,
    enable_random_erasing: bool = True,
    enable_motion_blur: bool = True,
    enable_grid_mask: bool = True,
    enable_cutout: bool = True,
    enable_hsv: bool = False,
    enable_scale: bool = False,
    enable_translate: bool = False,
    enable_shear: bool = False,
    enable_vflip: bool = False,
    # Probabilidades básicas
    hflip_prob: float = 0.5,
    vflip_prob: float = 0.0,
    color_jitter_prob: float = 0.5,
    gaussian_blur_prob: float = 0.3,
    # Probabilidades avançadas
    mosaic_prob: float = 0.3,
    mixup_prob: float = 0.2,
    cutmix_prob: float = 0.0,
    copy_paste_prob: float = 0.1,
    random_erasing_prob: float = 0.3,
    motion_blur_prob: float = 0.2,
    grid_mask_prob: float = 0.2,
    cutout_prob: float = 0.3,
    # Probabilidades YOLO-style
    hsv_prob: float = 0.0,
    scale_prob: float = 0.0,
    translate_prob: float = 0.0,
    shear_prob: float = 0.0,
    # Parâmetros HSV (YOLO defaults)
    hsv_h: float = 0.015,
    hsv_s: float = 0.7,
    hsv_v: float = 0.4,
    # Parâmetros geométricos (YOLO defaults)
    scale_range: Tuple[float, float] = (0.5, 1.5),
    translate_range: float = 0.1,
    shear_range: float = 0.0,
) -> Optional[AdvancedAugmentation]:
    """
    Cria pipeline de augmentation state-of-the-art para object detection.
    
    Inclui técnicas do YOLO: HSV, Scale, Translation, CutMix, Shear.
    
    Args:
        image_size: Tamanho da imagem
        training: Se está em modo treinamento
        small_objects: Se detecta objetos pequenos (para cutout_size)
        enable_*: Flags para habilitar/desabilitar cada augmentation
        *_prob: Probabilidades de cada augmentation
        hsv_h/s/v: Parâmetros HSV (frações)
        scale_range: Range de escala
        translate_range: Range de translation (fração)
        shear_range: Range de shear em graus
    
    Returns:
        Pipeline de augmentation ou None
    """
    if not training:
        return None
    
    # Cutout size baseado no tipo de objeto
    cutout_size = (20, 60) if small_objects else (50, 150)
    
    return AdvancedAugmentation(
        image_size=image_size,
        # Flips
        hflip_prob=hflip_prob,
        vflip_prob=vflip_prob if enable_vflip else 0.0,
        enable_vflip=enable_vflip,
        # Color
        color_jitter_prob=color_jitter_prob,
        gaussian_blur_prob=gaussian_blur_prob,
        # HSV (YOLO-style)
        hsv_prob=hsv_prob if enable_hsv else 0.0,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        enable_hsv=enable_hsv,
        # Geometric (YOLO-style)
        scale_prob=scale_prob if enable_scale else 0.0,
        scale_range=scale_range,
        enable_scale=enable_scale,
        translate_prob=translate_prob if enable_translate else 0.0,
        translate_range=translate_range,
        enable_translate=enable_translate,
        shear_prob=shear_prob if enable_shear else 0.0,
        shear_range=shear_range,
        enable_shear=enable_shear,
        # Advanced multi-image
        mosaic_prob=mosaic_prob if enable_mosaic else 0.0,
        mixup_prob=mixup_prob if enable_mixup else 0.0,
        cutmix_prob=cutmix_prob if enable_cutmix else 0.0,
        enable_mosaic=enable_mosaic,
        enable_mixup=enable_mixup,
        enable_cutmix=enable_cutmix,
        copy_paste_prob=copy_paste_prob if enable_copy_paste else 0.0,
        enable_copy_paste=enable_copy_paste,
        # Regularization
        random_erasing_prob=random_erasing_prob if enable_random_erasing else 0.0,
        enable_random_erasing=enable_random_erasing,
        motion_blur_prob=motion_blur_prob if enable_motion_blur else 0.0,
        enable_motion_blur=enable_motion_blur,
        grid_mask_prob=grid_mask_prob if enable_grid_mask else 0.0,
        enable_grid_mask=enable_grid_mask,
        cutout_prob=cutout_prob if enable_cutout else 0.0,
        cutout_size=cutout_size,
        enable_cutout=enable_cutout,
    )


# Alias para compatibilidade
SmallObjectAugmentation = AdvancedAugmentation
