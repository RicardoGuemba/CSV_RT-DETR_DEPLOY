"""
Gerenciador de dispositivos CUDA e otimizações.
Conforme PRD - Sistema de Detecção de Objetos com DETR
"""
import torch
from typing import Tuple, Optional


class DeviceManager:
    """Gerencia dispositivos CUDA e otimizações."""
    
    def __init__(self) -> None:
        """Inicializa DeviceManager."""
        self._cuda_available = torch.cuda.is_available()
        self._device = torch.device("cuda" if self._cuda_available else "cpu")
        
        if self._cuda_available:
            self._setup_cuda_optimizations()
    
    @property
    def cuda_available(self) -> bool:
        """Retorna se CUDA está disponível."""
        return self._cuda_available
    
    @property
    def device(self) -> torch.device:
        """Retorna dispositivo ativo."""
        return self._device
    
    def get_device(self) -> torch.device:
        """Retorna dispositivo ativo (alias para property)."""
        return self.device
    
    def get_status_string(self) -> str:
        """Retorna string de status do dispositivo."""
        if self._cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            return f"CUDA: {gpu_name} ({self.device})"
        return f"CPU: {self.device}"
    
    def validate_for_training(self) -> Tuple[bool, str]:
        """
        Valida se GPU está adequada para treinamento.
        
        Returns:
            Tupla (is_valid, message)
        """
        if not self._cuda_available:
            return False, "CUDA não disponível. Treinamento será muito lento em CPU."
        
        # Verificar VRAM mínima (4GB)
        try:
            _, total = torch.cuda.mem_get_info(0)
            total_gb = total / (1024**3)
            if total_gb < 4.0:
                return False, f"VRAM insuficiente: {total_gb:.1f}GB (mínimo 4GB)"
        except Exception:
            pass
        
        return True, "GPU validada para treinamento"
    
    def clear_cache(self) -> None:
        """Limpa cache da GPU."""
        if self._cuda_available:
            torch.cuda.empty_cache()
    
    def _setup_cuda_optimizations(self) -> None:
        """Configura otimizações CUDA."""
        # TF32 para GPUs Ampere+ (RTX 30xx+) e Blackwell (RTX 50xx)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Benchmark cuDNN para encontrar algoritmos mais rápidos
        torch.backends.cudnn.benchmark = True
        
        # Flash Attention via SDPA (PyTorch 2.0+)
        # Ativado automaticamente se compatível
