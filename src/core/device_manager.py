"""
Gerenciador de dispositivos e GPU.
Sistema DETR - RTX 5080 Optimized
"""
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any
import threading

from .logger import logger


@dataclass
class GPUInfo:
    """Informações da GPU."""
    name: str = "N/A"
    vram_total_gb: float = 0.0
    vram_available_gb: float = 0.0
    vram_used_gb: float = 0.0
    compute_capability: str = "N/A"
    cuda_version: str = "N/A"
    driver_version: str = "N/A"
    is_available: bool = False
    supports_bf16: bool = False
    supports_tf32: bool = False


@dataclass
class SystemInfo:
    """Informações do sistema."""
    torch_version: str = "N/A"
    torchvision_version: str = "N/A"
    transformers_version: str = "N/A"
    opencv_version: str = "N/A"
    python_version: str = "N/A"
    platform: str = "N/A"


class DeviceManager:
    """Gerencia dispositivos CUDA e otimizações."""
    
    _instance: Optional['DeviceManager'] = None
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
        self.gpu_info = GPUInfo()
        self.system_info = SystemInfo()
        self.device = None
        self._torch = None
        
        self._detect_system()
        self._detect_gpu()
    
    def _detect_system(self):
        """Detecta informações do sistema."""
        import sys
        import platform
        
        self.system_info.python_version = sys.version.split()[0]
        self.system_info.platform = platform.platform()
        
        try:
            import torch
            self._torch = torch
            self.system_info.torch_version = torch.__version__
        except ImportError:
            logger.error("PyTorch não instalado!")
            return
        
        try:
            import torchvision
            self.system_info.torchvision_version = torchvision.__version__
        except ImportError:
            pass
        
        try:
            import transformers
            self.system_info.transformers_version = transformers.__version__
        except ImportError:
            pass
        
        try:
            import cv2
            self.system_info.opencv_version = cv2.__version__
        except ImportError:
            pass
        
        logger.section("Informações do Sistema")
        logger.log_dict(asdict(self.system_info))
    
    def _detect_gpu(self):
        """Detecta e configura a GPU."""
        if self._torch is None:
            logger.error("PyTorch não disponível para detecção de GPU")
            return
        
        torch = self._torch
        
        if not torch.cuda.is_available():
            logger.warning("CUDA não disponível. Usando CPU.")
            self.device = torch.device("cpu")
            return
        
        self.gpu_info.is_available = True
        self.device = torch.device("cuda")
        
        # Informações básicas da GPU
        self.gpu_info.name = torch.cuda.get_device_name(0)
        
        # VRAM
        vram_total = torch.cuda.get_device_properties(0).total_memory
        self.gpu_info.vram_total_gb = vram_total / (1024**3)
        
        # Compute capability
        cc_major, cc_minor = torch.cuda.get_device_capability(0)
        self.gpu_info.compute_capability = f"{cc_major}.{cc_minor}"
        
        # CUDA version
        self.gpu_info.cuda_version = torch.version.cuda or "N/A"
        
        # Driver version (via pynvml se disponível)
        try:
            import pynvml
            pynvml.nvmlInit()
            self.gpu_info.driver_version = pynvml.nvmlSystemGetDriverVersion()
            pynvml.nvmlShutdown()
        except Exception:
            self.gpu_info.driver_version = "N/A"
        
        # Suporte a BF16 (CC >= 8.0 para Ampere+, CC >= 12.0 para Blackwell)
        cc_value = float(f"{cc_major}.{cc_minor}")
        self.gpu_info.supports_bf16 = cc_value >= 8.0
        
        # Suporte a TF32 (CC >= 8.0)
        self.gpu_info.supports_tf32 = cc_value >= 8.0
        
        logger.section("Informações da GPU")
        logger.log_dict(asdict(self.gpu_info))
        
        # Verificar se é RTX 5080 (Blackwell)
        if "5080" in self.gpu_info.name or cc_value >= 12.0:
            logger.info("[RTX 5080] Blackwell detectada! Otimizacoes habilitadas.")
    
    def get_vram_status(self) -> Tuple[float, float, float]:
        """Retorna VRAM (total, usada, livre) em GB."""
        if not self.gpu_info.is_available or self._torch is None:
            return (0.0, 0.0, 0.0)
        
        torch = self._torch
        
        try:
            free, total = torch.cuda.mem_get_info(0)
            used = total - free
            return (
                total / (1024**3),
                used / (1024**3),
                free / (1024**3)
            )
        except Exception:
            return (self.gpu_info.vram_total_gb, 0.0, self.gpu_info.vram_total_gb)
    
    def get_vram_tier(self) -> str:
        """Retorna o tier de VRAM para auto-tuning."""
        total, _, free = self.get_vram_status()
        
        if free >= 14.0:  # >= 14GB livre
            return "high"
        elif free >= 6.0:  # 6-14GB livre
            return "medium"
        else:
            return "low"
    
    def enable_tf32(self, enable: bool = True):
        """Habilita/desabilita TF32 para operações matmul."""
        if self._torch is None:
            return
        
        torch = self._torch
        
        if self.gpu_info.supports_tf32:
            torch.backends.cuda.matmul.allow_tf32 = enable
            torch.backends.cudnn.allow_tf32 = enable
            status = "habilitado" if enable else "desabilitado"
            logger.info(f"TF32 {status} para matmul e cudnn")
        else:
            logger.warning("GPU não suporta TF32")
    
    def enable_cudnn_benchmark(self, enable: bool = True):
        """Habilita/desabilita cudnn benchmark."""
        if self._torch is None:
            return
        
        self._torch.backends.cudnn.benchmark = enable
        status = "habilitado" if enable else "desabilitado"
        logger.info(f"cuDNN benchmark {status}")
    
    def get_optimal_precision(self) -> str:
        """Retorna a precisão ótima para a GPU."""
        if self.gpu_info.supports_bf16:
            return "bf16"
        elif self.gpu_info.is_available:
            return "fp16"
        else:
            return "fp32"
    
    def supports_torch_compile(self) -> bool:
        """Verifica se torch.compile está disponível."""
        if self._torch is None:
            return False
        
        try:
            version = self._torch.__version__
            major = int(version.split('.')[0])
            
            # PyTorch 2.x é necessário
            if major < 2:
                return False
            
            # Verificar se estamos no Windows - Triton não tem suporte oficial
            import platform
            if platform.system() == "Windows":
                # Verificar se Triton está instalado (raro no Windows)
                try:
                    import triton
                    return True
                except ImportError:
                    return False
            
            return True
        except Exception:
            return False
    
    def save_hardware_env(self, path: Path):
        """Salva informações de hardware em arquivo JSON."""
        env_data = {
            "gpu": asdict(self.gpu_info),
            "system": asdict(self.system_info),
            "device": str(self.device),
            "vram_status": {
                "total_gb": self.get_vram_status()[0],
                "used_gb": self.get_vram_status()[1],
                "free_gb": self.get_vram_status()[2]
            },
            "optimizations": {
                "optimal_precision": self.get_optimal_precision(),
                "supports_tf32": self.gpu_info.supports_tf32,
                "supports_torch_compile": self.supports_torch_compile(),
                "vram_tier": self.get_vram_tier()
            }
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(env_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Hardware environment salvo em: {path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Retorna todas as informações como dicionário."""
        return {
            "gpu": asdict(self.gpu_info),
            "system": asdict(self.system_info),
            "device": str(self.device)
        }


# Instância global
device_manager = DeviceManager()
