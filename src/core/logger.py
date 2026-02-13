"""
Sistema de logging para arquivo e UI.
Sistema DETR - RTX 5080 Optimized
"""
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, List
from queue import Queue
import threading

from .config import PATHS


class LogHandler(logging.Handler):
    """Handler customizado que envia logs para callbacks."""
    
    def __init__(self, callback: Optional[Callable[[str], None]] = None):
        super().__init__()
        self.callback = callback
        self.log_queue: Queue = Queue()
    
    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_queue.put(msg)
            if self.callback:
                self.callback(msg)
        except Exception:
            self.handleError(record)
    
    def set_callback(self, callback: Callable[[str], None]):
        """Define callback para atualização de UI."""
        self.callback = callback
    
    def get_pending_logs(self) -> List[str]:
        """Retorna logs pendentes na fila."""
        logs = []
        while not self.log_queue.empty():
            logs.append(self.log_queue.get_nowait())
        return logs


class Logger:
    """Sistema de logging centralizado."""
    
    _instance: Optional['Logger'] = None
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
        self._setup_logger()
    
    def _setup_logger(self):
        """Configura o logger principal."""
        # Criar diretório de logs
        PATHS.logs.mkdir(parents=True, exist_ok=True)
        
        # Nome do arquivo de log com timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = PATHS.logs / f"detr_system_{timestamp}.log"
        
        # Configurar logger raiz
        self.logger = logging.getLogger("DETR_System")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        
        # Formato
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Handler para arquivo
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Handler para console (com encoding seguro para Windows)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Filtro para remover emojis problemáticos no console Windows
        class SafeEncodingFilter(logging.Filter):
            def filter(self, record):
                if hasattr(record, 'msg') and isinstance(record.msg, str):
                    # Substituir emojis por equivalentes de texto
                    record.msg = record.msg.encode('ascii', 'replace').decode('ascii')
                return True
        
        console_handler.addFilter(SafeEncodingFilter())
        self.logger.addHandler(console_handler)
        
        # Handler customizado para UI
        self.ui_handler = LogHandler()
        self.ui_handler.setLevel(logging.INFO)
        self.ui_handler.setFormatter(formatter)
        self.logger.addHandler(self.ui_handler)
        
        self.info(f"Logger inicializado. Arquivo: {self.log_file}")
    
    def set_ui_callback(self, callback: Callable[[str], None]):
        """Define callback para atualização da UI."""
        self.ui_handler.set_callback(callback)
    
    def debug(self, msg: str):
        """Log nível DEBUG."""
        self.logger.debug(msg)
    
    def info(self, msg: str):
        """Log nível INFO."""
        self.logger.info(msg)
    
    def warning(self, msg: str):
        """Log nível WARNING."""
        self.logger.warning(msg)
    
    def error(self, msg: str):
        """Log nível ERROR."""
        self.logger.error(msg)
    
    def critical(self, msg: str):
        """Log nível CRITICAL."""
        self.logger.critical(msg)
    
    def separator(self, char: str = "=", length: int = 60):
        """Adiciona separador visual no log."""
        self.info(char * length)
    
    def section(self, title: str):
        """Inicia uma nova seção no log."""
        self.separator()
        self.info(f"  {title.upper()}")
        self.separator()
    
    def log_dict(self, data: dict, title: str = ""):
        """Loga um dicionário formatado."""
        if title:
            self.info(f"{title}:")
        for key, value in data.items():
            self.info(f"  {key}: {value}")
    
    def get_pending_logs(self) -> List[str]:
        """Retorna logs pendentes para atualização de UI."""
        return self.ui_handler.get_pending_logs()


# Instância global
logger = Logger()
