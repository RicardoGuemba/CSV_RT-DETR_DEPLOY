#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sistema DETR - Sistema de Detecção de Objetos
Otimizado para NVIDIA RTX 5080 (Blackwell)

Autor: Sistema gerado automaticamente
Versão: 1.0.0
"""
import sys
import os

# Adicionar src ao path
sys.path.insert(0, os.path.dirname(__file__))

def check_dependencies():
    """Verifica dependências essenciais."""
    missing = []
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")
    
    try:
        import PIL
    except ImportError:
        missing.append("Pillow")
    
    if missing:
        print("=" * 60)
        print("ERRO: Dependências não encontradas!")
        print("=" * 60)
        print(f"\nDependências faltando: {', '.join(missing)}")
        print("\nPor favor, instale as dependências:")
        print("  pip install -r requirements.txt")
        print("\nOu instale manualmente:")
        print(f"  pip install {' '.join(missing)}")
        print("=" * 60)
        return False
    
    return True


def main():
    """Função principal."""
    print("=" * 60)
    print("  SISTEMA DETR - Detecção de Objetos")
    print("  Otimizado para RTX 5080 (Blackwell)")
    print("=" * 60)
    
    # Verificar dependências
    if not check_dependencies():
        sys.exit(1)
    
    # Importar e iniciar aplicação
    from src.core.logger import logger
    from src.core.config import PATHS
    from src.core.device_manager import device_manager
    
    logger.section("INICIALIZANDO SISTEMA DETR")
    
    # Criar estrutura de diretórios
    PATHS.create_all()
    logger.info("Diretórios criados/verificados")
    
    # Log de sistema
    logger.info(f"Python: {sys.version.split()[0]}")
    logger.info(f"Device: {device_manager.device}")
    
    if device_manager.gpu_info.is_available:
        logger.info(f"GPU: {device_manager.gpu_info.name}")
        logger.info(f"VRAM: {device_manager.gpu_info.vram_total_gb:.1f} GB")
        logger.info(f"CUDA: {device_manager.gpu_info.cuda_version}")
        logger.info(f"Compute Capability: {device_manager.gpu_info.compute_capability}")
        
        # Verificar se é RTX 5080
        if "5080" in device_manager.gpu_info.name:
            logger.info("🚀 RTX 5080 detectada! Todas as otimizações disponíveis.")
    else:
        logger.warning("⚠️ GPU não detectada. O sistema funcionará em CPU (mais lento).")
    
    # Iniciar UI
    logger.info("Iniciando interface gráfica...")
    
    from src.ui.main_window import MainWindow
    
    app = MainWindow()
    app.run()


if __name__ == "__main__":
    main()
