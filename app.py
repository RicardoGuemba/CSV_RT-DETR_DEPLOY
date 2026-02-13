#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ponto de entrada principal da aplicação.
Conforme PRD - Sistema de Detecção de Objetos com DETR
"""
import sys
import os
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent))

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
    print("  SISTEMA DE DETECÇÃO DE OBJETOS COM DETR")
    print("  Conforme PRD - Refatoração Completa")
    print("=" * 60)
    
    # Verificar dependências
    if not check_dependencies():
        sys.exit(1)
    
    # Importar e inicializar
    from src.utils.device_manager import DeviceManager
    from src.utils.paths import PathManager
    
    device_manager = DeviceManager()
    path_manager = PathManager()
    
    # Log de sistema
    print(f"\nPython: {sys.version.split()[0]}")
    print(f"Device: {device_manager.get_status_string()}")
    
    if device_manager.cuda_available:
        import torch
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
    
    # Criar estrutura de diretórios
    path_manager.datasets_dir.mkdir(parents=True, exist_ok=True)
    path_manager.models_dir.mkdir(parents=True, exist_ok=True)
    path_manager.outputs_dir.mkdir(parents=True, exist_ok=True)
    path_manager.predictions_dir.mkdir(parents=True, exist_ok=True)
    path_manager.logs_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[OK] Estrutura de diretorios verificada")
    
    # Iniciar interface gráfica
    print("\nIniciando interface gráfica...")
    
    try:
        from interface_tkinter import MainApplication
        app = MainApplication()
        app.run()
    except ImportError:
        print("\nAVISO: interface_tkinter.py não encontrado.")
        print("Usando interface antiga (src/ui/)...")
        
        from src.ui.main_window import MainWindow
        app = MainWindow()
        app.run()


if __name__ == "__main__":
    main()
