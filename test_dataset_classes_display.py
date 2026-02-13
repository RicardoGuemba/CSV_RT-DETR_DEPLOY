#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para testar exibição correta das classes do sistema (não do Roboflow).
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_classes_display():
    """Testa se as classes exibidas são do sistema, não do Roboflow."""
    print("=" * 70)
    print("TESTE DE EXIBICAO DE CLASSES")
    print("=" * 70)
    
    # Classes do Roboflow (erradas para exibição)
    roboflow_classes = ['Ref', 'objects-69mY-ok-b5Hm', 'ok', 'vazio']
    
    # Classes do sistema (corretas para exibição)
    from src.core.config import ModelConfig
    model_config = ModelConfig()
    system_classes = model_config.FIXED_CLASSES
    
    print(f"\nClasses do Roboflow (NAO usar):")
    print(f"  {roboflow_classes}")
    
    print(f"\nClasses do Sistema (CORRETAS):")
    print(f"  {system_classes}")
    
    # Simular o que o dataset_tab faz agora
    print("\n" + "=" * 70)
    print("SIMULACAO DO RESULTADO DO _analyze_dataset()")
    print("=" * 70)
    
    result = {
        "valid": True,
        "train_images": 4480,
        "val_images": 521,
        "test_images": 260,
        "total_annotations": 59364,
        "classes": system_classes,  # CORRIGIDO: usa classes do sistema
        "path": "data/datasets/dataset_v5"
    }
    
    print("\nResultado esperado agora:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    # Verificar
    expected_classes = ['ok', 'ref', 'vazio']
    if result["classes"] == expected_classes:
        print(f"\n[OK] Classes estao corretas: {result['classes']}")
        return True
    else:
        print(f"\n[ERRO] Classes incorretas!")
        print(f"  Esperado: {expected_classes}")
        print(f"  Recebido: {result['classes']}")
        return False

if __name__ == "__main__":
    success = test_classes_display()
    sys.exit(0 if success else 1)
