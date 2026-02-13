#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para verificar consistência das classes em todos os módulos.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def verify_classes():
    """Verifica se as classes estão corretas em todos os módulos."""
    print("=" * 60)
    print("VERIFICACAO DE CLASSES - PADRAO DO SISTEMA")
    print("=" * 60)
    
    expected_classes = ["ok", "ref", "vazio"]
    expected_num = 3
    
    print(f"\nClasses esperadas: {expected_classes}")
    print(f"Numero esperado: {expected_num}")
    print("\n" + "=" * 60)
    
    # 1. ClassManager
    print("\n1. ClassManager (src/utils/class_manager.py)")
    try:
        from src.utils.class_manager import ClassManager, DEFAULT_CLASSES
        cm = ClassManager()
        classes = cm.class_names
        num = cm.num_classes
        id2label = cm.get_id2label()
        
        print(f"   Classes: {classes}")
        print(f"   Numero: {num}")
        print(f"   ID2Label: {id2label}")
        
        if classes == expected_classes and num == expected_num:
            print("   [OK] ClassManager correto")
        else:
            print(f"   [ERRO] Esperado: {expected_classes}, Recebido: {classes}")
            return False
        
        # Verificar DEFAULT_CLASSES
        default_names = [DEFAULT_CLASSES[i]["name"] for i in sorted(DEFAULT_CLASSES.keys())]
        if default_names == expected_classes:
            print("   [OK] DEFAULT_CLASSES correto")
        else:
            print(f"   [ERRO] DEFAULT_CLASSES incorreto: {default_names}")
            return False
            
    except Exception as e:
        print(f"   [ERRO] {e}")
        return False
    
    # 2. ModelConfig
    print("\n2. ModelConfig (src/core/config.py)")
    try:
        from src.core.config import ModelConfig
        mc = ModelConfig()
        classes = mc.FIXED_CLASSES
        num = mc.NUM_CLASSES
        
        print(f"   FIXED_CLASSES: {classes}")
        print(f"   NUM_CLASSES: {num}")
        
        if classes == expected_classes and num == expected_num:
            print("   [OK] ModelConfig correto")
        else:
            print(f"   [ERRO] Esperado: {expected_classes}, Recebido: {classes}")
            return False
    except Exception as e:
        print(f"   [ERRO] {e}")
        return False
    
    # 3. coco_utils - verificar keywords
    print("\n3. coco_utils (src/coco_utils.py)")
    try:
        from src.coco_utils import create_category_mapping
        
        # Testar com categorias de exemplo
        test_categories = [
            {"id": 0, "name": "ok"},
            {"id": 1, "name": "ref"},
            {"id": 2, "name": "vazio"}
        ]
        
        mapping = create_category_mapping(test_categories, expected_classes)
        print(f"   Mapeamento teste: {mapping}")
        
        if mapping == {0: 0, 1: 1, 2: 2}:
            print("   [OK] Mapeamento correto")
        else:
            print(f"   [ERRO] Mapeamento incorreto: {mapping}")
            return False
    except Exception as e:
        print(f"   [ERRO] {e}")
        return False
    
    # 4. Verificar cores
    print("\n4. Cores das Classes")
    try:
        cm = ClassManager()
        colors_ok = True
        for i, name in enumerate(expected_classes):
            color = cm.get_class_color(i, bgr=False)
            color_bgr = cm.get_class_color(i, bgr=True)
            print(f"   {i} ({name}): RGB={color}, BGR={color_bgr}")
            
            # Verificar se cores estão definidas
            if not color or not color_bgr:
                colors_ok = False
        
        if colors_ok:
            print("   [OK] Cores definidas corretamente")
        else:
            print("   [ERRO] Algumas cores nao estao definidas")
            return False
    except Exception as e:
        print(f"   [ERRO] {e}")
        return False
    
    print("\n" + "=" * 60)
    print("RESUMO")
    print("=" * 60)
    print(f"Classes definidas: {expected_classes}")
    print(f"Numero de classes: {expected_num}")
    print("Status: [OK] Todas as verificacoes passaram!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = verify_classes()
    sys.exit(0 if success else 1)
