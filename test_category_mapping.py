#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para testar mapeamento de categorias do dataset Roboflow.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_mapping():
    """Testa o mapeamento de categorias."""
    print("=" * 70)
    print("TESTE DE MAPEAMENTO DE CATEGORIAS - ROBOFLOW -> SISTEMA")
    print("=" * 70)
    
    # Importar funcoes
    from src.coco_utils import create_category_mapping, load_coco_annotations
    from src.utils.class_manager import ClassManager
    
    # Carregar dataset
    dataset_path = Path("data/datasets/dataset_v5/train/_annotations.coco.json")
    
    if not dataset_path.exists():
        print(f"ERRO: Dataset nao encontrado: {dataset_path}")
        return False
    
    # Carregar anotacoes
    coco_data = load_coco_annotations(dataset_path)
    categories = coco_data.get("categories", [])
    
    print(f"\nCategorias do dataset Roboflow:")
    print("-" * 50)
    for cat in categories:
        print(f"  ID: {cat['id']} | Nome: '{cat['name']}'")
    
    # ClassManager
    cm = ClassManager()
    system_classes = cm.class_names
    
    print(f"\nClasses do sistema:")
    print("-" * 50)
    for i, name in enumerate(system_classes):
        print(f"  ID: {i} | Nome: '{name}'")
    
    # Criar mapeamento
    print(f"\n" + "=" * 70)
    print("EXECUTANDO MAPEAMENTO...")
    print("=" * 70)
    
    mapping = create_category_mapping(categories, system_classes)
    
    # Verificar mapeamento
    print(f"\n" + "=" * 70)
    print("VERIFICACAO DO MAPEAMENTO")
    print("=" * 70)
    
    expected_mapping = {
        0: 0,  # 'objects-69mY-ok-b5Hm' -> ok (0)
        1: 1,  # 'Ref' -> ref (1)
        2: 0,  # 'ok' -> ok (0)
        3: 2,  # 'vazio' -> vazio (2)
    }
    
    all_correct = True
    for cat in categories:
        cat_id = cat["id"]
        cat_name = cat["name"]
        actual = mapping.get(cat_id)
        expected = expected_mapping.get(cat_id)
        
        status = "[OK]" if actual == expected else "[ERRO]"
        if actual != expected:
            all_correct = False
        
        actual_name = system_classes[actual] if actual is not None else "N/A"
        expected_name = system_classes[expected] if expected is not None else "N/A"
        
        print(f"  {status} '{cat_name}' (ID {cat_id}): "
              f"mapeado para {actual} ({actual_name}), "
              f"esperado {expected} ({expected_name})")
    
    # Testar contagem de anotacoes por classe do sistema
    print(f"\n" + "=" * 70)
    print("CONTAGEM DE ANOTACOES POR CLASSE DO SISTEMA")
    print("=" * 70)
    
    annotations = coco_data.get("annotations", [])
    count_by_system_class = {i: 0 for i in range(len(system_classes))}
    
    for ann in annotations:
        cat_id = ann["category_id"]
        system_id = mapping.get(cat_id)
        if system_id is not None:
            count_by_system_class[system_id] += 1
    
    total = 0
    for sys_id, count in count_by_system_class.items():
        class_name = system_classes[sys_id]
        print(f"  {class_name} (ID {sys_id}): {count:,} anotacoes")
        total += count
    
    print(f"  {'TOTAL':}: {total:,} anotacoes")
    
    # Resultado final
    print(f"\n" + "=" * 70)
    if all_correct:
        print("RESULTADO: [OK] Mapeamento correto!")
    else:
        print("RESULTADO: [ERRO] Mapeamento incorreto!")
    print("=" * 70)
    
    return all_correct

if __name__ == "__main__":
    success = test_mapping()
    sys.exit(0 if success else 1)
