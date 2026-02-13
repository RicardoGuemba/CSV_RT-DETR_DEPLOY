#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para verificar categorias do dataset Roboflow.
"""
import json
from pathlib import Path

def check_categories():
    """Verifica e exibe categorias do dataset."""
    dataset_path = Path("data/datasets/dataset_v5/train/_annotations.coco.json")
    
    if not dataset_path.exists():
        print(f"Arquivo nao encontrado: {dataset_path}")
        return
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=" * 60)
    print("CATEGORIAS DO DATASET ROBOFLOW")
    print("=" * 60)
    
    categories = data.get("categories", [])
    for cat in categories:
        print(f"  ID: {cat['id']:3d} | Nome: '{cat['name']}'")
    
    print("=" * 60)
    print(f"Total: {len(categories)} categorias")
    print(f"Total imagens: {len(data.get('images', []))}")
    print(f"Total anotacoes: {len(data.get('annotations', []))}")
    print("=" * 60)
    
    # Contagem de anotacoes por categoria
    print("\nANOTACOES POR CATEGORIA:")
    print("-" * 40)
    
    ann_by_cat = {}
    for ann in data.get("annotations", []):
        cat_id = ann["category_id"]
        ann_by_cat[cat_id] = ann_by_cat.get(cat_id, 0) + 1
    
    for cat in categories:
        cat_id = cat["id"]
        count = ann_by_cat.get(cat_id, 0)
        print(f"  {cat['name']:30s}: {count:5d} anotacoes")
    
    print("=" * 60)
    
    # Mostrar mapeamento proposto
    print("\nMAPEAMENTO PROPOSTO (Roboflow -> Sistema):")
    print("-" * 40)
    
    for cat in categories:
        cat_name = cat["name"].lower().strip()
        cat_id = cat["id"]
        
        # Determinar classe do sistema
        if cat_name == "ok" or "ok" in cat_name or "objects" in cat_name:
            system_class = "ok (ID 0)"
        elif cat_name == "ref" or "ref" in cat_name:
            system_class = "ref (ID 1)"
        elif cat_name == "vazio" or "vazio" in cat_name or "empty" in cat_name:
            system_class = "vazio (ID 2)"
        else:
            system_class = "DESCONHECIDO"
        
        print(f"  '{cat['name']}' (ID {cat_id}) -> {system_class}")
    
    print("=" * 60)

if __name__ == "__main__":
    check_categories()
