#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para testar carregamento completo do dataset para treinamento DETR.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_dataset_loading():
    """Testa carregamento do dataset para treinamento."""
    print("=" * 70)
    print("TESTE DE CARREGAMENTO DO DATASET PARA DETR")
    print("=" * 70)
    
    # Importar
    from src.train_detr import COCODetectionDataset, TrainingConfig
    from src.utils.class_manager import ClassManager
    from transformers import DetrImageProcessor
    
    # Configuracao
    dataset_path = Path("data/datasets/dataset_v5")
    
    if not dataset_path.exists():
        print(f"ERRO: Dataset nao encontrado: {dataset_path}")
        return False
    
    # Criar image processor
    print("\n1. Criando image processor...")
    image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    print("   [OK] Image processor criado")
    
    # Criar class manager
    print("\n2. Criando class manager...")
    class_manager = ClassManager()
    print(f"   [OK] Classes: {class_manager.class_names}")
    print(f"   [OK] ID2Label: {class_manager.get_id2label()}")
    
    # Criar dataset de treino
    print("\n3. Carregando dataset de treino...")
    try:
        train_dataset = COCODetectionDataset(
            images_dir=str(dataset_path / "train"),
            annotations_file=str(dataset_path / "train" / "_annotations.coco.json"),
            image_processor=image_processor,
            image_size=640,
            class_manager=class_manager
        )
        print(f"   [OK] Dataset carregado: {len(train_dataset)} imagens")
        print(f"   [OK] Mapeamento: {train_dataset.category_mapping}")
    except Exception as e:
        print(f"   [ERRO] Falha ao carregar dataset: {e}")
        return False
    
    # Testar um item
    print("\n4. Testando carregamento de um item...")
    try:
        item = train_dataset[0]
        print(f"   [OK] Item carregado com sucesso")
        print(f"   - Tipo: {type(item)}")
        if isinstance(item, dict):
            print(f"   - Chaves: {list(item.keys())}")
            if 'labels' in item:
                labels = item['labels']
                if isinstance(labels, dict):
                    print(f"   - Labels chaves: {list(labels.keys())}")
                    if 'class_labels' in labels:
                        unique_classes = set(labels['class_labels'].tolist())
                        print(f"   - Classes unicas: {unique_classes}")
    except Exception as e:
        print(f"   [ERRO] Falha ao carregar item: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Testar validacao
    print("\n5. Carregando dataset de validacao...")
    try:
        valid_dataset = COCODetectionDataset(
            images_dir=str(dataset_path / "valid"),
            annotations_file=str(dataset_path / "valid" / "_annotations.coco.json"),
            image_processor=image_processor,
            image_size=640,
            class_manager=class_manager
        )
        print(f"   [OK] Dataset de validacao: {len(valid_dataset)} imagens")
    except Exception as e:
        print(f"   [ERRO] Falha ao carregar validacao: {e}")
        return False
    
    # Verificar distribuicao de classes
    print("\n6. Verificando distribuicao de classes no dataset...")
    try:
        from collections import Counter
        class_counts = Counter()
        
        # Amostrar algumas imagens
        sample_size = min(100, len(train_dataset))
        for i in range(sample_size):
            item = train_dataset[i]
            if isinstance(item, dict) and 'labels' in item:
                labels = item['labels']
                if isinstance(labels, dict) and 'class_labels' in labels:
                    for cls in labels['class_labels'].tolist():
                        class_counts[cls] += 1
        
        print(f"   Distribuicao em {sample_size} imagens amostradas:")
        for cls_id in sorted(class_counts.keys()):
            cls_name = class_manager.get_class_name(cls_id)
            count = class_counts[cls_id]
            print(f"   - {cls_name} (ID {cls_id}): {count} anotacoes")
        
        print("   [OK] Distribuicao verificada")
    except Exception as e:
        print(f"   [ERRO] Falha ao verificar distribuicao: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("RESULTADO: [OK] Dataset carregado e validado com sucesso!")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    success = test_dataset_loading()
    sys.exit(0 if success else 1)
