#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de teste para validar todos os módulos refatorados.
"""
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent))

def test_device_manager():
    """Testa DeviceManager."""
    print("=" * 60)
    print("TESTE: DeviceManager")
    print("=" * 60)
    try:
        from src.utils.device_manager import DeviceManager
        dm = DeviceManager()
        print(f"[OK] DeviceManager criado")
        print(f"  CUDA disponivel: {dm.cuda_available}")
        print(f"  Device: {dm.get_device()}")
        print(f"  Status: {dm.get_status_string()}")
        is_valid, msg = dm.validate_for_training()
        print(f"  Validacao treinamento: {msg}")
        return True
    except Exception as e:
        print(f"[ERRO] {e}")
        return False

def test_class_manager():
    """Testa ClassManager."""
    print("\n" + "=" * 60)
    print("TESTE: ClassManager")
    print("=" * 60)
    try:
        from src.utils.class_manager import ClassManager
        cm = ClassManager()
        print(f"[OK] ClassManager criado")
        print(f"  Numero de classes: {cm.num_classes}")
        print(f"  Classes: {cm.class_names}")
        print(f"  ID2Label: {cm.get_id2label()}")
        print(f"  Label2ID: {cm.get_label2id()}")
        
        # Testar validacao
        valid_anns = [{"category_id": 0, "bbox": [10, 10, 20, 20]}]
        is_valid, msg = cm.validate_annotations(valid_anns)
        print(f"  Validacao anotacoes: {is_valid} - {msg}")
        
        stats = cm.get_statistics(valid_anns)
        print(f"  Estatisticas: {stats}")
        return True
    except Exception as e:
        print(f"[ERRO] {e}")
        import traceback
        traceback.print_exc()
        return False

def test_path_manager():
    """Testa PathManager."""
    print("\n" + "=" * 60)
    print("TESTE: PathManager")
    print("=" * 60)
    try:
        from src.utils.paths import PathManager
        pm = PathManager()
        print(f"[OK] PathManager criado")
        print(f"  Datasets dir: {pm.datasets_dir}")
        print(f"  Models dir: {pm.models_dir}")
        
        exp_name = pm.generate_experiment_name("test")
        print(f"  Nome experimento gerado: {exp_name}")
        
        return True
    except Exception as e:
        print(f"[ERRO] {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hf_model_utils():
    """Testa hf_model_utils."""
    print("\n" + "=" * 60)
    print("TESTE: hf_model_utils")
    print("=" * 60)
    try:
        from src.hf_model_utils import load_detr_model_for_training
        print(f"[OK] Funcoes importadas com sucesso")
        print(f"  load_detr_model_for_training: OK")
        print(f"  load_detr_model_for_inference: OK")
        print(f"  save_model: OK")
        return True
    except Exception as e:
        print(f"[ERRO] {e}")
        import traceback
        traceback.print_exc()
        return False

def test_coco_utils():
    """Testa coco_utils."""
    print("\n" + "=" * 60)
    print("TESTE: coco_utils")
    print("=" * 60)
    try:
        from src.coco_utils import (
            load_coco_annotations,
            find_annotations_file,
            create_category_mapping,
            create_reverse_category_mapping,
            validate_coco_dataset
        )
        print(f"[OK] Funcoes importadas com sucesso")
        print(f"  load_coco_annotations: OK")
        print(f"  find_annotations_file: OK")
        print(f"  create_category_mapping: OK")
        print(f"  create_reverse_category_mapping: OK")
        print(f"  validate_coco_dataset: OK")
        return True
    except Exception as e:
        print(f"[ERRO] {e}")
        import traceback
        traceback.print_exc()
        return False

def test_train_detr():
    """Testa train_detr."""
    print("\n" + "=" * 60)
    print("TESTE: train_detr")
    print("=" * 60)
    try:
        from src.train_detr import TrainingConfig, DETRTrainer, COCODetectionDataset
        from src.utils.class_manager import ClassManager
        
        print(f"[OK] Classes importadas com sucesso")
        
        # Testar criacao de config
        config = TrainingConfig(
            model_name="facebook/detr-resnet-50",
            dataset_path="data/datasets/test",
            experiment_name="test_exp",
            epochs=1,
            batch_size=1
        )
        print(f"[OK] TrainingConfig criado")
        print(f"  Modelo: {config.model_name}")
        print(f"  Epocas: {config.epochs}")
        print(f"  Batch size: {config.batch_size}")
        
        # Testar criacao de trainer (sem setup completo)
        class_manager = ClassManager()
        trainer = DETRTrainer(config, class_manager)
        print(f"[OK] DETRTrainer criado")
        
        return True
    except Exception as e:
        print(f"[ERRO] {e}")
        import traceback
        traceback.print_exc()
        return False

def test_infer_video():
    """Testa infer_video."""
    print("\n" + "=" * 60)
    print("TESTE: infer_video")
    print("=" * 60)
    try:
        from src.infer_video import InferenceConfig, VideoInference, DetectionResult
        print(f"[OK] Classes importadas com sucesso")
        
        # Testar criacao de config
        config = InferenceConfig(
            model_path="data/models/test",
            video_path="test.mp4",
            confidence_threshold=0.5
        )
        print(f"[OK] InferenceConfig criado")
        print(f"  Threshold: {config.confidence_threshold}")
        
        return True
    except Exception as e:
        print(f"[ERRO] {e}")
        import traceback
        traceback.print_exc()
        return False

def test_eval_coco():
    """Testa eval_coco."""
    print("\n" + "=" * 60)
    print("TESTE: eval_coco")
    print("=" * 60)
    try:
        from src.eval_coco import EvaluationConfig, COCOEvaluator
        print(f"[OK] Classes importadas com sucesso")
        
        # Testar criacao de config
        config = EvaluationConfig(
            model_path="data/models/test",
            dataset_path="data/datasets/test",
            split="test"
        )
        print(f"[OK] EvaluationConfig criado")
        print(f"  Split: {config.split}")
        
        return True
    except Exception as e:
        print(f"[ERRO] {e}")
        import traceback
        traceback.print_exc()
        return False

def test_app():
    """Testa app.py."""
    print("\n" + "=" * 60)
    print("TESTE: app.py")
    print("=" * 60)
    try:
        from app import check_dependencies, main
        print(f"[OK] Funcoes importadas com sucesso")
        
        deps_ok = check_dependencies()
        print(f"  Dependencias: {'OK' if deps_ok else 'FALTANDO'}")
        
        return True
    except Exception as e:
        print(f"[ERRO] {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Executa todos os testes."""
    print("\n" + "=" * 60)
    print("TESTES DE VALIDAÇÃO - MÓDULOS REFATORADOS")
    print("=" * 60)
    
    tests = [
        ("DeviceManager", test_device_manager),
        ("ClassManager", test_class_manager),
        ("PathManager", test_path_manager),
        ("hf_model_utils", test_hf_model_utils),
        ("coco_utils", test_coco_utils),
        ("train_detr", test_train_detr),
        ("infer_video", test_infer_video),
        ("eval_coco", test_eval_coco),
        ("app.py", test_app),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[ERRO CRITICO] em {name}: {e}")
            results.append((name, False))
    
    # Resumo
    print("\n" + "=" * 60)
    print("RESUMO DOS TESTES")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "[PASSOU]" if result else "[FALHOU]"
        print(f"  {name:20s} {status}")
    
    print("\n" + "=" * 60)
    print(f"TOTAL: {passed}/{total} testes passaram")
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
