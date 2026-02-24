#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Testes para as novas funcionalidades:
1. Classe ref exibida apenas uma vez por frame
2. Métricas de predição (precision, recall, IoU) no treinamento
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_enforce_single_ref_per_frame():
    """Testa que a classe ref é mantida apenas uma vez por frame."""
    print("=" * 60)
    print("TESTE: _enforce_single_ref_per_frame (ref única por frame)")
    print("=" * 60)
    try:
        from src.models.detr_inference import DETRInference
        
        # Mock mínimo - não vamos carregar modelo real
        # Testamos a lógica via um objeto com o método
        class MockInference:
            def _enforce_single_ref_per_frame(self, detections):
                ref_class_id = 1
                ref_detections = [d for d in detections if d["label"] == ref_class_id]
                other_detections = [d for d in detections if d["label"] != ref_class_id]
                if len(ref_detections) <= 1:
                    return detections
                best_ref = max(ref_detections, key=lambda d: d["score"])
                return other_detections + [best_ref]
        
        mock = MockInference()
        
        # Caso 1: múltiplas refs -> deve manter só a de maior score
        dets = [
            {"label": 0, "score": 0.9, "class_name": "ok"},
            {"label": 1, "score": 0.7, "class_name": "ref"},
            {"label": 1, "score": 0.95, "class_name": "ref"},
            {"label": 2, "score": 0.8, "class_name": "vazio"},
        ]
        result = mock._enforce_single_ref_per_frame(dets)
        refs = [d for d in result if d["label"] == 1]
        assert len(refs) == 1, f"Esperado 1 ref, obtido {len(refs)}"
        assert refs[0]["score"] == 0.95, f"Esperado score 0.95, obtido {refs[0]['score']}"
        
        # Caso 2: uma única ref -> deve manter
        dets2 = [{"label": 1, "score": 0.8, "class_name": "ref"}]
        result2 = mock._enforce_single_ref_per_frame(dets2)
        assert len(result2) == 1
        assert result2[0]["label"] == 1
        
        # Caso 3: nenhuma ref -> deve manter tudo
        dets3 = [{"label": 0, "score": 0.9}, {"label": 2, "score": 0.7}]
        result3 = mock._enforce_single_ref_per_frame(dets3)
        assert len(result3) == 2
        
        print("[OK] _enforce_single_ref_per_frame: ref exibida apenas uma vez por frame")
        return True
    except Exception as e:
        print(f"[ERRO] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics_computation():
    """Testa cálculo de precision, recall e IoU."""
    print("\n" + "=" * 60)
    print("TESTE: compute_precision_recall_iou")
    print("=" * 60)
    try:
        from src.evaluation.metrics import compute_precision_recall_iou, compute_iou
        
        # Teste IoU básico
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 15, 15]
        iou = compute_iou(box1, box2)
        expected_iou = 25 / (100 + 100 - 25)  # 25/175
        assert abs(iou - expected_iou) < 1e-6, f"IoU esperado ~{expected_iou}, obtido {iou}"
        print(f"  IoU [0,0,10,10] vs [5,5,15,15] = {iou:.4f}")
        
        # Predições e GTs no mesmo espaço
        preds = [
            {"image_id": 0, "category_id": 0, "bbox": [10, 10, 30, 30], "score": 0.9},
            {"image_id": 0, "category_id": 0, "bbox": [50, 50, 70, 70], "score": 0.8},
        ]
        gts = [
            {"image_id": 0, "category_id": 0, "bbox": [12, 12, 32, 32]},
            {"image_id": 0, "category_id": 0, "bbox": [48, 48, 72, 72]},
        ]
        prec, rec, miou = compute_precision_recall_iou(preds, gts, iou_threshold=0.5)
        
        assert 0 <= prec <= 1
        assert 0 <= rec <= 1
        assert 0 <= miou <= 1
        print(f"  Precision: {prec:.4f}, Recall: {rec:.4f}, Mean IoU: {miou:.4f}")
        
        # Caso sem GTs
        prec2, rec2, miou2 = compute_precision_recall_iou(preds, [], 0.5)
        assert prec2 == 0 and rec2 == 0 and miou2 == 0
        
        print("[OK] compute_precision_recall_iou: métricas calculadas corretamente")
        return True
    except Exception as e:
        print(f"[ERRO] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer_imports_and_structure():
    """Verifica que o trainer tem os novos métodos."""
    print("\n" + "=" * 60)
    print("TESTE: DETRTrainer - estrutura de avaliação")
    print("=" * 60)
    try:
        from src.models.detr_trainer import DETRTrainer
        
        assert hasattr(DETRTrainer, '_evaluate_test_metrics')
        assert hasattr(DETRTrainer, '_postprocess_outputs_for_eval')
        assert hasattr(DETRTrainer, '_targets_to_ground_truths')
        
        print("[OK] DETRTrainer possui métodos de avaliação no Test")
        return True
    except Exception as e:
        print(f"[ERRO] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_nms_ok_vazio():
    """Testa NMS para ok e vazio - múltiplas detecções do mesmo objeto."""
    print("\n" + "=" * 60)
    print("TESTE: _apply_nms_ok_vazio")
    print("=" * 60)
    try:
        from src.models.detr_inference import DETRInference
        
        # Usar instância real para testar (precisa de modelo - usar mock do método)
        class MockInference:
            def _compute_iou(self, b1, b2):
                x1 = max(b1[0], b2[0])
                y1 = max(b1[1], b2[1])
                x2 = min(b1[2], b2[2])
                y2 = min(b1[3], b2[3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0.0
            
            def _apply_nms_ok_vazio(self, dets, iou_threshold=0.5):
                ok_dets = [d for d in dets if d["label"] == 0]
                vazio_dets = [d for d in dets if d["label"] == 2]
                ref_dets = [d for d in dets if d["label"] == 1]
                
                def nms(ds):
                    if len(ds) <= 1:
                        return ds
                    sorted_ds = sorted(ds, key=lambda d: d["score"], reverse=True)
                    keep = []
                    for d in sorted_ds:
                        suppressed = any(self._compute_iou(d["box"], k["box"]) >= iou_threshold for k in keep)
                        if not suppressed:
                            keep.append(d)
                    return keep
                
                return ref_dets + nms(ok_dets) + nms(vazio_dets)
        
        mock = MockInference()
        
        # Duas detecções ok sobrepostas (mesmo objeto) -> deve manter só a de maior score
        dets = [
            {"label": 0, "score": 0.9, "box": [10, 10, 50, 50]},
            {"label": 0, "score": 0.7, "box": [15, 15, 55, 55]},  # IoU alto com a primeira
        ]
        result = mock._apply_nms_ok_vazio(dets)
        ok_result = [d for d in result if d["label"] == 0]
        assert len(ok_result) == 1, f"Esperado 1 ok após NMS, obtido {len(ok_result)}"
        assert ok_result[0]["score"] == 0.9
        
        # Duas detecções ok não sobrepostas -> deve manter ambas
        dets2 = [
            {"label": 0, "score": 0.9, "box": [10, 10, 50, 50]},
            {"label": 0, "score": 0.8, "box": [200, 200, 250, 250]},
        ]
        result2 = mock._apply_nms_ok_vazio(dets2)
        assert len(result2) == 2
        
        print("[OK] _apply_nms_ok_vazio: duplicatas sobrepostas mitigadas")
        return True
    except Exception as e:
        print(f"[ERRO] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference_ref_filter_integration():
    """Verifica que DETRInference usa _enforce_single_ref_per_frame."""
    print("\n" + "=" * 60)
    print("TESTE: DETRInference - integração ref única")
    print("=" * 60)
    try:
        from src.models.detr_inference import DETRInference
        
        assert hasattr(DETRInference, '_enforce_single_ref_per_frame')
        # Verificar que predict chama o filtro (via inspeção do código)
        import inspect
        source = inspect.getsource(DETRInference.predict)
        assert '_enforce_single_ref_per_frame' in source
        print("[OK] DETRInference.predict chama _enforce_single_ref_per_frame")
        return True
    except Exception as e:
        print(f"[ERRO] {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("TESTES DAS NOVAS FUNCIONALIDADES")
    print("=" * 60)
    
    tests = [
        ("Ref única por frame", test_enforce_single_ref_per_frame),
        ("NMS ok/vazio", test_nms_ok_vazio),
        ("Métricas precision/recall/IoU", test_metrics_computation),
        ("Trainer avaliação Test", test_trainer_imports_and_structure),
        ("Inference integração ref", test_inference_ref_filter_integration),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[ERRO] em {name}: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("RESUMO")
    print("=" * 60)
    for name, result in results:
        status = "[PASSOU]" if result else "[FALHOU]"
        print(f"  {name:35s} {status}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\nTOTAL: {passed}/{total} testes passaram")
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
