#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Smoke test: 1 época de treinamento para validar métricas de Test.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Encontrar dataset com Test
DATASETS = [
    Path("data/datasets/dataset_v8"),
    Path("data/datasets/merged_20260223_185706"),
    Path("data/datasets/dataset_v6_20260124_181712"),
]

def find_dataset_with_test():
    base = Path(__file__).parent
    for ds in DATASETS:
        full = base / ds
        test_dir = full / "test"
        ann = test_dir / "_annotations.coco.json"
        if ann.exists():
            return str(full)
    return None


def main():
    dataset_path = find_dataset_with_test()
    if not dataset_path:
        print("AVISO: Nenhum dataset com diretório Test encontrado.")
        print("Pulando smoke test de treinamento.")
        return True  # Não falhar por falta de dataset
    
    print("=" * 60)
    print("SMOKE TEST: Treinamento 1 época + métricas Test")
    print("=" * 60)
    print(f"Dataset: {dataset_path}")
    
    from src.core.session import session, ExperimentConfig
    from src.models.detr_trainer import DETRTrainer
    
    experiment = ExperimentConfig(
        name="smoke_test_metrics",
        model_name="DETR ResNet-50",
        model_id="facebook/detr-resnet-50",
        dataset_path=dataset_path,
        epochs=1,
        batch_size=2,
        num_workers=0,
        use_augmentation=False,
        use_torch_compile=False,
    )
    
    trainer = DETRTrainer(experiment)
    
    # Capturar log para verificar métricas
    import io
    from contextlib import redirect_stdout
    
    print("\nIniciando treinamento (1 época)...")
    trainer.train()
    
    print("\n[OK] Smoke test concluído. Verifique o log acima por:")
    print("  - 'Test Epoch 1 | Precision: ... | Recall: ... | IoU: ...'")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
