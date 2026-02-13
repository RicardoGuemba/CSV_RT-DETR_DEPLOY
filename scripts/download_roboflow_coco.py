#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para download de datasets Roboflow em formato COCO.
Conforme PRD - Sistema de Detecção de Objetos com DETR
"""
import sys
import argparse
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.paths import PathManager
from typing import Optional


def download_roboflow_dataset(
    dataset_url: str,
    output_dir: Optional[str] = None,
    version: Optional[str] = None
) -> Path:
    """
    Baixa dataset do Roboflow em formato COCO.
    
    Args:
        dataset_url: URL do dataset Roboflow
        output_dir: Diretório de saída (opcional)
        version: Versão do dataset (opcional)
    
    Returns:
        Path para o diretório do dataset baixado
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERRO: Biblioteca 'roboflow' não instalada!")
        print("Instale com: pip install roboflow")
        sys.exit(1)
    
    # Extrair informações da URL
    # Formato: https://app.roboflow.com/ds/{workspace}/{project}?key={api_key}
    # ou: https://app.roboflow.com/{workspace}/{project}/{version}?key={api_key}
    
    # Parsear URL
    if "?key=" in dataset_url:
        base_url, api_key = dataset_url.split("?key=")
    else:
        print("ERRO: URL deve conter API key: ?key=...")
        sys.exit(1)
    
    # Tentar extrair workspace e project da URL
    parts = base_url.split("/")
    if "ds" in parts:
        ds_index = parts.index("ds")
        workspace = parts[ds_index + 1] if ds_index + 1 < len(parts) else None
        project = parts[ds_index + 2] if ds_index + 2 < len(parts) else None
    else:
        # Formato alternativo
        workspace = None
        project = None
    
    if not workspace or not project:
        print("ERRO: Não foi possível extrair workspace/project da URL")
        print("Use formato: https://app.roboflow.com/ds/{workspace}/{project}?key={api_key}")
        sys.exit(1)
    
    # Configurar diretório de saída
    path_manager = PathManager()
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = path_manager.datasets_dir / project
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Baixando dataset: {workspace}/{project}")
    print(f"Versão: {version or 'latest'}")
    print(f"Destino: {output_path}")
    
    # Inicializar Roboflow
    rf = Roboflow(api_key=api_key)
    
    # Obter projeto
    project_obj = rf.workspace(workspace).project(project)
    
    # Baixar dataset
    if version:
        dataset = project_obj.version(int(version))
    else:
        dataset = project_obj.version(project_obj.versions[0].version_number)
    
    # Download em formato COCO
    dataset.download("coco", location=str(output_path))
    
    print(f"\n✓ Dataset baixado com sucesso!")
    print(f"  Localização: {output_path}")
    
    return output_path


def main():
    """Função principal do script."""
    parser = argparse.ArgumentParser(
        description="Download de dataset Roboflow em formato COCO"
    )
    
    parser.add_argument(
        "url",
        type=str,
        help="URL do dataset Roboflow (ex: https://app.roboflow.com/ds/workspace/project?key=api_key)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Diretório de saída (padrão: data/datasets/{project_name})"
    )
    
    parser.add_argument(
        "-v", "--version",
        type=str,
        help="Versão do dataset (padrão: última versão)"
    )
    
    args = parser.parse_args()
    
    try:
        output_path = download_roboflow_dataset(
            dataset_url=args.url,
            output_dir=args.output,
            version=args.version
        )
        
        print(f"\nDataset pronto em: {output_path}")
        print("\nEstrutura esperada:")
        print("  {output_path}/")
        print("    train/")
        print("      _annotations.coco.json")
        print("      *.jpg")
        print("    valid/")
        print("      _annotations.coco.json")
        print("      *.jpg")
        print("    test/")
        print("      _annotations.coco.json")
        print("      *.jpg")
        
    except Exception as e:
        print(f"\nERRO ao baixar dataset: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
