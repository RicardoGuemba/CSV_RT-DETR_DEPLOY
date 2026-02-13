"""
Aba de gerenciamento de Dataset - SIMPLIFICADA.
Sistema DETR - RTX 5080 Optimized
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

try:
    import ttkbootstrap as ttkb
    from ttkbootstrap.constants import *
    USE_TTKBOOTSTRAP = True
except ImportError:
    USE_TTKBOOTSTRAP = False
    ttkb = None

from src.core.logger import logger
from src.core.config import PATHS, Config
from src.core.user_settings import settings_manager


class DatasetTab:
    """Tab para gerenciamento de dataset - Interface Simplificada."""
    
    def __init__(self, parent: ttk.Frame):
        self.parent = parent
        self.dataset_path: Optional[Path] = None
        self.dataset_info: Dict[str, Any] = {}
        self.config = Config()
        
        # Carregar configurações salvas
        saved = settings_manager.dataset
        
        # Variáveis de controle
        self.dataset_url_var = tk.StringVar(value=saved.roboflow_url or self.config.roboflow.DATASET_URL)
        self.version_var = tk.StringVar(value=saved.roboflow_version or self.config.roboflow.VERSION)
        self.download_progress = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Aguardando...")
        
        # Restaurar path salvo
        if saved.dataset_path and Path(saved.dataset_path).exists():
            self.dataset_path = Path(saved.dataset_path)
        
        self._create_ui()
        
        # Carregar lista de datasets disponíveis
        self._refresh_datasets_list()
        
        # Validar dataset automaticamente se existir
        if self.dataset_path and self.dataset_path.exists():
            self.parent.after(300, self._validate_dataset)
            # Atualizar nome do dataset atual na UI
            self.current_dataset_var.set(self.dataset_path.name)
        else:
            self.current_dataset_var.set("Nenhum selecionado")
    
    def _create_ui(self):
        """Cria a interface SIMPLIFICADA."""
        self.parent.columnconfigure(0, weight=1)
        self.parent.rowconfigure(1, weight=1)
        
        # ======== SEÇÃO 1: BAIXAR NOVO DATASET ========
        download_frame = ttk.LabelFrame(self.parent, text="1. BAIXAR NOVO DATASET", padding=10)
        download_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        download_frame.columnconfigure(1, weight=1)
        
        # URL
        ttk.Label(download_frame, text="URL Roboflow:").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        ttk.Entry(download_frame, textvariable=self.dataset_url_var, width=70).grid(
            row=0, column=1, columnspan=2, sticky="ew", padx=5, pady=3
        )
        
        # Versão e Botão Download na mesma linha
        ttk.Label(download_frame, text="Versão:").grid(row=1, column=0, sticky="w", padx=5, pady=3)
        
        version_btn_frame = ttk.Frame(download_frame)
        version_btn_frame.grid(row=1, column=1, columnspan=2, sticky="w", pady=3)
        
        ttk.Entry(version_btn_frame, textvariable=self.version_var, width=8).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            version_btn_frame, 
            text="⬇ BAIXAR DATASET",
            command=self._download_dataset
        ).pack(side=tk.LEFT, padx=20)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(download_frame, variable=self.download_progress, maximum=100)
        self.progress_bar.grid(row=2, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        
        # ======== SEÇÃO 2: DATASETS DISPONÍVEIS ========
        datasets_frame = ttk.LabelFrame(self.parent, text="2. DATASETS DISPONÍVEIS (selecione um)", padding=10)
        datasets_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        datasets_frame.columnconfigure(0, weight=1)
        datasets_frame.rowconfigure(0, weight=1)
        
        # Lista de datasets
        list_frame = ttk.Frame(datasets_frame)
        list_frame.grid(row=0, column=0, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Listbox com datasets
        self.datasets_listbox = tk.Listbox(
            list_frame,
            height=8,
            font=("Consolas", 11),
            bg="#f1f8e9",
            fg="#1b5e20",
            selectbackground="#66bb6a",
            selectforeground="white",
            yscrollcommand=scrollbar.set
        )
        self.datasets_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.datasets_listbox.yview)
        
        # Bind para seleção
        self.datasets_listbox.bind('<<ListboxSelect>>', self._on_dataset_select)
        
        # Botões de ação
        btn_frame = ttk.Frame(datasets_frame)
        btn_frame.grid(row=1, column=0, pady=10)
        
        ttk.Button(btn_frame, text="🔄 Atualizar Lista", command=self._refresh_datasets_list).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="📁 Selecionar Pasta", command=self._browse_dataset).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="✓ Usar Dataset Selecionado", command=self._use_selected_dataset).pack(side=tk.LEFT, padx=5)
        
        # ======== SEÇÃO 3: DATASET ATUAL ========
        current_frame = ttk.LabelFrame(self.parent, text="3. DATASET ATUAL (usado para treinamento)", padding=10)
        current_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        current_frame.columnconfigure(1, weight=1)
        
        # Status
        ttk.Label(current_frame, text="Status:", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w", padx=5)
        ttk.Label(current_frame, textvariable=self.status_var, font=("Segoe UI", 10, "bold")).grid(
            row=0, column=1, sticky="w", padx=5
        )
        
        # Dataset selecionado
        ttk.Label(current_frame, text="Dataset:", font=("Segoe UI", 9, "bold")).grid(row=1, column=0, sticky="w", padx=5)
        self.current_dataset_var = tk.StringVar(value="Nenhum")
        ttk.Label(
            current_frame, 
            textvariable=self.current_dataset_var, 
            font=("Consolas", 11, "bold"),
            foreground="#1565c0"
        ).grid(row=1, column=1, sticky="w", padx=5)
        
        # Estatísticas em linha
        stats_frame = ttk.Frame(current_frame)
        stats_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=10)
        
        self.stats_labels = {}
        stats = [
            ("train", "Train:"),
            ("valid", "Valid:"),
            ("test", "Test:"),
            ("annotations", "Anotações:")
        ]
        
        for i, (key, label) in enumerate(stats):
            ttk.Label(stats_frame, text=label).grid(row=0, column=i*2, padx=(10, 2))
            var = tk.StringVar(value="--")
            self.stats_labels[key] = var
            ttk.Label(stats_frame, textvariable=var, font=("Segoe UI", 10, "bold")).grid(row=0, column=i*2+1, padx=(0, 10))
        
        # Classes
        ttk.Label(current_frame, text="Classes:", font=("Segoe UI", 9, "bold")).grid(row=3, column=0, sticky="w", padx=5)
        ttk.Label(current_frame, text="ok, ref, vazio", font=("Segoe UI", 10)).grid(row=3, column=1, sticky="w", padx=5)
    
    def _refresh_datasets_list(self):
        """Atualiza lista de datasets disponíveis."""
        self.datasets_listbox.delete(0, tk.END)
        
        datasets_dir = PATHS.datasets
        if not datasets_dir.exists():
            datasets_dir.mkdir(parents=True, exist_ok=True)
        
        datasets = []
        current_index = -1  # Para selecionar automaticamente o dataset atual
        
        for item in datasets_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Verificar se tem arquivos de anotação
                has_train = (item / "train" / "_annotations.coco.json").exists()
                has_valid = (item / "valid" / "_annotations.coco.json").exists() or \
                           (item / "val" / "_annotations.coco.json").exists()
                
                # Contar imagens
                train_count = len(list((item / "train").glob("*.jpg"))) if (item / "train").exists() else 0
                valid_count = len(list((item / "valid").glob("*.jpg"))) if (item / "valid").exists() else \
                             len(list((item / "val").glob("*.jpg"))) if (item / "val").exists() else 0
                test_count = len(list((item / "test").glob("*.jpg"))) if (item / "test").exists() else 0
                
                total = train_count + valid_count + test_count
                
                status = "[OK]" if has_train else "?"
                
                # Marcar dataset atual
                is_current = self.dataset_path and item == self.dataset_path
                current_marker = " << ATUAL" if is_current else ""
                
                datasets.append({
                    "name": item.name,
                    "path": item,
                    "train": train_count,
                    "valid": valid_count,
                    "test": test_count,
                    "total": total,
                    "status": status,
                    "current": current_marker,
                    "is_current": is_current
                })
        
        # Ordenar por nome
        datasets.sort(key=lambda x: x["name"])
        
        # Adicionar à listbox e encontrar índice do atual
        for i, ds in enumerate(datasets):
            display = f"{ds['status']} {ds['name']}  |  Train: {ds['train']}  Valid: {ds['valid']}  Test: {ds['test']}{ds['current']}"
            self.datasets_listbox.insert(tk.END, display)
            if ds["is_current"]:
                current_index = i
        
        # Guardar referência
        self.available_datasets = datasets
        
        # AUTO-SELECIONAR o dataset atual na lista
        if current_index >= 0:
            self.datasets_listbox.selection_clear(0, tk.END)
            self.datasets_listbox.selection_set(current_index)
            self.datasets_listbox.see(current_index)  # Scroll para mostrar
        
        logger.info(f"Encontrados {len(datasets)} datasets em {datasets_dir}")
    
    def _on_dataset_select(self, event):
        """Callback quando um dataset é selecionado na lista."""
        selection = self.datasets_listbox.curselection()
        if selection and hasattr(self, 'available_datasets'):
            idx = selection[0]
            if idx < len(self.available_datasets):
                ds = self.available_datasets[idx]
                # Mostrar preview das estatísticas
                self.stats_labels["train"].set(str(ds["train"]))
                self.stats_labels["valid"].set(str(ds["valid"]))
                self.stats_labels["test"].set(str(ds["test"]))
    
    def _use_selected_dataset(self):
        """Usa o dataset selecionado na lista."""
        selection = self.datasets_listbox.curselection()
        if not selection:
            messagebox.showwarning("Aviso", "Selecione um dataset na lista primeiro!")
            return
        
        if not hasattr(self, 'available_datasets'):
            return
        
        idx = selection[0]
        if idx < len(self.available_datasets):
            ds = self.available_datasets[idx]
            self.dataset_path = ds["path"]
            
            # Salvar configuração
            self._save_settings_auto()
            
            # Validar
            self._validate_dataset()
            
            # Atualizar lista
            self._refresh_datasets_list()
            
            logger.info(f"Dataset selecionado: {ds['name']}")
    
    def _download_dataset(self):
        """Baixa dataset do Roboflow."""
        dataset_url = self.dataset_url_var.get().strip()
        version = self.version_var.get().strip()
        
        if not dataset_url:
            messagebox.showerror("Erro", "Informe a URL do dataset Roboflow")
            return
        
        if not version:
            messagebox.showerror("Erro", "Informe a versão do dataset")
            return
        
        # Criar nome único baseado na versão e timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = f"dataset_v{version}"
        
        # Verificar se já existe
        download_dir = PATHS.datasets / dataset_name
        if download_dir.exists():
            # Perguntar se quer sobrescrever ou criar novo
            result = messagebox.askyesnocancel(
                "Dataset Existente",
                f"O dataset '{dataset_name}' já existe.\n\n"
                f"SIM = Sobrescrever\n"
                f"NÃO = Criar novo com timestamp\n"
                f"CANCELAR = Abortar"
            )
            
            if result is None:  # Cancelar
                return
            elif result is False:  # Criar novo
                dataset_name = f"dataset_v{version}_{timestamp}"
                download_dir = PATHS.datasets / dataset_name
        
        def download_thread():
            try:
                logger.info(f"Baixando dataset: {dataset_name}")
                self._update_status(f"Baixando {dataset_name}...")
                self.download_progress.set(10)
                
                import requests
                import zipfile
                import io
                
                download_dir.mkdir(parents=True, exist_ok=True)
                
                self.download_progress.set(30)
                
                # Baixar
                logger.info(f"URL: {dataset_url}")
                response = requests.get(dataset_url, stream=True, timeout=120)
                response.raise_for_status()
                
                self.download_progress.set(60)
                
                # Extrair
                logger.info("Extraindo arquivos...")
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                    zip_ref.extractall(download_dir)
                
                self.download_progress.set(90)
                
                # Definir como dataset atual
                self.dataset_path = download_dir
                self._save_settings_auto()
                
                self.download_progress.set(100)
                self._update_status(f"✅ {dataset_name} baixado!")
                logger.info(f"Dataset salvo em: {download_dir}")
                
                # Atualizar lista e validar
                self.parent.after(500, self._refresh_datasets_list)
                self.parent.after(1000, self._validate_dataset)
                
            except Exception as e:
                logger.error(f"Erro no download: {e}")
                self._update_status(f"❌ Erro: {e}")
                messagebox.showerror("Erro", f"Falha no download:\n{e}")
            finally:
                self.download_progress.set(0)
        
        thread = threading.Thread(target=download_thread, daemon=True)
        thread.start()
    
    def _browse_dataset(self):
        """Seleciona pasta de dataset manualmente."""
        folder = filedialog.askdirectory(
            title="Selecionar pasta do Dataset",
            initialdir=str(PATHS.datasets)
        )
        
        if folder:
            self.dataset_path = Path(folder)
            self._save_settings_auto()
            self._validate_dataset()
            self._refresh_datasets_list()
            logger.info(f"Dataset selecionado: {self.dataset_path}")
    
    def _validate_dataset(self):
        """Valida o dataset atual."""
        if self.dataset_path is None:
            self._update_status("❌ Nenhum dataset selecionado")
            self.current_dataset_var.set("Nenhum")
            return
        
        if not self.dataset_path.exists():
            self._update_status("❌ Pasta não existe")
            self.current_dataset_var.set("Inválido")
            return
        
        logger.info(f"Validando: {self.dataset_path}")
        self._update_status("Validando...")
        
        def validate_thread():
            try:
                info = self._analyze_dataset()
                
                if info.get("valid"):
                    self.dataset_info = info
                    
                    # Atualizar UI
                    def update_ui():
                        self.current_dataset_var.set(self.dataset_path.name)
                        self.stats_labels["train"].set(str(info.get("train_images", 0)))
                        self.stats_labels["valid"].set(str(info.get("val_images", 0)))
                        self.stats_labels["test"].set(str(info.get("test_images", 0)))
                        self.stats_labels["annotations"].set(str(info.get("total_annotations", 0)))
                        self._update_status("✅ Dataset válido")
                    
                    self.parent.after(0, update_ui)
                    logger.info("Dataset validado com sucesso")
                    logger.log_dict(info, "Estatísticas")
                else:
                    self._update_status(f"❌ {info.get('error', 'Inválido')}")
                    
            except Exception as e:
                logger.error(f"Erro na validação: {e}")
                self._update_status(f"❌ Erro: {e}")
        
        thread = threading.Thread(target=validate_thread, daemon=True)
        thread.start()
    
    def _analyze_dataset(self) -> Dict[str, Any]:
        """Analisa estrutura do dataset COCO."""
        from ..core.config import ModelConfig
        model_config = ModelConfig()
        
        result = {
            "valid": False,
            "train_images": 0,
            "val_images": 0,
            "test_images": 0,
            "total_annotations": 0,
            "classes": model_config.FIXED_CLASSES
        }
        
        if not self.dataset_path or not self.dataset_path.exists():
            result["error"] = "Caminho não existe"
            return result
        
        splits = ["train", "valid", "val", "test"]
        
        for split in splits:
            split_dir = self.dataset_path / split
            if not split_dir.exists():
                continue
            
            # Procurar anotações
            ann_file = split_dir / "_annotations.coco.json"
            if not ann_file.exists():
                ann_file = split_dir / "annotations.json"
            
            if ann_file.exists():
                try:
                    with open(ann_file, 'r', encoding='utf-8') as f:
                        coco_data = json.load(f)
                    
                    num_images = len(coco_data.get("images", []))
                    num_annotations = len(coco_data.get("annotations", []))
                    
                    if split == "train":
                        result["train_images"] = num_images
                    elif split in ["valid", "val"]:
                        result["val_images"] = num_images
                    elif split == "test":
                        result["test_images"] = num_images
                    
                    result["total_annotations"] += num_annotations
                    
                except Exception as e:
                    logger.warning(f"Erro ao ler {split}: {e}")
        
        result["valid"] = result["train_images"] > 0
        result["path"] = str(self.dataset_path)
        
        return result
    
    def _update_status(self, status: str):
        """Atualiza status."""
        def update():
            self.status_var.set(status)
        self.parent.after(0, update)
    
    def _save_settings_auto(self):
        """Salva configurações automaticamente."""
        settings_manager.update_section(
            "dataset",
            dataset_path=str(self.dataset_path) if self.dataset_path else "",
            roboflow_url=self.dataset_url_var.get(),
            roboflow_version=self.version_var.get()
        )
    
    def get_dataset_path(self) -> Optional[Path]:
        """Retorna o caminho do dataset selecionado."""
        return self.dataset_path
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Retorna informações do dataset."""
        return self.dataset_info
    
    def is_dataset_valid(self) -> bool:
        """Verifica se dataset é válido."""
        return self.dataset_info.get("valid", False)
