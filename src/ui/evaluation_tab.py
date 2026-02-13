"""
Aba de Avaliação COCO.
Sistema DETR - RTX 5080 Optimized
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path
from typing import Optional, Dict, Any
import json

try:
    import ttkbootstrap as ttkb
    from ttkbootstrap.constants import *
    USE_TTKBOOTSTRAP = True
except ImportError:
    USE_TTKBOOTSTRAP = False
    ttkb = None

from src.core.logger import logger
from src.core.config import PATHS


class EvaluationTab:
    """Tab de avaliação com métricas COCO."""
    
    def __init__(self, parent: ttk.Frame):
        self.parent = parent
        
        # Variáveis
        self.model_path_var = tk.StringVar(value="Nenhum modelo selecionado")
        self.dataset_path_var = tk.StringVar(value="Nenhum dataset selecionado")
        self.status_var = tk.StringVar(value="Pronto")
        self.progress_var = tk.DoubleVar(value=0)
        
        # Paths
        self.model_path: Optional[Path] = None
        self.dataset_path: Optional[Path] = None
        
        # Results
        self.evaluation_results: Dict[str, Any] = {}
        
        self._create_ui()
        
    def _create_ui(self):
        """Cria a interface."""
        # Configurar fundo limpo
        if not USE_TTKBOOTSTRAP:
            self.parent.configure(style='TFrame')
        
        self.parent.columnconfigure(0, weight=1)
        self.parent.rowconfigure(2, weight=1)
        
        # Frame de configuração
        self._create_config_frame()
        
        # Frame de controle
        self._create_control_frame()
        
        # Frame de resultados
        self._create_results_frame()
        
    def _create_config_frame(self):
        """Frame de configuração."""
        frame = ttk.LabelFrame(self.parent, text="⚙️ Configuração", padding=15)
        frame.grid(row=0, column=0, sticky="ew", padx=10, pady=8)
        frame.columnconfigure(1, weight=1)
        
        # Modelo
        ttk.Label(frame, text="Modelo:").grid(row=0, column=0, sticky="w", pady=5)
        model_frame = ttk.Frame(frame)
        model_frame.grid(row=0, column=1, sticky="ew", pady=5)
        model_frame.columnconfigure(0, weight=1)
        
        ttk.Label(model_frame, textvariable=self.model_path_var, wraplength=400).grid(
            row=0, column=0, sticky="w"
        )
        
        if USE_TTKBOOTSTRAP:
            model_btn = ttkb.Button(
                model_frame,
                text="📁 Selecionar Experimento",
                bootstyle="primary-outline",
                command=self._browse_model
            )
        else:
            model_btn = ttk.Button(model_frame, text="📁 Selecionar Experimento", command=self._browse_model)
        model_btn.grid(row=0, column=1, padx=5)
        
        # Dataset
        ttk.Label(frame, text="Dataset:").grid(row=1, column=0, sticky="w", pady=5)
        dataset_frame = ttk.Frame(frame)
        dataset_frame.grid(row=1, column=1, sticky="ew", pady=5)
        dataset_frame.columnconfigure(0, weight=1)
        
        ttk.Label(dataset_frame, textvariable=self.dataset_path_var, wraplength=400).grid(
            row=0, column=0, sticky="w"
        )
        
        if USE_TTKBOOTSTRAP:
            dataset_btn = ttkb.Button(
                dataset_frame,
                text="📂 Selecionar Dataset",
                bootstyle="primary-outline",
                command=self._browse_dataset
            )
        else:
            dataset_btn = ttk.Button(dataset_frame, text="📂 Selecionar Dataset", command=self._browse_dataset)
        dataset_btn.grid(row=0, column=1, padx=5)
        
    def _create_control_frame(self):
        """Frame de controle."""
        frame = ttk.LabelFrame(self.parent, text="🎮 Controle", padding=15)
        frame.grid(row=1, column=0, sticky="ew", padx=10, pady=8)
        
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X)
        
        if USE_TTKBOOTSTRAP:
            self.eval_btn = ttkb.Button(
                btn_frame,
                text="📊 Executar Avaliação",
                bootstyle="success",
                command=self._run_evaluation
            )
            self.export_btn = ttkb.Button(
                btn_frame,
                text="💾 Exportar Relatório",
                bootstyle="info",
                command=self._export_report,
                state=tk.DISABLED
            )
        else:
            self.eval_btn = ttk.Button(btn_frame, text="📊 Executar Avaliação", command=self._run_evaluation)
            self.export_btn = ttk.Button(btn_frame, text="💾 Exportar Relatório", command=self._export_report, state=tk.DISABLED)
        
        self.eval_btn.pack(side=tk.LEFT, padx=5)
        self.export_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress
        self.progress_bar = ttk.Progressbar(frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=10)
        
        # Status
        ttk.Label(frame, textvariable=self.status_var, font=("Segoe UI", 10, "bold")).pack()
        
    def _create_results_frame(self):
        """Frame de resultados."""
        frame = ttk.LabelFrame(self.parent, text="📊 Resultados COCO", padding=15)
        frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=8)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(1, weight=1)
        
        # Métricas principais
        main_metrics_frame = ttk.LabelFrame(frame, text="Métricas Principais", padding=12)
        main_metrics_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.main_metrics = {}
        metrics = [
            ("mAP", "mAP (IoU=0.50:0.95)"),
            ("AP50", "AP @ IoU=0.50"),
            ("AP75", "AP @ IoU=0.75"),
            ("AP_small", "AP (small objects)"),
            ("AP_medium", "AP (medium objects)"),
            ("AP_large", "AP (large objects)")
        ]
        
        for i, (key, label) in enumerate(metrics):
            ttk.Label(main_metrics_frame, text=f"{label}:").grid(row=i, column=0, sticky="w", pady=2)
            var = tk.StringVar(value="--")
            self.main_metrics[key] = var
            ttk.Label(
                main_metrics_frame, 
                textvariable=var,
                font=("Consolas", 11, "bold")
            ).grid(row=i, column=1, sticky="e", padx=10, pady=2)
        
        # Recall metrics
        recall_frame = ttk.LabelFrame(frame, text="Métricas de Recall", padding=12)
        recall_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        self.recall_metrics = {}
        recall_items = [
            ("AR1", "AR @ maxDets=1"),
            ("AR10", "AR @ maxDets=10"),
            ("AR100", "AR @ maxDets=100"),
            ("AR_small", "AR (small)"),
            ("AR_medium", "AR (medium)"),
            ("AR_large", "AR (large)")
        ]
        
        for i, (key, label) in enumerate(recall_items):
            ttk.Label(recall_frame, text=f"{label}:").grid(row=i, column=0, sticky="w", pady=2)
            var = tk.StringVar(value="--")
            self.recall_metrics[key] = var
            ttk.Label(
                recall_frame,
                textvariable=var,
                font=("Consolas", 11)
            ).grid(row=i, column=1, sticky="e", padx=10, pady=2)
        
        # Text area para detalhes
        details_frame = ttk.LabelFrame(frame, text="Detalhes por Classe", padding=10)
        details_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        details_frame.columnconfigure(0, weight=1)
        details_frame.rowconfigure(0, weight=1)
        
        # Cores verde pastel para o texto de detalhes
        details_bg = "#f1f8e9"  # Verde muito claro
        details_fg = "#1b5e20"  # Verde escuro
        
        self.details_text = tk.Text(
            details_frame,
            height=10,
            font=("Consolas", 10),
            bg=details_bg,
            fg=details_fg,
            relief="flat",
            borderwidth=1,
            highlightthickness=2,
            highlightbackground="#a5d6a7",
            wrap=tk.WORD
        )
        self.details_text.grid(row=0, column=0, sticky="nsew")
        
        scrollbar = ttk.Scrollbar(details_frame, orient=tk.VERTICAL, command=self.details_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.details_text.config(yscrollcommand=scrollbar.set)
        self.details_text.config(state=tk.DISABLED)
    
    def _browse_model(self):
        """Seleciona experimento/modelo."""
        folder = filedialog.askdirectory(
            title="Selecionar pasta do experimento",
            initialdir=str(PATHS.experiments)
        )
        
        if folder:
            folder_path = Path(folder)
            model_file = folder_path / "best_model.pt"
            
            if model_file.exists():
                self.model_path = folder_path
                self.model_path_var.set(str(folder_path))
                logger.info(f"Experimento selecionado: {folder_path}")
            else:
                pt_files = list(folder_path.glob("*.pt"))
                if pt_files:
                    self.model_path = folder_path
                    self.model_path_var.set(str(folder_path))
                else:
                    messagebox.showerror("Erro", "Nenhum modelo encontrado na pasta!")
    
    def _browse_dataset(self):
        """Seleciona dataset."""
        folder = filedialog.askdirectory(
            title="Selecionar pasta do dataset COCO",
            initialdir=str(PATHS.datasets)
        )
        
        if folder:
            self.dataset_path = Path(folder)
            self.dataset_path_var.set(str(self.dataset_path))
            logger.info(f"Dataset selecionado: {self.dataset_path}")
    
    def _run_evaluation(self):
        """Executa avaliação COCO."""
        if not self.model_path:
            messagebox.showerror("Erro", "Selecione um experimento/modelo!")
            return
        
        if not self.dataset_path:
            messagebox.showerror("Erro", "Selecione um dataset!")
            return
        
        # Atualizar UI
        self.eval_btn.config(state=tk.DISABLED)
        self.export_btn.config(state=tk.DISABLED)
        self.status_var.set("🟡 Executando avaliação...")
        
        # Executar em thread
        thread = threading.Thread(target=self._evaluation_thread, daemon=True)
        thread.start()
    
    def _evaluation_thread(self):
        """Thread de avaliação."""
        try:
            from src.evaluation.coco_evaluator import COCOEvaluator
            
            evaluator = COCOEvaluator(
                model_path=self.model_path,
                dataset_path=self.dataset_path,
                progress_callback=self._update_progress
            )
            
            results = evaluator.evaluate()
            self.evaluation_results = results
            
            # Atualizar UI com resultados
            self._update_results(results)
            
            self._update_status("✅ Avaliação concluída!")
            self.parent.after(0, lambda: self.export_btn.config(state=tk.NORMAL))
            
            logger.info("Avaliação COCO concluída")
            logger.log_dict(results.get("summary", {}), "Resultados COCO")
            
        except Exception as e:
            logger.error(f"Erro na avaliação: {e}")
            self._update_status(f"❌ Erro: {e}")
        finally:
            self.parent.after(0, lambda: self.eval_btn.config(state=tk.NORMAL))
            self.parent.after(0, lambda: self.progress_var.set(0))
    
    def _update_progress(self, value: float):
        """Atualiza progresso."""
        def update():
            self.progress_var.set(value)
        self.parent.after(0, update)
    
    def _update_status(self, status: str):
        """Atualiza status."""
        def update():
            self.status_var.set(status)
        self.parent.after(0, update)
    
    def _update_results(self, results: dict):
        """Atualiza display de resultados."""
        def update():
            summary = results.get("summary", {})
            
            # Métricas principais
            self.main_metrics["mAP"].set(f"{summary.get('mAP', 0)*100:.1f}%")
            self.main_metrics["AP50"].set(f"{summary.get('AP50', 0)*100:.1f}%")
            self.main_metrics["AP75"].set(f"{summary.get('AP75', 0)*100:.1f}%")
            self.main_metrics["AP_small"].set(f"{summary.get('AP_small', 0)*100:.1f}%")
            self.main_metrics["AP_medium"].set(f"{summary.get('AP_medium', 0)*100:.1f}%")
            self.main_metrics["AP_large"].set(f"{summary.get('AP_large', 0)*100:.1f}%")
            
            # Recall
            self.recall_metrics["AR1"].set(f"{summary.get('AR1', 0)*100:.1f}%")
            self.recall_metrics["AR10"].set(f"{summary.get('AR10', 0)*100:.1f}%")
            self.recall_metrics["AR100"].set(f"{summary.get('AR100', 0)*100:.1f}%")
            self.recall_metrics["AR_small"].set(f"{summary.get('AR_small', 0)*100:.1f}%")
            self.recall_metrics["AR_medium"].set(f"{summary.get('AR_medium', 0)*100:.1f}%")
            self.recall_metrics["AR_large"].set(f"{summary.get('AR_large', 0)*100:.1f}%")
            
            # Detalhes por classe
            self.details_text.config(state=tk.NORMAL)
            self.details_text.delete(1.0, tk.END)
            
            per_class = results.get("per_class", {})
            if per_class:
                self.details_text.insert(tk.END, "=" * 50 + "\n")
                self.details_text.insert(tk.END, "  MÉTRICAS POR CLASSE\n")
                self.details_text.insert(tk.END, "=" * 50 + "\n\n")
                
                for cls_name, metrics in per_class.items():
                    self.details_text.insert(tk.END, f"Classe: {cls_name}\n")
                    self.details_text.insert(tk.END, f"  AP: {metrics.get('AP', 0)*100:.1f}%\n")
                    self.details_text.insert(tk.END, f"  Precision: {metrics.get('precision', 0)*100:.1f}%\n")
                    self.details_text.insert(tk.END, f"  Recall: {metrics.get('recall', 0)*100:.1f}%\n")
                    self.details_text.insert(tk.END, "-" * 30 + "\n")
            
            self.details_text.config(state=tk.DISABLED)
        
        self.parent.after(0, update)
    
    def _export_report(self):
        """Exporta relatório de avaliação."""
        if not self.evaluation_results:
            messagebox.showerror("Erro", "Execute uma avaliação primeiro!")
            return
        
        # Selecionar destino
        output_path = filedialog.asksaveasfilename(
            title="Salvar relatório",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("Todos", "*.*")],
            initialdir=str(PATHS.reports)
        )
        
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Relatório exportado: {output_path}")
                messagebox.showinfo("Sucesso", f"Relatório salvo em:\n{output_path}")
                
            except Exception as e:
                logger.error(f"Erro ao exportar: {e}")
                messagebox.showerror("Erro", f"Erro ao salvar:\n{e}")
