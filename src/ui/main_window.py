"""
Janela principal da aplicação.
Sistema DETR - RTX 5080 Optimized
"""
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
from typing import Optional
import sys
import os

# Adicionar path do projeto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    import ttkbootstrap as ttkb
    from ttkbootstrap.constants import *
    USE_TTKBOOTSTRAP = True
except ImportError:
    USE_TTKBOOTSTRAP = False
    ttkb = None

from src.core.logger import logger
from src.core.device_manager import device_manager
from src.core.session import session
from src.core.config import PATHS


class MainWindow:
    """Janela principal do aplicativo."""
    
    def __init__(self):
        self.root = None
        self.notebook = None
        self.status_var = None
        self.gpu_var = None
        self.vram_var = None
        self.log_text = None
        
        # Tabs
        self.dataset_tab = None
        self.training_tab = None
        self.prediction_tab = None
        self.evaluation_tab = None
        
        # Update timer
        self._update_timer = None
        self._running = True
        
    def create_window(self):
        """Cria a janela principal."""
        if USE_TTKBOOTSTRAP:
            self.root = ttkb.Window(
                title="Sistema DETR - Detecção de Objetos (RTX 5080)",
                themename="minty",  # Tema verde pastel suave
                size=(1400, 900)
            )
            # Aplicar estilos adicionais para verde pastel
            style = ttkb.Style()
            style.configure('TFrame', background="#e8f5e9")
            style.configure('TLabelframe', background="#e8f5e9")
            style.configure('TLabelframe.Label', background="#e8f5e9", foreground="#2e7d32", font=("Segoe UI", 10, "bold"))
            style.configure('TLabel', background="#e8f5e9", foreground="#1b5e20")
            style.configure('TNotebook', background="#c8e6c9")
            style.configure('TNotebook.Tab', background="#a5d6a7", foreground="#1b5e20", padding=[15, 8])
            style.map('TNotebook.Tab', background=[('selected', '#e8f5e9'), ('active', '#c8e6c9')])
        else:
            self.root = tk.Tk()
            self.root.title("Sistema DETR - Detecção de Objetos (RTX 5080)")
            self.root.geometry("1400x900")
            
            # Configurar cor de fundo verde pastel suave
            self.root.configure(bg="#e8f5e9")
            
            # Estilo moderno verde pastel
            style = ttk.Style()
            style.theme_use('clam')
            
            # Cores verde pastel para evitar ruídos
            bg_color = "#e8f5e9"       # Verde muito claro (fundo)
            bg_light = "#f1f8e9"       # Verde mais claro (contraste)
            accent = "#66bb6a"         # Verde médio (destaque)
            accent_dark = "#2e7d32"    # Verde escuro (texto destaque)
            text_color = "#1b5e20"     # Verde escuro (texto)
            border_color = "#a5d6a7"   # Verde suave (bordas)
            
            style.configure('TFrame', background=bg_color)
            style.configure('TLabel', background=bg_color, foreground=text_color)
            style.configure('TLabelFrame', background=bg_light, foreground=accent_dark, borderwidth=2, relief="groove")
            style.configure('TLabelFrame.Label', background=bg_light, foreground=accent_dark, font=("Segoe UI", 10, "bold"))
            style.configure('TNotebook', background=bg_color, borderwidth=0)
            style.configure('TNotebook.Tab', background="#a5d6a7", foreground=text_color, padding=[15, 8], font=("Segoe UI", 10))
            style.map('TNotebook.Tab', 
                      background=[('selected', bg_light), ('active', '#c8e6c9')],
                      foreground=[('selected', accent_dark)])
            
            # Botões com destaque
            style.configure('TButton', background=accent, foreground="white", padding=[10, 5], font=("Segoe UI", 9))
            style.map('TButton', background=[('active', accent_dark), ('pressed', accent_dark)])
            
            # Entradas e Spinbox
            style.configure('TEntry', fieldbackground="white", foreground=text_color, bordercolor=border_color)
            style.configure('TSpinbox', fieldbackground="white", foreground=text_color, arrowcolor=accent)
            
            # Checkbuttons
            style.configure('TCheckbutton', background=bg_color, foreground=text_color)
            
            # Scrollbar
            style.configure('TScrollbar', background=border_color, troughcolor=bg_color)
            
            # Combobox
            style.configure('TCombobox', fieldbackground="white", foreground=text_color, selectbackground=accent)
            
            # Scale/Slider
            style.configure('TScale', background=bg_color, troughcolor=border_color)
            
            # Progressbar
            style.configure('TProgressbar', background=accent, troughcolor=bg_light)
        
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Configurar grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)
        self.root.rowconfigure(2, weight=0)
        
        # Criar componentes
        self._create_main_notebook()
        self._create_log_panel()
        self._create_status_bar()
        
        # Configurar callback de log
        logger.set_ui_callback(self._log_callback)
        
        # Iniciar atualizações periódicas
        self._schedule_updates()
        
        logger.info("Interface iniciada")
        self._log_system_info()
        
    def _create_main_notebook(self):
        """Cria o notebook com as abas principais."""
        if USE_TTKBOOTSTRAP:
            self.notebook = ttkb.Notebook(self.root, bootstyle="primary")
        else:
            self.notebook = ttk.Notebook(self.root)
        
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Importar e criar tabs
        from .dataset_tab import DatasetTab
        from .training_tab import TrainingTab
        from .prediction_tab import PredictionTab
        from .evaluation_tab import EvaluationTab
        
        # Dataset Tab
        dataset_frame = ttk.Frame(self.notebook)
        self.notebook.add(dataset_frame, text="📁 Dataset")
        self.dataset_tab = DatasetTab(dataset_frame)
        
        # Training Tab
        training_frame = ttk.Frame(self.notebook)
        self.notebook.add(training_frame, text="🎯 Treinamento")
        self.training_tab = TrainingTab(training_frame, self.dataset_tab)
        
        # Prediction Tab
        prediction_frame = ttk.Frame(self.notebook)
        self.notebook.add(prediction_frame, text="🔍 Predição")
        self.prediction_tab = PredictionTab(prediction_frame)
        
        # Evaluation Tab
        evaluation_frame = ttk.Frame(self.notebook)
        self.notebook.add(evaluation_frame, text="📊 Avaliação")
        self.evaluation_tab = EvaluationTab(evaluation_frame)
        
    def _create_log_panel(self):
        """Cria o painel de log."""
        log_frame = ttk.LabelFrame(self.root, text="📋 Log do Sistema", padding=10)
        log_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        log_frame.columnconfigure(0, weight=1)
        
        # Cores verde pastel para o log
        log_bg = "#f1f8e9"  # Verde muito claro
        log_fg = "#1b5e20"  # Verde escuro
        log_insert = "#66bb6a"  # Verde médio
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame, 
            height=8,
            wrap=tk.WORD,
            font=("Consolas", 9),
            bg=log_bg,
            fg=log_fg,
            insertbackground=log_insert,
            relief="flat",
            borderwidth=1,
            highlightthickness=2,
            highlightbackground="#a5d6a7",
            highlightcolor="#66bb6a"
        )
        self.log_text.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.log_text.config(state=tk.DISABLED)
        
        # Botão limpar
        if USE_TTKBOOTSTRAP:
            clear_btn = ttkb.Button(
                log_frame, 
                text="Limpar Log",
                bootstyle="secondary-outline",
                command=self._clear_log
            )
        else:
            clear_btn = ttk.Button(log_frame, text="Limpar Log", command=self._clear_log)
        clear_btn.grid(row=1, column=0, sticky="e", pady=5)
        
    def _create_status_bar(self):
        """Cria a barra de status."""
        status_frame = ttk.Frame(self.root)
        status_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        status_frame.columnconfigure(1, weight=1)
        
        # Status geral
        self.status_var = tk.StringVar(value="Pronto")
        status_label = ttk.Label(
            status_frame, 
            textvariable=self.status_var,
            font=("Segoe UI", 10, "normal")
        )
        status_label.grid(row=0, column=0, sticky="w", padx=5)
        
        # Separador visual mais suave
        separator = ttk.Separator(status_frame, orient="vertical")
        separator.grid(row=0, column=1, sticky="ns", padx=20)
        
        # Info GPU
        gpu_name = device_manager.gpu_info.name if device_manager.gpu_info.is_available else "CPU"
        self.gpu_var = tk.StringVar(value=f"🖥️ {gpu_name}")
        gpu_label = ttk.Label(
            status_frame, 
            textvariable=self.gpu_var,
            font=("Segoe UI", 10)
        )
        gpu_label.grid(row=0, column=2, sticky="e", padx=10)
        
        # VRAM
        self.vram_var = tk.StringVar(value="VRAM: --")
        vram_label = ttk.Label(
            status_frame, 
            textvariable=self.vram_var,
            font=("Segoe UI", 10)
        )
        vram_label.grid(row=0, column=3, sticky="e", padx=10)
        
        # CUDA
        cuda_text = f"CUDA {device_manager.gpu_info.cuda_version}" if device_manager.gpu_info.is_available else "N/A"
        cuda_label = ttk.Label(
            status_frame,
            text=cuda_text,
            font=("Segoe UI", 10)
        )
        cuda_label.grid(row=0, column=4, sticky="e", padx=10)
        
    def _log_callback(self, msg: str):
        """Callback para atualizar log na UI."""
        if self.log_text is None:
            return
        
        def update():
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, msg + "\n")
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
        
        if threading.current_thread() is threading.main_thread():
            update()
        else:
            self.root.after(0, update)
    
    def _clear_log(self):
        """Limpa o log da UI."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        
    def _update_vram_status(self):
        """Atualiza status de VRAM."""
        if device_manager.gpu_info.is_available:
            total, used, free = device_manager.get_vram_status()
            self.vram_var.set(f"VRAM: {used:.1f}/{total:.1f} GB")
    
    def _schedule_updates(self):
        """Agenda atualizações periódicas."""
        if not self._running:
            return
        
        self._update_vram_status()
        
        # Atualizar status de treinamento
        if session.is_training_active():
            self.status_var.set("🟢 Treinamento em andamento...")
        else:
            self.status_var.set("Pronto")
        
        # Reagendar
        self._update_timer = self.root.after(2000, self._schedule_updates)
    
    def _log_system_info(self):
        """Loga informações do sistema."""
        logger.separator()
        logger.info("🚀 Sistema DETR - Detecção de Objetos")
        logger.info(f"   GPU: {device_manager.gpu_info.name}")
        logger.info(f"   VRAM: {device_manager.gpu_info.vram_total_gb:.1f} GB")
        logger.info(f"   Compute Capability: {device_manager.gpu_info.compute_capability}")
        logger.info(f"   CUDA: {device_manager.gpu_info.cuda_version}")
        logger.info(f"   PyTorch: {device_manager.system_info.torch_version}")
        logger.info(f"   Transformers: {device_manager.system_info.transformers_version}")
        
        # Otimizações suportadas
        logger.info("   Otimizações:")
        logger.info(f"     - BF16: {'✓' if device_manager.gpu_info.supports_bf16 else '✗'}")
        logger.info(f"     - TF32: {'✓' if device_manager.gpu_info.supports_tf32 else '✗'}")
        logger.info(f"     - torch.compile: {'✓' if device_manager.supports_torch_compile() else '✗'}")
        logger.separator()
    
    def _on_close(self):
        """Handler para fechar a janela."""
        self._running = False
        
        if self._update_timer:
            self.root.after_cancel(self._update_timer)
        
        # Verificar se há treinamento em andamento
        if session.is_training_active():
            from tkinter import messagebox
            if messagebox.askyesno(
                "Treinamento em andamento",
                "Há um treinamento em andamento. Deseja realmente sair?\n"
                "O treinamento será cancelado."
            ):
                session.request_stop()
            else:
                self._running = True
                self._schedule_updates()
                return
        
        logger.info("Encerrando aplicação...")
        self.root.destroy()
    
    def set_status(self, message: str):
        """Define mensagem de status."""
        self.status_var.set(message)
    
    def run(self):
        """Inicia o loop principal."""
        self.create_window()
        self.root.mainloop()


def main():
    """Função principal."""
    app = MainWindow()
    app.run()


if __name__ == "__main__":
    main()
