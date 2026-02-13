"""
Janela de preview de vídeo durante inferência.
Permite controle de threshold em tempo real.
"""
import tkinter as tk
from tkinter import ttk
from typing import Optional, Callable
import threading

try:
    import ttkbootstrap as ttkb
    from ttkbootstrap.constants import *
    USE_TTKBOOTSTRAP = True
except ImportError:
    USE_TTKBOOTSTRAP = False
    ttkb = None


class VideoPreviewWindow:
    """Janela separada para preview de vídeo com controles em tempo real."""
    
    def __init__(
        self,
        parent,
        title: str = "Preview - Detecção em Tempo Real",
        initial_threshold: float = 0.05,
        on_threshold_change: Optional[Callable[[float], None]] = None,
        on_close: Optional[Callable[[], None]] = None
    ):
        """
        Inicializa janela de preview.
        
        Args:
            parent: Janela pai
            title: Título da janela
            initial_threshold: Threshold inicial
            on_threshold_change: Callback quando threshold muda
            on_close: Callback quando janela é fechada
        """
        self.parent = parent
        self.on_threshold_change = on_threshold_change
        self.on_close = on_close
        
        # Criar janela com tema limpo
        if USE_TTKBOOTSTRAP:
            self.window = ttkb.Toplevel(parent)
            self.window.title(title)
        else:
            self.window = tk.Toplevel(parent)
            self.window.title(title)
            # Fundo limpo e profissional
            self.window.configure(bg="#f0f0f0")
        
        self.window.geometry("1280x800")
        self.window.minsize(800, 600)
        
        # Configurar protocolo de fechamento
        self.window.protocol("WM_DELETE_WINDOW", self._on_window_close)
        
        # Variáveis
        self.threshold_var = tk.DoubleVar(value=initial_threshold)
        self.threshold_display_var = tk.StringVar(value=f"{initial_threshold:.0%}")
        self.fps_var = tk.StringVar(value="FPS: --")
        self.frame_var = tk.StringVar(value="Frame: 0/0")
        self.detections_var = tk.StringVar(value="Detecções: 0")
        self.status_var = tk.StringVar(value="Aguardando...")
        
        # Estado
        self.is_open = True
        self.preview_image = None
        self.current_frame = None
        self._update_lock = threading.Lock()
        
        # Criar UI
        self._create_ui()
        
        # Adicionar trace para threshold
        self.threshold_var.trace_add("write", self._on_threshold_change)
        
        # Colocar janela em foco
        self.window.lift()
        self.window.focus_force()
    
    def _create_ui(self):
        """Cria interface da janela."""
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(1, weight=1)
        
        # Frame superior - controles
        self._create_controls_frame()
        
        # Frame central - preview do vídeo
        self._create_preview_frame()
        
        # Frame inferior - estatísticas
        self._create_stats_frame()
    
    def _create_controls_frame(self):
        """Frame de controles."""
        frame = ttk.Frame(self.window, padding=12)
        frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        frame.columnconfigure(1, weight=1)
        
        # Título
        ttk.Label(
            frame,
            text="🎬 Preview em Tempo Real",
            font=("Segoe UI", 14, "bold")
        ).grid(row=0, column=0, columnspan=4, pady=(0, 10))
        
        # Threshold slider
        ttk.Label(frame, text="Threshold:", font=("Segoe UI", 10)).grid(row=1, column=0, sticky="w", padx=5)
        
        threshold_frame = ttk.Frame(frame)
        threshold_frame.grid(row=1, column=1, sticky="ew", padx=10)
        threshold_frame.columnconfigure(0, weight=1)
        
        self.threshold_slider = ttk.Scale(
            threshold_frame,
            variable=self.threshold_var,
            from_=0.01,
            to=1.0,
            orient=tk.HORIZONTAL,
            length=400
        )
        self.threshold_slider.grid(row=0, column=0, sticky="ew")
        
        # Label do threshold
        self.threshold_label = ttk.Label(
            threshold_frame,
            textvariable=self.threshold_display_var,
            font=("Segoe UI", 12, "bold"),
            width=8
        )
        self.threshold_label.grid(row=0, column=1, padx=10)
        
        # Botões de preset de threshold
        preset_frame = ttk.Frame(frame)
        preset_frame.grid(row=1, column=2, padx=10)
        
        presets = [("5%", 0.05), ("10%", 0.10), ("25%", 0.25), ("50%", 0.50)]
        for i, (text, value) in enumerate(presets):
            btn = ttk.Button(
                preset_frame,
                text=text,
                width=5,
                command=lambda v=value: self.set_threshold(v)
            )
            btn.pack(side=tk.LEFT, padx=2)
        
        # Status
        ttk.Label(
            frame,
            textvariable=self.status_var,
            font=("Segoe UI", 10)
        ).grid(row=1, column=3, padx=10)
    
    def _create_preview_frame(self):
        """Frame de preview do vídeo."""
        frame = ttk.LabelFrame(self.window, text="Vídeo", padding=10)
        frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=8)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        
        # Canvas para o vídeo com cores verde pastel
        canvas_bg = "#f1f8e9"  # Verde muito claro
        self.canvas = tk.Canvas(
            frame,
            bg=canvas_bg,
            highlightthickness=2,
            highlightbackground="#a5d6a7",
            relief="flat"
        )
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Texto inicial com cor suave
        placeholder_color = "#66bb6a"  # Verde médio
        self.canvas.create_text(
            400, 300,
            text="Aguardando frames...",
            fill=placeholder_color,
            font=("Segoe UI", 16),
            tags="placeholder"
        )
    
    def _create_stats_frame(self):
        """Frame de estatísticas."""
        frame = ttk.Frame(self.window, padding=12)
        frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        
        # Estatísticas em linha
        stats = [
            ("📊 FPS:", self.fps_var),
            ("🎞️ Frame:", self.frame_var),
            ("🎯 Detecções:", self.detections_var),
        ]
        
        for i, (label, var) in enumerate(stats):
            ttk.Label(
                frame,
                text=label,
                font=("Segoe UI", 10)
            ).pack(side=tk.LEFT, padx=5)
            
            ttk.Label(
                frame,
                textvariable=var,
                font=("Segoe UI", 10, "bold"),
                width=15
            ).pack(side=tk.LEFT, padx=(0, 20))
        
        # Legenda de cores
        legend_frame = ttk.Frame(frame)
        legend_frame.pack(side=tk.RIGHT, padx=10)
        
        colors = [
            ("ok", "#00FF00"),
            ("ref", "#FFA500"),
            ("vazio", "#FF0000")
        ]
        
        ttk.Label(legend_frame, text="Legenda:", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=5)
        
        for name, color in colors:
            # Criar um pequeno canvas para o círculo colorido
            c = tk.Canvas(legend_frame, width=12, height=12, highlightthickness=0)
            c.create_oval(2, 2, 10, 10, fill=color, outline="white")
            c.pack(side=tk.LEFT, padx=2)
            ttk.Label(legend_frame, text=name, font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(0, 10))
    
    def _on_threshold_change(self, *args):
        """Callback quando threshold muda."""
        value = self.threshold_var.get()
        self.threshold_display_var.set(f"{value:.0%}")
        
        if self.on_threshold_change:
            self.on_threshold_change(value)
    
    def _on_window_close(self):
        """Callback quando janela é fechada."""
        self.is_open = False
        if self.on_close:
            self.on_close()
        self.window.destroy()
    
    def set_threshold(self, value: float):
        """Define threshold."""
        self.threshold_var.set(value)
    
    def get_threshold(self) -> float:
        """Retorna threshold atual."""
        return self.threshold_var.get()
    
    def update_frame(self, frame, detections_count: int = 0):
        """
        Atualiza frame no preview.
        
        Args:
            frame: Frame BGR do OpenCV
            detections_count: Número de detecções no frame
        """
        if not self.is_open:
            return
        
        try:
            from PIL import Image, ImageTk
            import cv2
            
            with self._update_lock:
                # Converter BGR para RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Obter tamanho do canvas
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                
                if canvas_width > 10 and canvas_height > 10:
                    h, w = frame_rgb.shape[:2]
                    
                    # Calcular escala mantendo aspect ratio
                    scale = min(canvas_width / w, canvas_height / h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    
                    # Redimensionar
                    frame_resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    
                    # Converter para PhotoImage
                    image = Image.fromarray(frame_resized)
                    photo = ImageTk.PhotoImage(image)
                    
                    # Atualizar na thread principal
                    def update():
                        if not self.is_open:
                            return
                        self.canvas.delete("all")
                        self.canvas.create_image(
                            canvas_width // 2,
                            canvas_height // 2,
                            image=photo,
                            anchor=tk.CENTER
                        )
                        self.preview_image = photo  # Manter referência
                        self.detections_var.set(str(detections_count))
                    
                    self.window.after(0, update)
                    
        except Exception as e:
            pass  # Ignorar erros de preview
    
    def update_stats(self, fps: float, current_frame: int, total_frames: int):
        """Atualiza estatísticas."""
        if not self.is_open:
            return
        
        def update():
            if not self.is_open:
                return
            self.fps_var.set(f"{fps:.1f}")
            self.frame_var.set(f"{current_frame}/{total_frames}")
        
        self.window.after(0, update)
    
    def set_status(self, status: str):
        """Define status."""
        if not self.is_open:
            return
        
        def update():
            if not self.is_open:
                return
            self.status_var.set(status)
        
        self.window.after(0, update)
    
    def close(self):
        """Fecha a janela."""
        self.is_open = False
        try:
            self.window.destroy()
        except:
            pass
