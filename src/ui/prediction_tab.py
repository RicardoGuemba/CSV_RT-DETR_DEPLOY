"""
Aba de Predição/Inferência.
Sistema DETR - RTX 5080 Optimized
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path
from typing import Optional
import time

try:
    import ttkbootstrap as ttkb
    from ttkbootstrap.constants import *
    USE_TTKBOOTSTRAP = True
except ImportError:
    USE_TTKBOOTSTRAP = False
    ttkb = None

from src.core.logger import logger
from src.core.config import PATHS
from src.core.device_manager import device_manager
from src.core.user_settings import settings_manager


class PredictionTab:
    """Tab de predição/inferência."""
    
    def __init__(self, parent: ttk.Frame):
        self.parent = parent
        
        # Carregar configurações salvas
        saved = settings_manager.prediction
        
        # Variáveis de controle - com valores salvos ou defaults
        self.model_path_var = tk.StringVar(value="Nenhum modelo selecionado")
        self.video_path_var = tk.StringVar(value="Nenhum vídeo selecionado")
        self.threshold_var = tk.DoubleVar(value=saved.threshold)
        self.fps_limit_var = tk.IntVar(value=saved.fps_limit)
        self.fast_mode_var = tk.BooleanVar(value=saved.fast_mode)
        self.show_preview_var = tk.BooleanVar(value=saved.show_preview)
        self.natural_speed_var = tk.BooleanVar(value=saved.natural_speed)
        
        # ROI - Region of Interest
        self.roi_enabled_var = tk.BooleanVar(value=getattr(saved, 'roi_enabled', True))
        self.roi_x_var = tk.DoubleVar(value=getattr(saved, 'roi_x_percent', 25.0))
        self.roi_y_var = tk.DoubleVar(value=getattr(saved, 'roi_y_percent', 25.0))
        
        # Status
        self.status_var = tk.StringVar(value="Pronto")
        self.fps_var = tk.StringVar(value="FPS: --")
        self.progress_var = tk.DoubleVar(value=0)
        self.frame_var = tk.StringVar(value="Frame: 0/0")
        
        # Paths - restaurar se existirem
        self.model_path: Optional[Path] = None
        self.video_path: Optional[Path] = None
        
        if saved.model_path and Path(saved.model_path).exists():
            self.model_path = Path(saved.model_path)
            self.model_path_var.set(str(self.model_path.name))
        
        if saved.video_path and Path(saved.video_path).exists():
            self.video_path = Path(saved.video_path)
            self.video_path_var.set(str(self.video_path.name))
        
        # Inference state
        self.inference_running = False
        self.stop_inference = threading.Event()
        
        # Preview canvas
        self.preview_canvas = None
        self.preview_image = None
        
        self._create_ui()
        
        # Adicionar trace para salvar automaticamente quando configurações mudam
        self.threshold_var.trace_add("write", lambda *args: self._save_settings_auto())
        self.fps_limit_var.trace_add("write", lambda *args: self._save_settings_auto())
        self.natural_speed_var.trace_add("write", lambda *args: self._save_settings_auto())
        self.show_preview_var.trace_add("write", lambda *args: self._save_settings_auto())
        self.fast_mode_var.trace_add("write", lambda *args: self._save_settings_auto())
        self.roi_enabled_var.trace_add("write", lambda *args: self._save_settings_auto())
        self.roi_x_var.trace_add("write", lambda *args: self._save_settings_auto())
        self.roi_y_var.trace_add("write", lambda *args: self._save_settings_auto())
        
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
        
        # Frame de preview
        self._create_preview_frame()
        
        # Frame de estatísticas
        self._create_stats_frame()
        
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
                text="📁 Selecionar",
                bootstyle="primary-outline",
                command=self._browse_model
            )
        else:
            model_btn = ttk.Button(model_frame, text="📁 Selecionar", command=self._browse_model)
        model_btn.grid(row=0, column=1, padx=5)
        
        # Vídeo
        ttk.Label(frame, text="Vídeo:").grid(row=1, column=0, sticky="w", pady=5)
        video_frame = ttk.Frame(frame)
        video_frame.grid(row=1, column=1, sticky="ew", pady=5)
        video_frame.columnconfigure(0, weight=1)
        
        ttk.Label(video_frame, textvariable=self.video_path_var, wraplength=400).grid(
            row=0, column=0, sticky="w"
        )
        
        if USE_TTKBOOTSTRAP:
            video_btn = ttkb.Button(
                video_frame,
                text="🎬 Selecionar",
                bootstyle="primary-outline",
                command=self._browse_video
            )
        else:
            video_btn = ttk.Button(video_frame, text="🎬 Selecionar", command=self._browse_video)
        video_btn.grid(row=0, column=1, padx=5)
        
        # Threshold
        ttk.Label(frame, text="Threshold:").grid(row=2, column=0, sticky="w", pady=5)
        thresh_frame = ttk.Frame(frame)
        thresh_frame.grid(row=2, column=1, sticky="w", pady=5)
        
        ttk.Scale(
            thresh_frame,
            variable=self.threshold_var,
            from_=0.01,
            to=1.0,
            orient=tk.HORIZONTAL,
            length=200
        ).pack(side=tk.LEFT)
        
        ttk.Label(thresh_frame, textvariable=tk.StringVar()).pack(side=tk.LEFT, padx=10)
        
        # Atualizar label do threshold
        def update_thresh_label(*args):
            thresh_frame.winfo_children()[1].config(text=f"{self.threshold_var.get():.0%}")
        self.threshold_var.trace_add("write", update_thresh_label)
        update_thresh_label()
        
        # FPS Limit
        ttk.Label(frame, text="FPS Preview:").grid(row=3, column=0, sticky="w", pady=5)
        fps_frame = ttk.Frame(frame)
        fps_frame.grid(row=3, column=1, sticky="w", pady=5)
        
        ttk.Spinbox(
            fps_frame,
            textvariable=self.fps_limit_var,
            from_=1, to=60,
            width=5
        ).pack(side=tk.LEFT)
        
        # Checkboxes
        check_frame = ttk.Frame(frame)
        check_frame.grid(row=4, column=0, columnspan=2, sticky="w", pady=10)
        
        ttk.Checkbutton(
            check_frame,
            text="🎬 Velocidade Natural",
            variable=self.natural_speed_var,
            command=self._on_natural_speed_change
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Checkbutton(
            check_frame,
            text="👁️ Mostrar Preview",
            variable=self.show_preview_var
        ).pack(side=tk.LEFT, padx=5)
        
        # Segunda linha de checkboxes
        check_frame2 = ttk.Frame(frame)
        check_frame2.grid(row=5, column=0, columnspan=2, sticky="w", pady=5)
        
        ttk.Checkbutton(
            check_frame2,
            text="🚀 Modo Rápido (sem preview, menor resolução)",
            variable=self.fast_mode_var,
            command=self._on_fast_mode_change
        ).pack(side=tk.LEFT, padx=5)
        
        # ROI Frame
        roi_frame = ttk.LabelFrame(frame, text="📐 ROI - Região de Interesse", padding=10)
        roi_frame.grid(row=6, column=0, columnspan=2, sticky="ew", pady=10)
        
        # Checkbox para habilitar/desabilitar ROI
        ttk.Checkbutton(
            roi_frame,
            text="✅ Filtrar detecções pela ROI",
            variable=self.roi_enabled_var
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=5)
        
        # Controles X e Y da ROI
        roi_controls = ttk.Frame(roi_frame)
        roi_controls.grid(row=1, column=0, columnspan=4, sticky="ew", pady=5)
        
        # ROI X
        ttk.Label(roi_controls, text="Margem X:").pack(side=tk.LEFT, padx=5)
        ttk.Scale(
            roi_controls,
            variable=self.roi_x_var,
            from_=0,
            to=45,
            orient=tk.HORIZONTAL,
            length=120
        ).pack(side=tk.LEFT)
        self.roi_x_label = ttk.Label(roi_controls, text=f"{self.roi_x_var.get():.0f}%")
        self.roi_x_label.pack(side=tk.LEFT, padx=5)
        
        # ROI Y
        ttk.Label(roi_controls, text="Margem Y:").pack(side=tk.LEFT, padx=15)
        ttk.Scale(
            roi_controls,
            variable=self.roi_y_var,
            from_=0,
            to=45,
            orient=tk.HORIZONTAL,
            length=120
        ).pack(side=tk.LEFT)
        self.roi_y_label = ttk.Label(roi_controls, text=f"{self.roi_y_var.get():.0f}%")
        self.roi_y_label.pack(side=tk.LEFT, padx=5)
        
        # Atualizar labels quando valores mudam
        def update_roi_labels(*args):
            self.roi_x_label.config(text=f"{self.roi_x_var.get():.0f}%")
            self.roi_y_label.config(text=f"{self.roi_y_var.get():.0f}%")
        self.roi_x_var.trace_add("write", update_roi_labels)
        self.roi_y_var.trace_add("write", update_roi_labels)
        
        # Descrição da ROI
        roi_desc = ttk.Label(
            roi_frame,
            text="Define a área válida para detecções. Margem 25% = ROI de 25% a 75% do frame.\n"
                 "Apenas detecções cujo centróide esteja dentro da ROI serão exibidas.",
            font=("Segoe UI", 8),
            foreground="#2e7d32"
        )
        roi_desc.grid(row=2, column=0, columnspan=4, sticky="w", pady=5)
        
    def _create_control_frame(self):
        """Frame de controle."""
        frame = ttk.LabelFrame(self.parent, text="🎮 Controle", padding=15)
        frame.grid(row=1, column=0, sticky="ew", padx=10, pady=8)
        
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X)
        
        if USE_TTKBOOTSTRAP:
            self.start_btn = ttkb.Button(
                btn_frame,
                text="▶️ Iniciar Inferência",
                bootstyle="success",
                command=self._start_inference
            )
            self.stop_btn = ttkb.Button(
                btn_frame,
                text="⏹️ Parar",
                bootstyle="danger",
                command=self._stop_inference,
                state=tk.DISABLED
            )
            self.export_btn = ttkb.Button(
                btn_frame,
                text="💾 Exportar Vídeo",
                bootstyle="info",
                command=self._export_video,
                state=tk.DISABLED
            )
        else:
            self.start_btn = ttk.Button(btn_frame, text="▶️ Iniciar Inferência", command=self._start_inference)
            self.stop_btn = ttk.Button(btn_frame, text="⏹️ Parar", command=self._stop_inference, state=tk.DISABLED)
            self.export_btn = ttk.Button(btn_frame, text="💾 Exportar Vídeo", command=self._export_video, state=tk.DISABLED)
        
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.export_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(
            frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.pack(fill=tk.X, pady=10)
        
        # Status
        status_frame = ttk.Frame(frame)
        status_frame.pack(fill=tk.X)
        
        ttk.Label(status_frame, textvariable=self.status_var, font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        ttk.Label(status_frame, textvariable=self.frame_var).pack(side=tk.RIGHT, padx=10)
        ttk.Label(status_frame, textvariable=self.fps_var).pack(side=tk.RIGHT, padx=10)
        
    def _create_preview_frame(self):
        """Frame de preview do vídeo."""
        frame = ttk.LabelFrame(self.parent, text="👁️ Preview", padding=15)
        frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=8)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        
        # Canvas para preview com cores verde pastel
        preview_bg = "#f1f8e9"  # Verde muito claro
        self.preview_canvas = tk.Canvas(
            frame,
            bg=preview_bg,
            highlightthickness=2,
            highlightbackground="#a5d6a7",
            relief="flat"
        )
        self.preview_canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Placeholder text com cor verde suave
        placeholder_color = "#66bb6a"  # Verde médio
        self.preview_canvas.create_text(
            300, 200,
            text="Preview do vídeo aparecerá aqui",
            fill=placeholder_color,
            font=("Segoe UI", 14),
            tags="placeholder"
        )
        
    def _create_stats_frame(self):
        """Frame de estatísticas."""
        frame = ttk.LabelFrame(self.parent, text="📊 Estatísticas", padding=15)
        frame.grid(row=3, column=0, sticky="ew", padx=10, pady=8)
        
        # Stats labels
        self.stats_labels = {}
        stats = [
            ("total_frames", "Total de Frames:"),
            ("processed_frames", "Frames Processados:"),
            ("avg_fps", "FPS Médio:"),
            ("detections", "Total Detecções:"),
            ("inference_time", "Tempo de Inferência:")
        ]
        
        for i, (key, label) in enumerate(stats):
            ttk.Label(frame, text=label).grid(row=i // 3, column=(i % 3) * 2, sticky="w", padx=5, pady=2)
            var = tk.StringVar(value="--")
            self.stats_labels[key] = var
            ttk.Label(frame, textvariable=var).grid(row=i // 3, column=(i % 3) * 2 + 1, sticky="w", padx=5, pady=2)
    
    def _save_settings_auto(self):
        """Salva configurações automaticamente."""
        try:
            settings_manager.update_section(
                "prediction",
                model_path=str(self.model_path) if self.model_path else "",
                video_path=str(self.video_path) if self.video_path else "",
                threshold=self.threshold_var.get(),
                fps_limit=self.fps_limit_var.get(),
                natural_speed=self.natural_speed_var.get(),
                show_preview=self.show_preview_var.get(),
                fast_mode=self.fast_mode_var.get(),
                roi_enabled=self.roi_enabled_var.get(),
                roi_x_percent=self.roi_x_var.get(),
                roi_y_percent=self.roi_y_var.get()
            )
        except Exception:
            pass  # Silenciar erros durante trace
    
    def _browse_model(self):
        """Seleciona modelo treinado."""
        # Abrir diretório de experimentos
        folder = filedialog.askdirectory(
            title="Selecionar pasta do experimento (com best_model.pt)",
            initialdir=str(PATHS.experiments)
        )
        
        if folder:
            folder_path = Path(folder)
            model_file = folder_path / "best_model.pt"
            
            if model_file.exists():
                self.model_path = model_file
                self.model_path_var.set(str(model_file))
                logger.info(f"Modelo selecionado: {model_file}")
                self._save_settings_auto()
            else:
                # Tentar encontrar qualquer .pt
                pt_files = list(folder_path.glob("*.pt"))
                if pt_files:
                    self.model_path = pt_files[0]
                    self.model_path_var.set(str(pt_files[0]))
                    logger.info(f"Modelo selecionado: {pt_files[0]}")
                else:
                    messagebox.showerror("Erro", "Nenhum arquivo .pt encontrado na pasta!")
    
    def _browse_video(self):
        """Seleciona vídeo para inferência."""
        filetypes = [
            ("Vídeos", "*.mp4 *.avi *.mov *.mkv *.wmv"),
            ("Todos", "*.*")
        ]
        
        file = filedialog.askopenfilename(
            title="Selecionar vídeo",
            filetypes=filetypes
        )
        
        if file:
            self.video_path = Path(file)
            self.video_path_var.set(str(self.video_path))
            logger.info(f"Vídeo selecionado: {self.video_path}")
            self._save_settings_auto()
            
            # Atualizar estatísticas básicas
            self._update_video_stats()
    
    def _update_video_stats(self):
        """Atualiza estatísticas do vídeo."""
        if not self.video_path:
            return
        
        try:
            import cv2
            cap = cv2.VideoCapture(str(self.video_path))
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            self.stats_labels["total_frames"].set(str(total_frames))
            self.stats_labels["avg_fps"].set(f"{fps:.1f}")
            
            cap.release()
        except Exception as e:
            logger.warning(f"Erro ao ler vídeo: {e}")
    
    def _on_natural_speed_change(self):
        """Callback quando velocidade natural é alterada."""
        # Se velocidade natural está ativa, desabilita o controle de FPS manual
        pass  # O FPS será obtido do vídeo automaticamente
    
    def _on_fast_mode_change(self):
        """Callback quando modo rápido é alterado."""
        if self.fast_mode_var.get():
            self.show_preview_var.set(False)
            self.natural_speed_var.set(False)
    
    def _start_inference(self):
        """Inicia inferência."""
        if not self.model_path or not self.model_path.exists():
            messagebox.showerror("Erro", "Selecione um modelo válido!")
            return
        
        if not self.video_path or not self.video_path.exists():
            messagebox.showerror("Erro", "Selecione um vídeo válido!")
            return
        
        # Atualizar UI
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.export_btn.config(state=tk.DISABLED)
        self.status_var.set("🟡 Carregando modelo...")
        
        self.stop_inference.clear()
        self.inference_running = True
        
        # Iniciar thread
        thread = threading.Thread(target=self._inference_loop, daemon=True)
        thread.start()
    
    def _stop_inference(self):
        """Para inferência."""
        self.stop_inference.set()
        self.status_var.set("🟠 Parando...")
        
        # Fechar janela de preview se existir
        if hasattr(self, 'preview_window') and self.preview_window:
            try:
                self.preview_window.close()
            except:
                pass
            self.preview_window = None
    
    def _on_threshold_change_realtime(self, new_threshold: float):
        """Callback quando threshold muda em tempo real."""
        self.threshold_var.set(new_threshold)
        # O threshold será lido dinamicamente no loop de inferência
    
    def _on_preview_window_close(self):
        """Callback quando janela de preview é fechada."""
        self.stop_inference.set()
        self.preview_window = None
    
    def _inference_loop(self):
        """Loop de inferência (thread separada) com preview em janela separada."""
        preview_window = None
        
        try:
            from src.models.detr_inference import DETRInference
            from src.ui.video_preview_window import VideoPreviewWindow
            import cv2
            import time
            from collections import deque
            
            # Configurações
            image_size = 640 if self.fast_mode_var.get() else 800
            show_preview = self.show_preview_var.get() and not self.fast_mode_var.get()
            
            # Criar janela de preview se necessário
            if show_preview:
                def create_preview():
                    self.preview_window = VideoPreviewWindow(
                        parent=self.parent.winfo_toplevel(),
                        title=f"Preview - {self.video_path.name}",
                        initial_threshold=self.threshold_var.get(),
                        on_threshold_change=self._on_threshold_change_realtime,
                        on_close=self._on_preview_window_close
                    )
                    self.preview_window.set_status("Carregando modelo...")
                
                self.parent.after(0, create_preview)
                time.sleep(0.5)  # Aguardar criação da janela
            
            # Criar inferência
            inference = DETRInference(
                model_path=self.model_path,
                threshold=self.threshold_var.get(),
                image_size=image_size
            )
            
            self._update_status("🟢 Processando vídeo...")
            if show_preview and hasattr(self, 'preview_window') and self.preview_window:
                self.preview_window.set_status("Processando...")
            
            # Abrir vídeo
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                raise ValueError(f"Não foi possível abrir o vídeo: {self.video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Determinar FPS para preview
            if self.natural_speed_var.get():
                effective_fps = video_fps
            else:
                effective_fps = self.fps_limit_var.get()
            
            frame_interval = 1.0 / effective_fps if effective_fps > 0 else 0
            
            # Estatísticas
            processed_frames = 0
            total_detections = 0
            fps_history = deque(maxlen=30)
            start_time = time.time()
            last_frame_time = time.time()
            
            results = {
                "video_path": str(self.video_path),
                "total_frames": total_frames,
                "video_fps": video_fps,
                "detections_per_frame": []
            }
            
            # Obter dimensões do vídeo para ROI
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            while cap.isOpened():
                if self.stop_inference.is_set():
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_start = time.time()
                
                # Obter threshold atual (pode ter mudado via slider)
                current_threshold = self.threshold_var.get()
                inference.threshold = current_threshold
                
                # Obter configuração da ROI atual
                roi_enabled = self.roi_enabled_var.get()
                roi_x = self.roi_x_var.get()
                roi_y = self.roi_y_var.get()
                
                # Inferência
                detections = inference.predict(frame)
                
                # Filtrar por ROI se habilitado
                if roi_enabled:
                    detections = inference.filter_by_roi(
                        detections,
                        frame_width,
                        frame_height,
                        roi_x,
                        roi_y
                    )
                    # Filtrar bounding boxes grandes (máximo 25% da ROI)
                    detections = inference.filter_large_boxes(
                        detections,
                        frame_width,
                        frame_height,
                        roi_x,
                        roi_y,
                        max_size_percent=25.0
                    )
                
                frame_detections = len(detections)
                total_detections += frame_detections
                
                # Configuração da ROI para desenho
                roi_config = {
                    "enabled": roi_enabled,
                    "x_percent": roi_x,
                    "y_percent": roi_y
                }
                
                # Desenhar detecções (e ROI se habilitada)
                frame_drawn = inference.draw_detections(frame, detections, roi_config)
                
                # Calcular FPS
                frame_time = time.time() - frame_start
                current_fps = 1.0 / frame_time if frame_time > 0 else 0
                fps_history.append(current_fps)
                avg_fps = sum(fps_history) / len(fps_history)
                
                processed_frames += 1
                
                # Guardar resultados
                results["detections_per_frame"].append({
                    "frame": processed_frames,
                    "detections": detections,
                    "threshold": current_threshold,
                    "fps": current_fps
                })
                
                # Atualizar UI principal
                self._update_inference_progress(processed_frames, total_frames, avg_fps)
                
                # Atualizar janela de preview
                if show_preview and hasattr(self, 'preview_window') and self.preview_window and self.preview_window.is_open:
                    # Sincronizar com FPS do vídeo
                    current_time = time.time()
                    elapsed = current_time - last_frame_time
                    
                    if frame_interval > 0 and elapsed < frame_interval:
                        time.sleep(frame_interval - elapsed)
                    
                    self.preview_window.update_frame(frame_drawn, frame_detections)
                    self.preview_window.update_stats(avg_fps, processed_frames, total_frames)
                    last_frame_time = time.time()
                
                # Também atualizar preview interno (se habilitado)
                if not show_preview:
                    self._update_preview(frame_drawn)
            
            cap.release()
            
            total_time = time.time() - start_time
            
            results.update({
                "processed_frames": processed_frames,
                "total_detections": total_detections,
                "total_time": total_time,
                "avg_fps": processed_frames / total_time if total_time > 0 else 0
            })
            
            # Atualizar estatísticas finais
            self._update_final_stats(results)
            
            if self.stop_inference.is_set():
                self._update_status("⏹️ Inferência cancelada")
                if show_preview and hasattr(self, 'preview_window') and self.preview_window:
                    self.preview_window.set_status("Cancelado")
            else:
                self._update_status("✅ Inferência concluída!")
                if show_preview and hasattr(self, 'preview_window') and self.preview_window:
                    self.preview_window.set_status("Concluído!")
                self.parent.after(0, lambda: self.export_btn.config(state=tk.NORMAL))
            
            # Guardar resultados para export
            self.last_results = results
            
            logger.info(f"Vídeo processado: {processed_frames} frames em {total_time:.1f}s")
            logger.info(f"  FPS médio: {results['avg_fps']:.1f}")
            logger.info(f"  Total detecções: {total_detections}")
            
        except Exception as e:
            logger.error(f"Erro na inferência: {e}")
            self._update_status(f"❌ Erro: {e}")
            if hasattr(self, 'preview_window') and self.preview_window:
                self.preview_window.set_status(f"Erro: {e}")
        finally:
            self.inference_running = False
            self._inference_finished()
    
    def _inference_finished(self):
        """Chamado quando inferência termina."""
        def update():
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.progress_var.set(0)
        self.parent.after(0, update)
    
    def _update_inference_progress(self, current: int, total: int, fps: float):
        """Atualiza progresso da inferência."""
        def update():
            progress = (current / total) * 100 if total > 0 else 0
            self.progress_var.set(progress)
            self.frame_var.set(f"Frame: {current}/{total}")
            self.fps_var.set(f"FPS: {fps:.1f}")
        self.parent.after(0, update)
    
    def _update_preview(self, frame):
        """Atualiza preview do frame."""
        if not self.show_preview_var.get():
            return
        
        try:
            from PIL import Image, ImageTk
            import cv2
            
            # Converter BGR para RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Redimensionar para caber no canvas
            canvas_width = self.preview_canvas.winfo_width()
            canvas_height = self.preview_canvas.winfo_height()
            
            if canvas_width > 10 and canvas_height > 10:
                h, w = frame_rgb.shape[:2]
                scale = min(canvas_width / w, canvas_height / h)
                new_w, new_h = int(w * scale), int(h * scale)
                
                frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
                
                # Converter para PhotoImage
                image = Image.fromarray(frame_resized)
                photo = ImageTk.PhotoImage(image)
                
                def update():
                    self.preview_canvas.delete("all")
                    self.preview_canvas.create_image(
                        canvas_width // 2,
                        canvas_height // 2,
                        image=photo,
                        anchor=tk.CENTER
                    )
                    self.preview_image = photo  # Manter referência
                
                self.parent.after(0, update)
                
        except Exception as e:
            pass  # Ignorar erros de preview
    
    def _update_final_stats(self, results: dict):
        """Atualiza estatísticas finais."""
        def update():
            self.stats_labels["processed_frames"].set(str(results.get("processed_frames", 0)))
            self.stats_labels["avg_fps"].set(f"{results.get('avg_fps', 0):.1f}")
            self.stats_labels["detections"].set(str(results.get("total_detections", 0)))
            self.stats_labels["inference_time"].set(f"{results.get('total_time', 0):.1f}s")
        self.parent.after(0, update)
    
    def _update_status(self, status: str):
        """Atualiza status."""
        def update():
            self.status_var.set(status)
        self.parent.after(0, update)
    
    def _export_video(self):
        """Exporta vídeo com anotações."""
        if not hasattr(self, 'last_results'):
            messagebox.showerror("Erro", "Execute uma inferência primeiro!")
            return
        
        # Selecionar destino
        output_path = filedialog.asksaveasfilename(
            title="Salvar vídeo anotado",
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4")],
            initialdir=str(PATHS.predictions)
        )
        
        if output_path:
            self.status_var.set("💾 Exportando vídeo...")
            
            def export_thread():
                try:
                    from src.models.detr_inference import DETRInference
                    
                    inference = DETRInference(
                        model_path=self.model_path,
                        threshold=self.threshold_var.get()
                    )
                    
                    inference.export_video(
                        video_path=self.video_path,
                        output_path=Path(output_path),
                        progress_callback=lambda c, t, f: self._update_inference_progress(c, t, f)
                    )
                    
                    self._update_status(f"✅ Vídeo exportado: {output_path}")
                    logger.info(f"Vídeo exportado: {output_path}")
                    
                except Exception as e:
                    logger.error(f"Erro ao exportar: {e}")
                    self._update_status(f"❌ Erro: {e}")
            
            thread = threading.Thread(target=export_thread, daemon=True)
            thread.start()
