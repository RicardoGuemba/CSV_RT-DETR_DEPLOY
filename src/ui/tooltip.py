"""
Sistema de Tooltips para widgets Tkinter.
Sistema DETR - RTX 5080 Optimized
"""
import tkinter as tk
from typing import Optional


class ToolTip:
    """Cria tooltips para widgets Tkinter."""
    
    def __init__(self, widget: tk.Widget, text: str = '', delay: int = 500):
        """
        Args:
            widget: Widget que receberá o tooltip
            text: Texto do tooltip
            delay: Delay em ms antes de mostrar o tooltip
        """
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tipwindow: Optional[tk.Toplevel] = None
        self.id: Optional[str] = None
        self.x = 0
        self.y = 0
        
        # Bind eventos
        self.widget.bind('<Enter>', self.enter)
        self.widget.bind('<Leave>', self.leave)
        self.widget.bind('<ButtonPress>', self.leave)
    
    def enter(self, event=None):
        """Evento quando mouse entra no widget."""
        self.schedule()
    
    def leave(self, event=None):
        """Evento quando mouse sai do widget."""
        self.unschedule()
        self.hidetip()
    
    def schedule(self):
        """Agenda exibição do tooltip."""
        self.unschedule()
        self.id = self.widget.after(self.delay, self.showtip)
    
    def unschedule(self):
        """Cancela agendamento do tooltip."""
        id_ = self.id
        self.id = None
        if id_:
            self.widget.after_cancel(id_)
    
    def showtip(self, event=None):
        """Mostra o tooltip."""
        x, y, _, _ = self.widget.bbox("insert") if hasattr(self.widget, 'bbox') else (0, 0, 0, 0)
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        
        # Criar janela de tooltip
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        
        # Frame com fundo
        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            font=("Segoe UI", 9),
            wraplength=300,
            padx=8,
            pady=4
        )
        label.pack(ipadx=1)
    
    def hidetip(self):
        """Esconde o tooltip."""
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


def create_tooltip(widget: tk.Widget, text: str, delay: int = 500) -> ToolTip:
    """
    Cria um tooltip para um widget.
    
    Args:
        widget: Widget que receberá o tooltip
        text: Texto explicativo
        delay: Delay em ms (padrão: 500ms)
    
    Returns:
        Instância do ToolTip criado
    """
    return ToolTip(widget, text, delay)
