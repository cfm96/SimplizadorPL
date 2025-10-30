import tkinter as tk
from tkinter import ttk, messagebox
import os, sys
import numpy as np

from simplizador_gui import SimplizadorGUI
from simplex_window import SimplexWindow
from two_phase_gui import TwoPhaseSimplexWindow

def _set_windows_app_id(app_id: str = "UniLP.SimplizadorPL"):
    """Fija un AppUserModelID para que Windows use el icono del ejecutable en la barra de tareas.
    Seguro en plataformas no Windows (no hace nada).
    """
    try:
        if sys.platform.startswith('win'):
            import ctypes  # type: ignore
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass

if __name__ == "__main__":
    # Sugerir AppID antes de crear ventanas (mejora uso de icono en barra de tareas en Windows)
    _set_windows_app_id()
    app = SimplizadorGUI()
    app.mainloop()
