import os
import tkinter as tk

# Variable global de depuración (imprime trazas solo si está en True)
DEBUG = False

# Resuelve rutas a recursos tanto en modo script como empaquetado (PyInstaller)
def _resource_path(relative_path: str) -> str:
    try:
        base_path = sys._MEIPASS  # type: ignore[attr-defined]
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)



def _set_window_icon(win: tk.Tk | tk.Toplevel):
    """Intenta establecer el icono de la ventana desde icon.ico si existe.
    En Windows, iconbitmap con .ico es lo más fiable.
    """
    try:
        icon_path = _resource_path('icon.ico')
        if os.path.exists(icon_path):
            win.iconbitmap(icon_path)
    except Exception:
        # Silencioso si no está disponible
        pass

def _dbg(msg: str):
    if DEBUG:
        try:
            print(msg)
        except Exception:
            pass
