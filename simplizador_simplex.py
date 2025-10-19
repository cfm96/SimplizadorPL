import tkinter as tk
from tkinter import ttk, messagebox
import os, sys
import numpy as np

# Variable global de depuración (imprime trazas solo si está en True)
DEBUG = False

def _dbg(msg: str):
    if DEBUG:
        try:
            print(msg)
        except Exception:
            pass

# Formatea números evitando notación científica y eliminando ceros innecesarios
def _fmt(x, decimals: int = 6):
    try:
        v = float(x)
    except Exception:
        return str(x)
    if abs(v) < 1e-10:
        v = 0.0
    s = f"{v:.{decimals}f}"
    s = s.rstrip('0').rstrip('.')
    if s == "-0":
        s = "0"
    return s

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

# Construye el modelo extendido para el método de 2 fases (agrega holguras y variables artificiales)
def _estandarizar_problema_dos_fases(tipo, n, c, restricciones):
    # todas las variables serán denotadas por x1...xK
    nombres = [f"x{i+1}" for i in range(n)]
    c_arr = np.array(c, dtype=float)
    if tipo == 'min':
        c_arr = -c_arr

    A_rows = []
    b_rows = []
    holg = 0
    artific = 0
    indices_art = []
    tipos = []

    for coefs, op, b in restricciones:
        fila = np.array(coefs, dtype=float)
        if fila.size != n:
            raise ValueError('Cada restricción debe tener n coeficientes')
        if b < 0:
            fila = -fila
            b = -b
            if op == '<=':
                op = '>='
            elif op == '>=':
                op = '<='

        current_cols = n + holg + artific

        if op == '<=':
            for i in range(len(A_rows)):
                A_rows[i] = np.append(A_rows[i], 0.0)
            fila_ext = np.concatenate([fila, np.zeros(holg + artific), [1.0]])
            A_rows.append(fila_ext)
            b_rows.append(b)
            holg += 1
            nombres.append(f'x{len(nombres)+1}')
            tipos.append('<=')

        elif op == '>=':
            for i in range(len(A_rows)):
                A_rows[i] = np.append(A_rows[i], 0.0)
            fila_ext = np.concatenate([fila, np.zeros(holg + artific), [-1.0]])
            A_rows.append(fila_ext)
            b_rows.append(b)
            holg += 1
            nombres.append(f'x{len(nombres)+1}')
            # En este caso es necesario agregar una variable artificial
            for i in range(len(A_rows)-1):
                A_rows[i] = np.append(A_rows[i], 0.0)
            A_rows[-1] = np.append(A_rows[-1], 1.0)
            artific += 1
            nombres.append(f'x{len(nombres)+1}')
            indices_art.append(len(nombres)-1)
            tipos.append('>=')

        elif op == '=':
            for i in range(len(A_rows)):
                if len(A_rows[i]) < current_cols:
                    A_rows[i] = np.append(A_rows[i], np.zeros(current_cols - len(A_rows[i])))
            fila_ext = np.concatenate([fila, np.zeros(holg + artific)])
            A_rows.append(fila_ext)
            b_rows.append(b)
            # En este caso es necesario agregar una variable artificial
            for i in range(len(A_rows)-1):
                A_rows[i] = np.append(A_rows[i], 0.0)
            A_rows[-1] = np.append(A_rows[-1], 1.0)
            artific += 1
            nombres.append(f'x{len(nombres)+1}')
            indices_art.append(len(nombres)-1)
            tipos.append('=')

        else:
            raise ValueError('Operador inválido (usa <=, >=, =)')

    if A_rows:
        A = np.array(A_rows, dtype=float)
    else:
        A = np.zeros((0, n + holg + artific), dtype=float)
    b = np.array(b_rows, dtype=float)
    c_ext = np.concatenate([np.array(c_arr, dtype=float), np.zeros(holg + artific)])

    # Reordenar columnas: colocar variables artificiales al final conservando el orden relativo
    if indices_art:
        total_cols = A.shape[1]
        art_set = set(indices_art)
        non_art = [j for j in range(total_cols) if j not in art_set]
        new_order = non_art + indices_art
        A = A[:, new_order]
        c_ext = c_ext[new_order]
        nombres = [nombres[j] for j in new_order]
        # Reindexar posiciones de variables artificiales al final
        k = len(indices_art)
        indices_art = list(range(total_cols - k, total_cols))
        # Renombrar secuencialmente como x1...xn en el nuevo orden
        nombres = [f"x{i+1}" for i in range(total_cols)]

    return A, b, c_ext, nombres, indices_art, tipos


# Crea una matriz correspondiente a la tabla simplex a partir de A, b y el vector de costos c
def _construir_tabla_simplex(A, b, c):
    m, n = A.shape
    tabla = np.zeros((m+1, n+1))
    tabla[1:, :-1] = A
    tabla[1:, -1] = b
    tabla[0, :-1] = -c
    tabla[0, -1] = 0.0
    return tabla


# Detecta índices de variables básicas por fila en A
def _encontrar_variables_basicas(A, indices_artificiales):
    m, n = A.shape
    basicas = []
    usados = set()
    for i in range(m):
        encontrado = False
        for j in indices_artificiales:
            if j < n and j not in usados and abs(A[i,j] - 1.0) < 1e-10 and np.all(np.delete(np.abs(A[:,j]), i) < 1e-10):
                basicas.append(j)
                usados.add(j)
                encontrado = True
                break
        if encontrado:
            continue
        for j in range(n):
            if j not in usados and abs(A[i,j] - 1.0) < 1e-10 and np.all(np.delete(np.abs(A[:,j]), i) < 1e-10):
                basicas.append(j)
                usados.add(j)
                encontrado = True
                break
        if not encontrado:
            basicas.append(None)
    return basicas


# Realiza un pivote en la tabla simplex sobre (fila, col) y lleva la columna a ser un vector de la matriz canónica
def _pivotear(tabla, fila, col):
    tabla[fila,:] /= tabla[fila,col]
    m, n = tabla.shape
    for i in range(m):
        if i != fila:
            tabla[i,:] -= tabla[i,col] * tabla[fila,:]


# Saca variables artificiales de la base cuando sea posible durante Fase I
def _pivot_out_artificiales_en_fase1(tabla, indices_art, original_n):
    m, n = tabla.shape
    for j in indices_art:
        if j >= n-1:
            continue
        col_vec = tabla[1:, j]
        ones = np.where(np.abs(col_vec - 1.0) < 1e-10)[0]
        if len(ones) != 1 or not np.all(np.delete(np.abs(col_vec), ones[0]) < 1e-10):
            continue
        basic_row = 1 + int(ones[0])
        candidates = [k for k in range(n-1) if k not in indices_art and tabla[basic_row, k] > 1e-10]
        if not candidates:
            continue
        orig_candidates = [k for k in candidates if k < original_n]
        k = orig_candidates[0] if orig_candidates else candidates[0]
        _pivotear(tabla, basic_row, k)
    return tabla


# Ejecuta Fase I (con función objetivo W) hasta encontrar factibilidad o hasta no poder mejorar
def _fase_uno(tabla, A, indices_artificiales, nombres):
    m, n = tabla.shape
    basicas = _encontrar_variables_basicas(A, indices_artificiales)
    for i, var in enumerate(basicas):
        if var in indices_artificiales:
            tabla[0, :] -= tabla[i+1, :]
    original_n = sum(1 for nm in nombres if nm.startswith('x'))
    tabla = _pivot_out_artificiales_en_fase1(tabla, indices_artificiales, original_n)
    while True:
        coefs = tabla[0, :-1]
        neg_cols = [j for j, v in enumerate(coefs) if v < -1e-10 and j < len(nombres)]
        if not neg_cols:
            break
        chosen = None
        for j in sorted(neg_cols, key=lambda jj: coefs[jj]):
            if np.any(tabla[1:, j] > 1e-10):
                chosen = j
                break
        if chosen is None:
            break
        col = chosen
        col_vals = tabla[1:, col]
        rhs = tabla[1:, -1]
        ratios = [rhs[i]/col_vals[i] if col_vals[i] > 1e-10 else np.inf for i in range(len(col_vals))]
        fila = 1 + int(np.argmin(ratios))
        _pivotear(tabla, fila, col)
    return tabla


# Elimina columnas relacionadas a variables artificiales de la tabla y sus nombres asociados
def _eliminar_artificiales_tabla(tabla, nombres, indices_art):
    cols_keep = [j for j in range(tabla.shape[1]-1) if j not in indices_art]
    cols_keep.append(tabla.shape[1]-1)
    tabla2 = tabla[:, cols_keep]
    nombres2 = [nombres[j] for j in range(len(nombres)) if j not in indices_art]
    return tabla2, nombres2


# Ejecuta Fase II (con función objetivo Z) a partir de una tabla sin variables artificiales
def _fase_dos(tabla, c_original, nombres):
    n = tabla.shape[1]-1
    tabla[0,:-1] = -np.array([c_original[j] if j < len(c_original) else 0.0 for j in range(n)])
    it = 0
    while it < 500:
        coefs = tabla[0,:-1]
        neg_cols = [j for j, v in enumerate(coefs) if v < -1e-10 and j < len(nombres)]
        if not neg_cols:
            break
        chosen = None
        for j in sorted(neg_cols, key=lambda jj: coefs[jj]):
            if np.any(tabla[1:, j] > 1e-10):
                chosen = j
                break
        if chosen is None:
            break
        col = chosen
        col_vals = tabla[1:, col]
        rhs = tabla[1:, -1]
        ratios = [rhs[i]/col_vals[i] if col_vals[i] > 1e-10 else np.inf for i in range(len(col_vals))]
        fila = 1 + int(np.argmin(ratios))
        _pivotear(tabla, fila, col)
        it += 1
    return tabla


# Resuelve un modelo con el método de 2 fases
def resolver_dos_fases(tipo, n, c, restricciones):
    A, b, c_ext, nombres, indices_art, _ = _estandarizar_problema_dos_fases(tipo, n, c, restricciones)
    if A.size == 0:
        raise ValueError('No hay restricciones')
    m, ncol = A.shape
    c_fase1 = np.zeros(ncol)
    for idx in indices_art:
        if idx < ncol:
            c_fase1[idx] = -1.0  # maximizar -W
    tabla = _construir_tabla_simplex(A, b, c_fase1)
    tabla = _fase_uno(tabla, A, indices_art, nombres)
    if abs(tabla[0,-1]) > 1e-8:
        return None, 'infactible'
    tabla, nombres = _eliminar_artificiales_tabla(tabla, nombres, indices_art)
    tabla = _fase_dos(tabla, c, nombres)
    m2, n2 = tabla.shape
    sol = {nombres[j]:0.0 for j in range(n2-1)}
    for i in range(1,m2):
        for j in range(n2-1):
            col = tabla[1:,j]
            if abs(col[i-1]-1.0)<1e-8 and np.all(np.delete(np.abs(col), i-1) < 1e-8):
                sol[nombres[j]] = float(tabla[i,-1])
                break
    valor = float(tabla[0,-1])
    return (sol, valor), 'ok'

# Genera una representación estandarizada sin variables artificiales
def estandarizar_problema(tipo, n, c, restricciones):
    nombres_vars = [f"x{i+1}" for i in range(n)]
    c_estandar = np.array(c, dtype=float)
    if tipo == "min":
        c_estandar = -c_estandar

    A_filas, b_filas = [], []
    holg_exce = 0

    for coefs, op, val in restricciones:
        fila = np.array(coefs, dtype=float)

        if val < 0:
            fila = -fila
            val = -val
            if op == "<=":
                op = ">="
            elif op == ">=":
                op = "<="

        if op == "<=":
            coef_h_e = np.zeros(holg_exce + 1)
            coef_h_e[-1] = 1
            for i in range(len(A_filas)):
                A_filas[i] = np.append(A_filas[i], 0)
            A_filas.append(np.concatenate([fila, coef_h_e]))
            b_filas.append(val)
            holg_exce += 1
            nombres_vars.append(f"x{holg_exce + n}")

        elif op == ">=":
            coef_h_e = np.zeros(holg_exce + 1)
            coef_h_e[-1] = -1
            for i in range(len(A_filas)):
                A_filas[i] = np.append(A_filas[i], 0)
            A_filas.append(np.concatenate([fila, coef_h_e]))
            b_filas.append(val)
            holg_exce += 1
            nombres_vars.append(f"x{holg_exce + n}")

        elif op == "=":
            if holg_exce > 0:
                fila = np.concatenate([fila, np.zeros(holg_exce)])
            A_filas.append(fila)
            b_filas.append(val)
        else:
            raise ValueError("Operador no reconocido: usa <=, >= o =")

    A_estandar = np.array(A_filas, dtype=float)
    b_estandar = np.array(b_filas, dtype=float)
    c_estandar = np.concatenate([c_estandar, np.zeros(holg_exce)])
    return A_estandar, b_estandar, c_estandar, nombres_vars

# Ventana principal de la aplicación
class SimplizadorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simplizador de P.L.")
        self.geometry("800x680")
        _set_window_icon(self)

        ttk.Label(
            self,
            text="¡Bienvenido a Simplizador!\n\nIngrese los datos de su Problema de Programación Lineal",
            font=("Helvetica", 14),
            foreground="blue",
            justify="center"
        ).pack(pady=15)

        self.tipo_var = tk.StringVar(value="max")
        self.n_var = tk.IntVar()
        self.m_var = tk.IntVar()

        frm_top = ttk.Frame(self)
        frm_top.pack(pady=10)
        ttk.Label(frm_top, text="Clasificación:").grid(row=0, column=0, padx=5)
        ttk.Combobox(frm_top, textvariable=self.tipo_var,
                     values=["max", "min"], width=6).grid(row=0, column=1, padx=5)
        ttk.Label(frm_top, text="N° variables:").grid(row=0, column=2, padx=5)
        ttk.Entry(frm_top, textvariable=self.n_var, width=5).grid(row=0, column=3, padx=5)
        ttk.Label(frm_top, text="N° restricciones:").grid(row=0, column=4, padx=5)
        ttk.Entry(frm_top, textvariable=self.m_var, width=5).grid(row=0, column=5, padx=5)

        # Etiquetas informativas sobre los límites
        frm_limits = ttk.Frame(self)
        frm_limits.pack(pady=(0, 10))
        ttk.Label(frm_limits, text="Límites máximos: 50 variables y 50 restricciones", 
                 font=("Arial", 9), foreground="gray").pack()

        ttk.Button(self, text="Continuar", command=self.crear_campos).pack(pady=10)

        # Crear el área con scrollbars
        self.canvas_frame = ttk.Frame(self)
        self.canvas_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Canvas para el contenido con scroll
        self.canvas = tk.Canvas(self.canvas_frame, bg="white")
        
        # Scrollbars vertical y horizontal
        self.v_scrollbar = ttk.Scrollbar(self.canvas_frame, orient="vertical", command=self.canvas.yview)
        self.h_scrollbar = ttk.Scrollbar(self.canvas_frame, orient="horizontal", command=self.canvas.xview)
        
        # Configuración del canvas
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)
        
        # Frame interno que contendrá todos los widgets
        self.frm_datos = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.frm_datos, anchor="nw")
        
        # Posicionar scrollbars y canvas
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")

        # Configuración del grid para que se expanda
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        
        # Bind para actualizar el scroll region
        self.frm_datos.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        
        # Bind para scroll con rueda del mouse
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.bind_all("<Shift-MouseWheel>", self.on_shift_mousewheel)

        self.coef_entries = []   # función objetivo
        self.restr_widgets = []  # restricciones
        self.creation_job_id = None  # para cancelar procesos anteriores

    def crear_campos(self):
        # Cancelar cualquier proceso de creación anterior
        if self.creation_job_id:
            self.after_cancel(self.creation_job_id)
            self.creation_job_id = None
        
        # Limpiar
        for w in self.frm_datos.winfo_children():
            w.destroy()
        self.coef_entries.clear()
        self.restr_widgets.clear()
        
        # Resetear scroll region
        self.canvas.configure(scrollregion=(0, 0, 0, 0))

        n = self.n_var.get()
        m = self.m_var.get()
        
        # Validaciones de entrada
        if n <= 0 or m <= 0:
            messagebox.showerror("Error", "Debes ingresar números positivos.")
            return
        
        if n > 50:
            messagebox.showerror("Error", f"Número máximo de variables: 50\nIngresaste: {n}")
            return
            
        if m > 50:
            messagebox.showerror("Error", f"Número máximo de restricciones: 50\nIngresaste: {m}")
            return

        ttk.Label(self.frm_datos, text="Función objetivo:").pack()
        frm_c = ttk.Frame(self.frm_datos)
        frm_c.pack(pady=5)
        for i in range(n):
            e = ttk.Entry(frm_c, width=6)
            e.grid(row=0, column=3 * i, padx=2)
            e.insert(0, "0")
            ttk.Label(frm_c, text=f"x{i+1}").grid(row=0, column=3 * i + 1, padx=2)
            if i < n - 1:
                ttk.Label(frm_c, text="+").grid(row=0, column=3 * i + 2, padx=2)
            self.coef_entries.append(e)

        ttk.Label(self.frm_datos, text="Restricciones:").pack(pady=(10, 0))
        
        # Crear restricciones en lotes para mejorar rendimiento
        self.crear_restricciones_progresivamente(n, m)

    def crear_restricciones_progresivamente(self, n, m):
        # Tamaño de lote optimizado para límites de 50x50
        if m <= 10:
            BATCH_SIZE = m  # Crear todas de una vez si son pocas
        elif m <= 25:
            BATCH_SIZE = 10
        else:
            BATCH_SIZE = 5  # Lotes pequeños para 25-50 restricciones
        
        # Mostrar progreso para restricciones múltiples
        if m > 15:
            progress_label = ttk.Label(self.frm_datos, 
                                     text=f"Cargando restricciones... 0/{m}",
                                     foreground="darkgreen", justify="center")
            progress_label.pack(pady=5)
        else:
            progress_label = None
        
        def crear_lote(inicio):
            # Verificar si el proceso fue cancelado
            if not hasattr(self, 'frm_datos') or not self.frm_datos.winfo_exists():
                return
                
            fin = min(inicio + BATCH_SIZE, m)
            
            for r in range(inicio, fin):
                fila = ttk.Frame(self.frm_datos)
                fila.pack(pady=3)
                coef_e = []
                
                # Agregar etiqueta de restricción R1), R2), etc. con ancho fijo
                label_text = f"R{r+1})"
                ttk.Label(fila, text=label_text, font=("Arial", 10, "bold"), 
                         foreground="blue", width=6, anchor="w").grid(row=0, column=0, padx=(5, 5), sticky="w")
                
                for i in range(n):
                    # Ajustar posición de columnas para dar espacio a la etiqueta Rx)
                    col_offset = 1  # Offset por la etiqueta Rx)
                    e = ttk.Entry(fila, width=6)
                    e.grid(row=0, column=col_offset + 3 * i, padx=2)
                    e.insert(0, "0")
                    ttk.Label(fila, text=f"x{i+1}").grid(row=0, column=col_offset + 3 * i + 1, padx=2)
                    if i < n - 1:
                        ttk.Label(fila, text="+").grid(row=0, column=col_offset + 3 * i + 2, padx=2)
                    coef_e.append(e)
                
                op = ttk.Combobox(fila, values=["<=", ">=", "="], width=4)
                op.set("<=")
                op.grid(row=0, column=col_offset + 3 * n, padx=5)
                val = ttk.Entry(fila, width=6)
                val.grid(row=0, column=col_offset + 3 * n + 1, padx=5)
                val.insert(0, "0")
                self.restr_widgets.append({"coef": coef_e, "op": op, "val": val})
            
            # Actualizar progreso
            if progress_label:
                progress_label.config(text=f"Cargando restricciones... {fin}/{m}\n✓ Última cargada: R{fin})")
            
            # Actualizar scroll region
            try:
                self.update_scroll_region()
            except Exception as e:
                if progress_label:
                    progress_label.config(text=f"Error de memoria en restricción R{fin})\nIntente con menos restricciones",
                                        foreground="red")
                return
            
            # Continuar con el siguiente lote si hay más restricciones
            if fin < m:
                # Delay optimizado para límites conocidos (50x50)
                delay = 30 if m <= 25 else 50
                try:
                    self.creation_job_id = self.after(delay, lambda: crear_lote(fin))
                except Exception as e:
                    if progress_label:
                        progress_label.config(text=f"Error inesperado en restricción R{fin})",
                                            foreground="red")
                    self.creation_job_id = None
            else:
                # Terminado - remover label de progreso y crear botón
                if progress_label:
                    progress_label.config(text=f"✓ Completado: {m} restricciones cargadas exitosamente", 
                                        foreground="darkgreen")
                    self.creation_job_id = self.after(1500, progress_label.destroy)  # Mostrar mensaje por 1.5 segundos
                self.creation_job_id = self.after(1500, self.crear_boton_estandarizar)
        
        # Iniciar la creación del primer lote
        crear_lote(0)
    
    def crear_boton_estandarizar(self):
        # Verificar si ya existe un botón para evitar duplicados
        for widget in self.frm_datos.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Button) and child.cget("text") == "Estandarizar":
                        return  # Ya existe, no crear otro
        
        # Frame para contener el botón y alinearlo a la izquierda
        button_frame = ttk.Frame(self.frm_datos)
        button_frame.pack(fill="x", pady=15)
        
        ttk.Button(button_frame, text="Estandarizar",
                   command=self.estandarizar).pack(side="left", padx=10)
        
        # Limpiar el job ID ya que terminamos
        self.creation_job_id = None
        
        # Actualización final del scroll region
        self.update_scroll_region()
        self.after(100, self.update_scroll_region)

    def update_scroll_region(self):
        # Forzar actualización de todos los widgets
        self.frm_datos.update_idletasks()
        self.update_idletasks()
        
        # Obtener el bbox real del contenido
        bbox = self.canvas.bbox("all")
        if bbox:
            self.canvas.configure(scrollregion=bbox)
        
        # Asegurar que el canvas tenga el ancho correcto
        frame_width = self.frm_datos.winfo_reqwidth()
        canvas_width = self.canvas.winfo_width()
        
        if frame_width > canvas_width:
            self.canvas.itemconfig(self.canvas_window, width=frame_width)

    def on_frame_configure(self, event):
        self.update_scroll_region()
    
    def on_canvas_configure(self, event):
        canvas_width = event.width
        frame_width = self.frm_datos.winfo_reqwidth()
        
        # Solo ajustar el ancho si el frame es más pequeño que el canvas
        # Esto permite scroll horizontal cuando el contenido es más ancho
        if frame_width < canvas_width:
            self.canvas.itemconfig(self.canvas_window, width=canvas_width)
        else:
            self.canvas.itemconfig(self.canvas_window, width=frame_width)
    
    def on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def on_shift_mousewheel(self, event):
        self.canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

    def estandarizar(self):
        try:
            tipo = self.tipo_var.get()
            n = self.n_var.get()
            c = [float(e.get()) for e in self.coef_entries]
            restricciones = []
            for r in self.restr_widgets:
                coefs = [float(e.get()) for e in r["coef"]]
                if len(coefs) != n:
                    raise ValueError("Todas las restricciones deben tener coeficientes para cada variable.")
                op = r["op"].get()
                val = float(r["val"].get())
                restricciones.append((coefs, op, val))

            # Estandarización para mostrar (sin variables artificiales aún)
            A, b, c_est, nombres = estandarizar_problema(tipo, n, c, restricciones)

            out = ["=== Problema lineal estándar ==="]
            out.append("Variables: " + ", ".join(nombres))

            # Mostrar la función objetivo en su forma original
            if tipo == "min":
                # Deshacer la negación de c_est para mostrar los coeficientes originales
                coefs_originales = [-coef for coef in c_est]
                terminos = []
                for i, (coef, var) in enumerate(zip(coefs_originales, nombres)):
                    if i == 0:
                        sign = '-' if coef < 0 else ''
                        terminos.append(f"{sign}{_fmt(abs(coef))}{var}")
                    else:
                        sign = ' - ' if coef < 0 else ' + '
                        terminos.append(f"{sign}{_fmt(abs(coef))}{var}")
                out.append("Min Z = " + "".join(terminos))
            else:
                terminos = []
                for i, (coef, var) in enumerate(zip(c_est, nombres)):
                    if i == 0:
                        sign = '-' if coef < 0 else ''
                        terminos.append(f"{sign}{_fmt(abs(coef))}{var}")
                    else:
                        sign = ' - ' if coef < 0 else ' + '
                        terminos.append(f"{sign}{_fmt(abs(coef))}{var}")
                out.append("Max Z = " + "".join(terminos))

            out.append("Sujeto a:")
            for i, (fila, val) in enumerate(zip(A, b)):
                term_parts = []
                for j, (coef, var) in enumerate(zip(fila, nombres)):
                    if j == 0:
                        sign = '-' if coef < 0 else ''
                        term_parts.append(f"{sign}{_fmt(abs(coef))}{var}")
                    else:
                        sign = ' - ' if coef < 0 else ' + '
                        term_parts.append(f"{sign}{_fmt(abs(coef))}{var}")
                out.append(f"R{i+1})  {''.join(term_parts)} = {_fmt(val)}")
            out.append("Todas las variables ≥ 0")

            # Anexar el problema auxiliar si aplica, sin alterar el estándar mostrado
            A2, b2, c_ext2, nombres2, indices_art2, tipos2 = _estandarizar_problema_dos_fases(tipo, n, c, restricciones)
            if indices_art2:
                out.append("")
                out.append("=== Problema auxiliar (Fase I) ===")
                # Variables
                out.append("Variables: " + ", ".join(nombres2))
                # Función objetivo W con el mismo formato que el estándar, incluyendo 0x en variables no artificiales
                terminos_w = []
                signo_art = -1 if tipo == 'max' else 1
                for j, var in enumerate(nombres2):
                    coef = signo_art if j in indices_art2 else 0
                    if j == 0:
                        sign = '-' if coef < 0 else ''
                        if coef == 1:
                            terminos_w.append(f"{sign}{var}")
                        elif coef == -1:
                            terminos_w.append(f"{sign}{var}")
                        else:
                            terminos_w.append(f"{sign}{_fmt(abs(coef))}{var}")
                    else:
                        sign = ' - ' if coef < 0 else ' + '
                        if coef == 1:
                            terminos_w.append(f"{sign}{var}")
                        elif coef == -1:
                            terminos_w.append(f"{sign}{var}")
                        else:
                            terminos_w.append(f"{sign}{_fmt(abs(coef))}{var}")
                titulo_w = "Max W = " if tipo == 'max' else "Min W = "
                out.append(titulo_w + ("".join(terminos_w) if terminos_w else "0"))
                out.append("Sujeto a:")
                for i, (fila, val) in enumerate(zip(A2, b2)):
                    term_parts = []
                    for j, (coef, var) in enumerate(zip(fila, nombres2)):
                        # Incluir también 0x para claridad
                        sign = ' + ' if (j > 0) else ''
                        if coef < -1e-12:
                            sign = ' - ' if j > 0 else '-'
                            term_parts.append(f"{sign}{_fmt(abs(coef))}{var}")
                        elif coef > 1e-12:
                            term_parts.append(f"{sign}{_fmt(coef)}{var}")
                        else:
                            term_parts.append(f"{sign}0{var}")
                    out.append(f"R{i+1})  {''.join(term_parts)} = {_fmt(val)}")
                # Agregar positividad de variables en el auxiliar
                out.append("Todas las variables ≥ 0")

            # Mostrar ventana y resolver unificado (2 fases si aplica)
            self.mostrar_ventana_estandarizacion(
                A, b, c_est, nombres,
                "\n".join(out),
                c_original=c,
                tipo_original=tipo,
                n_original=n,
                restricciones_originales=restricciones,
            )
            # Guardar pre-cómputo de 2 fases para usarlo al resolver y garantizar mismas columnas
            self._precomputed_two_phase = (A2, b2, c_ext2, nombres2, indices_art2, tipos2)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def mostrar_ventana_estandarizacion(self, A, b, c_est, nombres, texto_estandar,
                                        c_original, tipo_original, n_original, restricciones_originales):
        ventana = tk.Toplevel(self)
        ventana.title("Problema Lineal Estandarizado")
        ventana.geometry("600x500")
        ventana.resizable(True, True)
        
        # Frame principal con scroll
        main_frame = ttk.Frame(ventana)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Texto con el problema estandarizado
        text_widget = tk.Text(main_frame, wrap=tk.WORD, font=("Courier", 10))
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        text_widget.insert("1.0", texto_estandar)
        text_widget.config(state="disabled")
        
        # Frame para botones
        button_frame = ttk.Frame(ventana)
        button_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        ttk.Button(button_frame, text="Cerrar", 
                  command=ventana.destroy).pack(side="right", padx=(5, 0))
        ttk.Button(button_frame, text="Resolver (Simplex)", 
                  command=lambda: self.iniciar_simplex_unificado(
                      tipo_original, n_original, c_original, restricciones_originales, ventana
                  )).pack(side="right")

    def resolver_dos_fases_gui(self, tipo, n, c, restricciones):
        try:
            resultado, status = resolver_dos_fases(tipo, n, c, restricciones)
            if status != 'ok' or resultado is None:
                messagebox.showwarning("Resultado", "El problema es infactible en Fase I.")
                return
            sol, valor = resultado
            # Armar texto ordenado para variables originales primero
            nombres_vars = [f"x{i+1}" for i in range(n)]
            texto = ["=== SOLUCIÓN (Simplex 2 Fases) ==="]
            texto.append(f"Tipo: {tipo.upper()}")
            texto.append(f"Valor óptimo: {float(valor):.6g}")
            texto.append("")
            texto.append("Variables:")
            # Mostrar primero variables originales
            for nm in nombres_vars:
                v = float(sol.get(nm, 0.0))
                texto.append(f"{nm} = {v:.6g}")
            # Luego variables adicionales si existen
            extras = [k for k in sol.keys() if k not in nombres_vars]
            if extras:
                texto.append("")
                texto.append("Variables adicionales:")
                for nm in sorted(extras):
                    v = float(sol.get(nm, 0.0))
                    texto.append(f"{nm} = {v:.6g}")
            messagebox.showinfo("Simplex 2 Fases", "\n".join(texto))
        except Exception as e:
            messagebox.showerror("Error", f"Fallo al resolver 2 fases: {e}")

    def iniciar_simplex_unificado(self, tipo, n, c_original, restricciones, ventana_padre):
        ventana_padre.destroy()
        # garantiza que el solver reciba las columnas artificiales correctas
        try:
            A2, b2, c_ext2, nombres2, indices_art2, tipos2 = _estandarizar_problema_dos_fases(
                tipo, n, c_original, restricciones
            )
            pre = (A2, b2, c_ext2, nombres2, indices_art2, tipos2)
        except Exception:
            pre = None
        self.ventana_simplex = TwoPhaseSimplexWindow(
            self, tipo, n, c_original, ('__PRECOMPUTED__', pre) if pre is not None else restricciones
        )


# Ventana de demostración del simplex canónico
class SimplexWindow(tk.Toplevel):
    def __init__(self, parent, A, b, c, nombres, tipo_original, c_original):
        super().__init__(parent)
        self.title(f"Método Simplex - Problema de {tipo_original.upper()}")
        self.geometry("900x700")
        self.resizable(True, True)
        _set_window_icon(self)
        
        # Datos del problema
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.c = np.array(c, dtype=float)  # Coeficientes para el simplex (negados si es MIN)
        self.c_original = np.array(c_original, dtype=float)  # Coeficientes originales del usuario
        self.nombres = nombres
        self.tipo_original = tipo_original  # 'max' o 'min'
        self.iteracion = 0
        
        # Crear tabla inicial
        self.crear_tabla_inicial()
        self.setup_interface()
    
    def crear_tabla_inicial(self):
        m, n = self.A.shape
        
        # Tabla simplex: [c^T | 0]
        #                [A  | b]
        self.tabla = np.zeros((m + 1, n + 1))
        
        # Primera fila: coeficientes de la función objetivo (negativo para maximización)
        self.tabla[0, :-1] = -self.c
        self.tabla[0, -1] = 0  # Valor inicial de Z
        
        # Resto de filas: matriz A y vector b
        self.tabla[1:, :-1] = self.A
        self.tabla[1:, -1] = self.b
        
        # Identificar variables básicas iniciales (variables slack/surplus/artificiales)
        # Son las variables que se agregaron durante la estandarización
        m, n = self.A.shape
        
        # Asumir que las variables básicas iniciales son las últimas m variables
        # (variables slack/surplus que se agregaron al estandarizar)
        if n >= m:
            self.variables_basicas = list(range(n - m, n))
        else:
            # Caso especial: si hay menos variables que restricciones
            self.variables_basicas = list(range(n))
    
    def setup_interface(self):
        # Frame principal
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Título de iteración
        self.titulo_label = ttk.Label(main_frame, 
                                     text=f"Iteración {self.iteracion}: Tabla Canónica Inicial",
                                     font=("Arial", 14, "bold"))
        self.titulo_label.pack(pady=(0, 10))
        
        # Frame para la tabla con scroll
        table_frame = ttk.Frame(main_frame)
        table_frame.pack(fill="both", expand=True)
        
        # Canvas para scroll
        self.canvas = tk.Canvas(table_frame)
        v_scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(table_frame, orient="horizontal", command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Frame interno para la tabla
        self.tabla_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.tabla_frame, anchor="nw")
        
        # Posicionar scrollbars y canvas
        self.canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        # Status y botones
        self.status_frame = ttk.Frame(main_frame)
        self.status_frame.pack(fill="x", pady=(10, 0))
        
        self.status_label = ttk.Label(self.status_frame, text="", font=("Arial", 11))
        self.status_label.pack()
        
        # Botones
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(10, 0))
        
        self.btn_siguiente = ttk.Button(button_frame, text="Siguiente Iteración", 
                                       command=self.siguiente_iteracion)
        self.btn_siguiente.pack(side="left")
        
        ttk.Button(button_frame, text="Cerrar", 
                  command=self.destroy).pack(side="right")
        
        # Mostrar tabla inicial
        self.mostrar_tabla()
        self.verificar_optimalidad()
    
    def mostrar_tabla(self):
        # Limpiar tabla anterior
        for widget in self.tabla_frame.winfo_children():
            widget.destroy()
        
        m, n = self.tabla.shape
        
        # Fila de encabezados principales
        # cⱼ | xᵦ | x₁ | x₂ | ... | bᵢ | bᵢ/yᵢₖ (razones)
        base_headers = ["cⱼ", "xᵦ"] + [self.nombres[i] for i in range(n-1)] + ["bᵢ"]
        
        # Determinar el texto para la columna de razones (siempre genérico)
        razon_header = "bᵢ/yᵢₖ"
        
        headers = base_headers + [razon_header]
        
        for col, header in enumerate(headers):
            ttk.Label(self.tabla_frame, text=header, font=("Arial", 11, "bold"),
                     relief="ridge", borderwidth=1, background="#E6E6FA").grid(
                     row=0, column=col, sticky="nsew", padx=1, pady=1)
        
        # Fila de coeficientes de función objetivo (segunda fila de encabezados)
        ttk.Label(self.tabla_frame, text="", font=("Arial", 10),
                 relief="ridge", borderwidth=1).grid(row=1, column=0, sticky="nsew", padx=1, pady=1)
        ttk.Label(self.tabla_frame, text="", font=("Arial", 10),
                 relief="ridge", borderwidth=1).grid(row=1, column=1, sticky="nsew", padx=1, pady=1)
        
        # Coeficientes originales de la función objetivo (mostrar coeficientes del problema original)
        for j in range(n-1):
            # Mostrar coeficiente original del problema del usuario
            if j < len(self.c_original):
                coef_original = self.c_original[j]
            else:
                coef_original = 0  # Variables slack/surplus tienen coeficiente 0
            
            # Evitar mostrar -0
            if abs(coef_original) < 1e-10:
                coef_original = 0
            ttk.Label(self.tabla_frame, text=_fmt(coef_original), font=("Arial", 10),
                     relief="ridge", borderwidth=1, background="#F0F8FF").grid(
                     row=1, column=j+2, sticky="nsew", padx=1, pady=1)
        
        # Columnas vacías para bᵢ y razones en fila de coeficientes
        ttk.Label(self.tabla_frame, text="", font=("Arial", 10),
                 relief="ridge", borderwidth=1).grid(row=1, column=n+1, sticky="nsew", padx=1, pady=1)
        ttk.Label(self.tabla_frame, text="", font=("Arial", 10),
                 relief="ridge", borderwidth=1).grid(row=1, column=n+2, sticky="nsew", padx=1, pady=1)
        
        # Filas de restricciones (variables básicas)
        current_row = 2
        for i in range(1, m):  # Empezar desde 1 para saltar la fila objetivo
            # cⱼ para variable básica
            indice_var_basica = i - 1
            if (indice_var_basica < len(self.variables_basicas) and 
                self.variables_basicas[indice_var_basica] < len(self.c)):
                coef_var_basica = self.c[self.variables_basicas[indice_var_basica]]
            else:
                coef_var_basica = 0
            
            ttk.Label(self.tabla_frame, text=f"{coef_var_basica:.3g}", font=("Arial", 10),
                     relief="ridge", borderwidth=1).grid(row=current_row, column=0, sticky="nsew", padx=1, pady=1)
            
            # xᵦ (variable básica)
            if (indice_var_basica < len(self.variables_basicas) and 
                self.variables_basicas[indice_var_basica] < len(self.nombres)):
                var_basica = self.nombres[self.variables_basicas[indice_var_basica]]
            else:
                var_basica = f"R{i}"
            
            ttk.Label(self.tabla_frame, text=var_basica, font=("Arial", 10, "bold"),
                     relief="ridge", borderwidth=1, background="#FFE4E1").grid(
                     row=current_row, column=1, sticky="nsew", padx=1, pady=1)
            
            # Coeficientes de las variables
            for j in range(n-1):
                valor = self.tabla[i, j]
                ttk.Label(self.tabla_frame, text=_fmt(valor), font=("Arial", 10),
                         relief="ridge", borderwidth=1).grid(row=current_row, column=j+2, sticky="nsew", padx=1, pady=1)
            
            # bᵢ (lado derecho)
            bi_valor = self.tabla[i, -1]
            ttk.Label(self.tabla_frame, text=_fmt(bi_valor), font=("Arial", 10),
                     relief="ridge", borderwidth=1, background="#F0FFF0").grid(
                     row=current_row, column=n+1, sticky="nsew", padx=1, pady=1)
            
            # bᵢ/yᵢₖ (razones) - calcular y mostrar si no es óptima
            razon_texto = ""
            if not self.es_optima():
                # Encontrar columna entrante para calcular razones
                coefs_objetivo = self.tabla[0, :-1]
                if np.any(coefs_objetivo < -1e-10):
                    col_entrante = np.argmin(coefs_objetivo)
                    elemento_columna = self.tabla[i, col_entrante]
                    
                    # Solo calcular razón si yᵢ > 0 (regla del simplex)
                    if elemento_columna > 1e-10:
                        razon = bi_valor / elemento_columna
                        razon_texto = _fmt(razon)
                    else:
                        razon_texto = "-"  # No se calcula cuando yᵢ ≤ 0
            
            ttk.Label(self.tabla_frame, text=razon_texto, font=("Arial", 10),
                     relief="ridge", borderwidth=1, background="#FFF8DC").grid(
                     row=current_row, column=n+2, sticky="nsew", padx=1, pady=1)
            
            current_row += 1
        
        # Fila Zⱼ
        ttk.Label(self.tabla_frame, text="", font=("Arial", 10),
                 relief="ridge", borderwidth=1).grid(row=current_row, column=0, sticky="nsew", padx=1, pady=1)
        ttk.Label(self.tabla_frame, text="Zⱼ", font=("Arial", 10, "bold"),
                 relief="ridge", borderwidth=1, background="#FFFACD").grid(
                 row=current_row, column=1, sticky="nsew", padx=1, pady=1)
        
        # Calcular Zⱼ para cada variable usando coeficientes originales
        for j in range(n-1):
            zj_valor = 0
            for i in range(1, m):
                indice_var_basica = i - 1
                if (indice_var_basica < len(self.variables_basicas)):
                    var_basica_idx = self.variables_basicas[indice_var_basica]
                    # Usar coeficiente original si es variable original, 0 si es slack/surplus
                    if var_basica_idx < len(self.c_original):
                        coef_basica = self.c_original[var_basica_idx]
                    else:
                        coef_basica = 0
                    zj_valor += coef_basica * self.tabla[i, j]
            
            # Evitar mostrar -0
            ttk.Label(self.tabla_frame, text=_fmt(zj_valor), font=("Arial", 10),
                     relief="ridge", borderwidth=1, background="#FFFACD").grid(
                     row=current_row, column=j+2, sticky="nsew", padx=1, pady=1)
        
        # Z (valor función objetivo) - ajustar según tipo de problema
        z_valor_tabla = self.tabla[0, -1]
        if self.tipo_original == "min":
            # Para minimización, mostrar el valor negativo (convertir de vuelta del problema transformado)
            z_valor = -z_valor_tabla
        else:
            # Para maximización, mostrar el valor directo
            z_valor = z_valor_tabla
        
        # Evitar mostrar -0
        ttk.Label(self.tabla_frame, text=_fmt(z_valor), font=("Arial", 10),
                 relief="ridge", borderwidth=1, background="#FFFACD").grid(
                 row=current_row, column=n+1, sticky="nsew", padx=1, pady=1)
        ttk.Label(self.tabla_frame, text="", font=("Arial", 10),
                 relief="ridge", borderwidth=1).grid(row=current_row, column=n+2, sticky="nsew", padx=1, pady=1)
        
        current_row += 1
        
        # Fila cⱼ - Zⱼ (etiqueta según convención usada)
        ttk.Label(self.tabla_frame, text="", font=("Arial", 10),
                 relief="ridge", borderwidth=1).grid(row=current_row, column=0, sticky="nsew", padx=1, pady=1)
        
        # Etiqueta según el tipo de problema y convención usada
        if self.tipo_original == "min":
            etiqueta_fila = "cⱼ - Zⱼ"
        else:
            etiqueta_fila = "Zⱼ - cⱼ"
            
        ttk.Label(self.tabla_frame, text=etiqueta_fila, font=("Arial", 10, "bold"),
                 relief="ridge", borderwidth=1, background="#FFE4B5").grid(
                 row=current_row, column=1, sticky="nsew", padx=1, pady=1)
        
        # Calcular y mostrar cⱼ - Zⱼ usando coeficientes originales
        for j in range(n-1):
            # Obtener cⱼ original (siempre usar coeficientes originales en la tabla canónica)
            if j < len(self.c_original):
                cj_original = self.c_original[j]
            else:
                cj_original = 0
                
            # Calcular Zⱼ para esta columna (usar coeficientes originales para la tabla canónica)
            zj_valor = 0
            for i in range(1, m):
                indice_var_basica = i - 1
                if (indice_var_basica < len(self.variables_basicas)):
                    var_basica_idx = self.variables_basicas[indice_var_basica]
                    if var_basica_idx < len(self.c_original):
                        # Siempre usar coeficientes originales para la tabla canónica
                        coef_basica = self.c_original[var_basica_idx]
                    else:
                        coef_basica = 0
                    zj_valor += coef_basica * self.tabla[i, j]
            
            # Calcular cⱼ - Zⱼ con el signo correcto según el tipo de problema
            if self.tipo_original == "min":
                # Para minimización, usar cⱼ - Zⱼ directo
                cj_zj_valor = cj_original - zj_valor
            else:
                # Para maximización, usar Zⱼ - cⱼ (invertir signo)
                cj_zj_valor = zj_valor - cj_original
            
            # Evitar mostrar -0
            ttk.Label(self.tabla_frame, text=_fmt(cj_zj_valor), font=("Arial", 10),
                     relief="ridge", borderwidth=1, background="#FFE4B5").grid(
                     row=current_row, column=j+2, sticky="nsew", padx=1, pady=1)
        
        ttk.Label(self.tabla_frame, text="", font=("Arial", 10),
                 relief="ridge", borderwidth=1).grid(row=current_row, column=n+1, sticky="nsew", padx=1, pady=1)
        ttk.Label(self.tabla_frame, text="", font=("Arial", 10),
                 relief="ridge", borderwidth=1).grid(row=current_row, column=n+2, sticky="nsew", padx=1, pady=1)
        
        # Configurar pesos de columnas y filas
        total_rows = current_row + 1
        total_cols = n + 3
        
        for i in range(total_rows):
            self.tabla_frame.grid_rowconfigure(i, weight=1)
        for j in range(total_cols):
            self.tabla_frame.grid_columnconfigure(j, weight=1)
        
        # Actualizar scroll region
        self.tabla_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def es_optima(self):
        coefs_objetivo = self.tabla[0, :-1]
        return np.all(coefs_objetivo >= -1e-10)
    
    def verificar_optimalidad(self):
        # El criterio de optimalidad sigue siendo el mismo (basado en la tabla simplex interna)
        # pero ahora mostramos información más clara al usuario
        coefs_objetivo = self.tabla[0, :-1]
        
        if np.all(coefs_objetivo >= -1e-10):  # Tolerancia numérica
            self.mostrar_solucion_optima()
            return True
        else:
            min_coef = np.min(coefs_objetivo)
            col_entrada = np.argmin(coefs_objetivo)
            
            # Mostrar mensaje apropiado según el tipo de problema
            if self.tipo_original == "min":
                criterio_valor = -(-min_coef)  # Convertir a formato de minimización
                self.status_label.config(text=f"No es óptima. Variable entrante: {self.nombres[col_entrada]} (cⱼ - Zⱼ = {_fmt(criterio_valor)})")
            else:
                # Para maximización, calcular el valor tal como aparece en la tabla canónica
                # Necesitamos calcular Zⱼ - cⱼ para esta columna específica
                cj_original = self.c_original[col_entrada] if col_entrada < len(self.c_original) else 0
                
                # Calcular Zⱼ para esta columna
                zj_valor = 0
                m = self.tabla.shape[0]
                for i in range(1, m):
                    indice_var_basica = i - 1
                    if (indice_var_basica < len(self.variables_basicas)):
                        var_basica_idx = self.variables_basicas[indice_var_basica]
                        if var_basica_idx < len(self.c_original):
                            coef_basica = self.c_original[var_basica_idx]
                        else:
                            coef_basica = 0
                        zj_valor += coef_basica * self.tabla[i, col_entrada]
                
                # Para maximización: Zⱼ - cⱼ (como aparece en la tabla canónica)
                criterio_valor = zj_valor - cj_original
                self.status_label.config(text=f"No es óptima. Variable entrante: {self.nombres[col_entrada]} (Zⱼ - cⱼ = {_fmt(criterio_valor)})")
            return False
    
    def mostrar_solucion_optima(self):
        self.status_label.config(text="¡SOLUCIÓN ÓPTIMA ENCONTRADA!")
        self.btn_siguiente.config(state="disabled")
        
        # Calcular valores de variables
        solucion = {}
        valor_z_tabla = self.tabla[0, -1]  # Valor de Z en la tabla (siempre para maximización)
        
        # Variables básicas
        for i, var_idx in enumerate(self.variables_basicas):
            if var_idx < len(self.nombres):
                solucion[self.nombres[var_idx]] = self.tabla[i+1, -1]
        
        # Variables no básicas (valor 0)
        for i, nombre in enumerate(self.nombres[:-1]):  # Excluir RHS
            if nombre not in solucion:
                solucion[nombre] = 0
        
        # Ajustar valor óptimo según el tipo de problema original
        if self.tipo_original == "min":
            # Para problemas de minimización, el valor óptimo es -Z
            valor_optimo_original = -valor_z_tabla
            titulo_problema = "MINIMIZACIÓN"
        else:
            # Para problemas de maximización, usar el valor directo
            valor_optimo_original = valor_z_tabla
            titulo_problema = "MAXIMIZACIÓN"
        
        # Mostrar solución
        texto_solucion = f"=== SOLUCIÓN ÓPTIMA ===\n"
        texto_solucion += f"Problema original de {titulo_problema}\n"
        texto_solucion += f"Valor óptimo = {_fmt(valor_optimo_original)}\n\n"
        texto_solucion += "Valores de variables:\n"
        
        for nombre in self.nombres[:-1]:
            if nombre in solucion:
                texto_solucion += f"{nombre} = {_fmt(solucion[nombre])}\n"
        
        messagebox.showinfo("Solución Óptima", texto_solucion)
    
    def siguiente_iteracion(self):
        if self.verificar_optimalidad():
            return
        # Si en esta misma llamada se transicionó a Fase II, mostrar la tabla de transición y detener
        if self.phase == 2 and getattr(self, '_fase2_mostrar_transicion', False):
            # Asegurar mensaje y no pivotear aún
            try:
                self.status_label.config(text="Fase II: Iteración 0 (tabla de transición). Pulse 'Siguiente Iteración' para continuar")
            except Exception:
                pass
            self._mostrar_tabla()
            return
        # Selección de pivote: r_j < 0 (definición ya contempla fase/tipo)
        _dbg(f"DEBUG: Iniciando iteración {self.iteracion + 1}")
        _dbg(f"variables_basicas antes: {self.variables_basicas}")
        
        # Encontrar variable entrante (columna pivote)
        coefs_objetivo = self.tabla[0, :-1]
        col_pivote = np.argmin(coefs_objetivo)
        
        _dbg(f"Variable entrante: columna {col_pivote} ({self.nombres[col_pivote] if col_pivote < len(self.nombres) else 'N/A'})")
        
        # Encontrar variable saliente (fila pivote) - criterio de razón mínima
        columna_pivote = self.tabla[1:, col_pivote]
        rhs = self.tabla[1:, -1]
        
        _dbg(f"Columna pivote: {columna_pivote}")
        _dbg(f"RHS: {rhs}")
        
        # Calcular razones solo para elementos yᵢ > 0 (regla del simplex)
        razones = []
        razones_validas = []
        for i in range(len(columna_pivote)):
            if columna_pivote[i] > 1e-10:
                razon = rhs[i] / columna_pivote[i]
                razones.append(razon)
                # Solo considerar razones positivas para el criterio de salida
                if razon >= 0:
                    razones_validas.append((razon, i))
            else:
                razones.append(float('inf'))  # No válida para criterio de salida
        
        _dbg(f"Razones: {razones}")
        _dbg(f"Razones válidas (≥0): {razones_validas}")
        
        if not razones_validas:  # No hay razones positivas
            messagebox.showerror("Error", "Problema no acotado - no hay solución finita")
            return
        
        # Encontrar la menor razón positiva
        min_razon, min_indice = min(razones_validas)
        fila_pivote = min_indice + 1  # +1 porque excluimos fila objetivo
        elemento_pivote = self.tabla[fila_pivote, col_pivote]
        
        _dbg(f"Fila pivote: {fila_pivote}, elemento pivote: {elemento_pivote}")
        
        # Las razones ya se muestran automáticamente en mostrar_tabla()
        
        # Realizar operaciones de fila
        # 1. Hacer el elemento pivote = 1
        self.tabla[fila_pivote, :] /= elemento_pivote
        
        # 2. Hacer ceros en el resto de la columna
        m, n = self.tabla.shape
        for i in range(m):
            if i != fila_pivote:
                factor = self.tabla[i, col_pivote]
                self.tabla[i, :] -= factor * self.tabla[fila_pivote, :]
        
        # Actualizar variable básica
        indice_basica = fila_pivote - 1  # -1 porque fila_pivote incluye la fila objetivo
        if 0 <= indice_basica < len(self.variables_basicas):
            self.variables_basicas[indice_basica] = col_pivote
        else:
            _dbg(f"ERROR: indice_basica={indice_basica}, len(variables_basicas)={len(self.variables_basicas)}")
            _dbg(f"fila_pivote={fila_pivote}, col_pivote={col_pivote}")
            # Expandir la lista si es necesario
            while len(self.variables_basicas) <= indice_basica:
                self.variables_basicas.append(0)
            self.variables_basicas[indice_basica] = col_pivote
        
        # Actualizar iteración
        self.iteracion += 1
        self.titulo_label.config(text=f"Iteración {self.iteracion}: Tabla Canónica")
        
        # Mostrar tabla actualizada
        self.mostrar_tabla()
        self.verificar_optimalidad()


# Ventana principal del método simplex de 2 fases con tablas paso a paso
class TwoPhaseSimplexWindow(tk.Toplevel):
    def __init__(self, parent, tipo, n, c_original, restricciones):
        super().__init__(parent)
        self.geometry("900x700")
        self.resizable(True, True)
        _set_window_icon(self)

        # Guardar originales
        self.tipo_original = tipo
        self.n_original = n
        self.c_original = np.array(c_original, dtype=float)

        # Estandarizar con núcleo 2 fases (usar pre-cómputo si viene adjunto)
        if isinstance(restricciones, tuple) and restricciones and restricciones[0] == '__PRECOMPUTED__':
            payload = restricciones[1]
            if payload is None:
                # Recalcular de inmediato
                A, b, c_ext, nombres, indices_art, tipos = _estandarizar_problema_dos_fases(
                    tipo, n, c_original, [] if restricciones is None else restricciones
                )
            else:
                # Aceptar payload de 5 o 6 elementos
                if len(payload) == 6:
                    A, b, c_ext, nombres, indices_art, tipos = payload
                else:
                    A, b, c_ext, nombres, indices_art = payload
                    tipos = None
        else:
            A, b, c_ext, nombres, indices_art, tipos = _estandarizar_problema_dos_fases(
                tipo, n, c_original, restricciones
            )
        self.A = A
        self.b = b
        self.nombres = nombres
        # Si por alguna razón indices_art llegó vacío pero 'tipos' indica >= o =, reconstruir artificiales
        if (indices_art is None or len(indices_art) == 0) and A.size > 0:
            try:
                # Heurística: columnas con patrón de base inicial que no son holguras suelen ser artificiales.
                # Pero preferimos confiar en 'tipos' si está disponible: cada restricción >= o = debe tener una artificial.
                indices_art_calc = []
                if tipos is not None:
                    # Buscar columnas unitarias por fila y mapear las que no sean holguras (+1) o excesos (-1)
                    subA = A
                    m, ncols = subA.shape
                    unit_cols = []
                    for j in range(ncols):
                        col = subA[:, j]
                        ones = np.where(np.abs(col - 1.0) < 1e-10)[0]
                        if len(ones) == 1 and np.all(np.delete(np.abs(col), ones[0]) < 1e-10):
                            unit_cols.append((j, int(ones[0])))
                    # Para cada restricción con >= o =, tomar la columna unitaria que cae en esa fila y no sea holgura ni exceso
                    for i, t in enumerate(tipos):
                        if t in ('>=', '='):
                            candidates = [j for (j, r) in unit_cols if r == i]
                            if candidates:
                                # Si las artificiales se reordenan al final, prioriza columnas más a la derecha
                                candidates.sort()
                                guess = candidates[-1]
                                # Evitar marcar holguras/excesos: si existe además una columna con -1 en esa fila (exceso),
                                # la artificial suele ser la otra unitaria.
                                indices_art_calc.append(guess)
                indices_art = sorted(set(indices_art_calc)) if indices_art_calc else indices_art
            except Exception:
                pass
        self.indices_art = indices_art or []
        # Si tipos indica que existe al menos una restricción >= o =, forzar Fase I
        if tipos is not None and any(t in ('>=', '=') for t in tipos):
            self.phase = 1
        else:
            self.phase = 1 if len(self.indices_art) > 0 else 2
        self.title("Método Simplex - Fase I (W)" if self.phase == 1 else "Método Simplex - Fase II (Z)")
        self.iteracion = 0
        # Mostrar siempre Fase I al menos una vez si hay artificiales
        self._fase1_mostrada = False
        # Señal para transicionar a Fase II en el siguiente clic, una vez mostrada la Fase I factible
        self._fase1_listo_transicion = False
        # Mostrar tabla de transición (Iteración 0) al entrar en Fase II
        self._fase2_mostrar_transicion = False

        # Construir tabla inicial
        if self.phase == 1:
            if len(self.indices_art) == 0:
                # Si llegamos aquí, tipos obligó Fase I pero no hay índices; caída segura: todas las columnas extra como 0 y no transicionar aún
                pass
            # Fase I: costos para artificiales dependen del tipo original
            # - MAX: maximizar -W => c_art = -1
            # - MIN: minimizar W  => c_art = +1
            signo_art = -1.0 if tipo == 'max' else 1.0
            c_fase1 = np.zeros(self.A.shape[1])
            for j in self.indices_art:
                if j < len(c_fase1):
                    c_fase1[j] = signo_art
            self.tabla = _construir_tabla_simplex(self.A, self.b, c_fase1)
            basicas = _encontrar_variables_basicas(self.A, self.indices_art)
            # Canonicalizar objetivo: fila0 = -c.
            # - Si c_art = +1 (MIN): fila0 tiene -1 en art => sumar filas básicas artificiales
            # - Si c_art = -1 (MAX): fila0 tiene +1 en art => restar filas básicas artificiales
            for i, var in enumerate(basicas):
                if var in self.indices_art:
                    if signo_art > 0:
                        self.tabla[0, :] += self.tabla[i+1, :]
                    else:
                        self.tabla[0, :] -= self.tabla[i+1, :]
            # No forzar RHS a 0: mantener W inicial = suma de b de filas artificiales básicas
        else:
            # Fase II directa
            self.tabla = _construir_tabla_simplex(self.A, self.b, np.zeros(self.A.shape[1]))
            # Fila 0 = -c_original extendido
            ncols = self.tabla.shape[1] - 1
            coefs = -np.array([self.c_original[j] if j < len(self.c_original) else 0.0 for j in range(ncols)])
            self.tabla[0, :-1] = coefs

        # UI similar a SimplexWindow
        self._setup_interface()
        self._mostrar_tabla()
        self._verificar_optimalidad()

    def _setup_interface(self):
        # Configura widgets y layout (título, canvas con scroll, tabla y barra de estado)
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.titulo_label = ttk.Label(
            main_frame,
            text=f"Fase {self.phase} - Iteración {self.iteracion}",
            font=("Arial", 14, "bold"),
        )
        self.titulo_label.pack(pady=(0, 10))

        table_frame = ttk.Frame(main_frame)
        table_frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(table_frame)
        v_scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(table_frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        self.tabla_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.tabla_frame, anchor="nw")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        self.status_frame = ttk.Frame(main_frame)
        self.status_frame.pack(fill="x", pady=(10, 0))
        self.status_label = ttk.Label(self.status_frame, text="", font=("Arial", 11))
        self.status_label.pack()

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(10, 0))
        self.btn_siguiente = ttk.Button(button_frame, text="Siguiente Iteración", command=self._siguiente_iteracion)
        self.btn_siguiente.pack(side="left")
        ttk.Button(button_frame, text="Cerrar", command=self.destroy).pack(side="right")

    def _compute_basics_zj_reduced(self):
        # Determinar columnas básicas (unitarias) actuales y calcular Zj/Wj y costos reducidos coherentes con la fase
        subA = self.tabla[1:, :-1]
        m, n_minus = subA.shape
        indices_art_set = set(self.indices_art)
    # En Fase I: c_B para artificiales es signo_art (±1 según tipo original), resto 0
        # Básicas por fila
        basicas_idx = []
        for i in range(m):
            var_idx = None
            for j in range(n_minus):
                col = subA[:, j]
                if abs(col[i] - 1.0) < 1e-10 and np.all(np.delete(np.abs(col), i) < 1e-10):
                    var_idx = j
                    break
            if var_idx is None:
                var_idx = 0
            basicas_idx.append(var_idx)
        # Calcular Zj/Wj
        zj_vals = []
        for j in range(n_minus):
            acc = 0.0
            for i in range(m):
                var_idx = basicas_idx[i] if i < len(basicas_idx) else 0
                if self.phase == 1:
                    signo_art = -1.0 if self.tipo_original == 'max' else 1.0
                    # En Fase I, cB = signo_art si la variable básica es artificial; 0 en otro caso
                    cB = signo_art if var_idx in indices_art_set else 0.0
                else:
                    cB = self.c_original[var_idx] if var_idx < len(self.c_original) else 0.0
                acc += cB * subA[i, j]
            zj_vals.append(acc)
        # Costos reducidos r_j según fase/tipo mostrado
        reduced = []
        for j in range(n_minus):
            if self.phase == 1:
                # En Fase I, cj = signo_art para artificiales y 0 para el resto
                signo_art = -1.0 if self.tipo_original == 'max' else 1.0
                cj = signo_art if j in indices_art_set else 0.0
                zj = zj_vals[j]
                # Criterio consistente con Fase II: MAX usa rj=Zj-cj (entrante si rj<0), MIN usa rj=cj-Zj
                rj = (zj - cj) if self.tipo_original == 'max' else (cj - zj)
            else:
                cj = self.c_original[j] if j < len(self.c_original) else 0.0
                zj = zj_vals[j]
                # En Fase II, usar convención del problema original
                rj = (zj - cj) if self.tipo_original == 'max' else (cj - zj)
            reduced.append(rj)
        return basicas_idx, zj_vals, reduced

    def _mostrar_tabla(self):
        # Renderiza la tabla actual (Fase I/Fase II) y, si aplica, la vista previa de Fase II
        # Reutiliza lógica de mostrar_tabla del SimplexWindow adaptada a atributos locales
        for widget in self.tabla_frame.winfo_children():
            widget.destroy()
        m, n = self.tabla.shape
        base_headers = ["cⱼ", "xᵦ"] + [self.nombres[i] for i in range(n-1)] + ["bᵢ"]
        razon_header = "bᵢ/yᵢₖ"
        headers = base_headers + [razon_header]
        for col, header in enumerate(headers):
            ttk.Label(self.tabla_frame, text=header, font=("Arial", 11, "bold"),
                     relief="ridge", borderwidth=1, background="#E6E6FA").grid(
                     row=0, column=col, sticky="nsew", padx=1, pady=1)

        ttk.Label(self.tabla_frame, text="", font=("Arial", 10),
                 relief="ridge", borderwidth=1).grid(row=1, column=0, sticky="nsew", padx=1, pady=1)
        ttk.Label(self.tabla_frame, text="", font=("Arial", 10),
                 relief="ridge", borderwidth=1).grid(row=1, column=1, sticky="nsew", padx=1, pady=1)
        # Fila de costos c_j según la fase:
        # - Fase I (problema W): 0 para no artificiales; ±1 para artificiales (1 si MIN, -1 si MAX)
        # - Fase II: coeficientes originales del usuario para variables originales; 0 en extras
        indices_art_set = set(self.indices_art)
        for j in range(n-1):
            if self.phase == 1:
                signo_art = -1.0 if self.tipo_original == 'max' else 1.0
                coef_mostrar = signo_art if j in indices_art_set else 0.0
            else:
                coef_mostrar = self.c_original[j] if j < len(self.c_original) else 0.0
            if abs(coef_mostrar) < 1e-10:
                coef_mostrar = 0.0
            ttk.Label(self.tabla_frame, text=_fmt(coef_mostrar), font=("Arial", 10),
                     relief="ridge", borderwidth=1, background="#F0F8FF").grid(
                     row=1, column=j+2, sticky="nsew", padx=1, pady=1)
        ttk.Label(self.tabla_frame, text="", font=("Arial", 10),
                 relief="ridge", borderwidth=1).grid(row=1, column=n+1, sticky="nsew", padx=1, pady=1)
        ttk.Label(self.tabla_frame, text="", font=("Arial", 10),
                 relief="ridge", borderwidth=1).grid(row=1, column=n+2, sticky="nsew", padx=1, pady=1)

        current_row = 2
        basicas_idx, zj_vals, reduced = self._compute_basics_zj_reduced()
        # Calcular RHS de Z/W como suma c_B * b según la fase
        subA = self.tabla[1:, :-1]
        rhs_b = self.tabla[1:, -1]
        z_rhs = 0.0
        for i in range(subA.shape[0]):
            var_idx = basicas_idx[i] if i < len(basicas_idx) else 0
            if self.phase == 1:
                signo_art = -1.0 if self.tipo_original == 'max' else 1.0
                cB = signo_art if var_idx in indices_art_set else 0.0
            else:
                cB = self.c_original[var_idx] if var_idx < len(self.c_original) else 0.0
            z_rhs += cB * rhs_b[i]
        for i in range(1, m):
            indice_var_basica = i - 1
            var_idx = basicas_idx[indice_var_basica] if indice_var_basica < len(basicas_idx) else 0
            # Coste básico mostrado según la fase
            if self.phase == 1:
                # En Fase I, mostrar costo básico signo_art si la básica es artificial; 0 en otro caso
                signo_art = -1.0 if self.tipo_original == 'max' else 1.0
                coef_var_basica = signo_art if var_idx in indices_art_set else 0.0
            else:
                coef_var_basica = self.c_original[var_idx] if var_idx < len(self.c_original) else 0.0
            ttk.Label(self.tabla_frame, text=_fmt(coef_var_basica), font=("Arial", 10),
                     relief="ridge", borderwidth=1).grid(row=current_row, column=0, sticky="nsew", padx=1, pady=1)
            var_basica = self.nombres[var_idx] if var_idx < len(self.nombres) else f"R{i}"
            ttk.Label(self.tabla_frame, text=var_basica, font=("Arial", 10, "bold"),
                     relief="ridge", borderwidth=1, background="#FFE4E1").grid(
                     row=current_row, column=1, sticky="nsew", padx=1, pady=1)
            for j in range(n-1):
                valor = self.tabla[i, j]
                ttk.Label(self.tabla_frame, text=_fmt(valor), font=("Arial", 10),
                         relief="ridge", borderwidth=1).grid(row=current_row, column=j+2, sticky="nsew", padx=1, pady=1)
            bi_valor = self.tabla[i, -1]
            ttk.Label(self.tabla_frame, text=_fmt(bi_valor), font=("Arial", 10),
                     relief="ridge", borderwidth=1, background="#F0FFF0").grid(
                     row=current_row, column=n+1, sticky="nsew", padx=1, pady=1)
            razon_texto = ""
            # Elegir columna entrante: siempre r_j < 0 (con r_j definido según el tipo)
            tol = 1e-10
            candidatos = [j for j, v in enumerate(reduced) if v < -tol]
            orden = sorted(candidatos, key=lambda jj: reduced[jj])  # más negativo primero
            col_entrante = None
            for j in orden:
                if np.any(self.tabla[1:, j] > 1e-10):
                    col_entrante = j
                    break
            if col_entrante is not None:
                elemento_columna = self.tabla[i, col_entrante]
                if elemento_columna > 1e-10:
                    razon = bi_valor / elemento_columna
                    razon_texto = _fmt(razon)
                else:
                    razon_texto = "-"
            ttk.Label(self.tabla_frame, text=razon_texto, font=("Arial", 10),
                     relief="ridge", borderwidth=1, background="#FFF8DC").grid(
                     row=current_row, column=n+2, sticky="nsew", padx=1, pady=1)
            current_row += 1

        # Fila Wj/Zj y valor W/Z (usa costos de la fase actual)
        ttk.Label(self.tabla_frame, text="", font=("Arial", 10),
                 relief="ridge", borderwidth=1).grid(row=current_row, column=0, sticky="nsew", padx=1, pady=1)
        etiqueta_sum = "Wⱼ" if self.phase == 1 else "Zⱼ"
        ttk.Label(self.tabla_frame, text=etiqueta_sum, font=("Arial", 10, "bold"),
                 relief="ridge", borderwidth=1, background="#FFFACD").grid(
                 row=current_row, column=1, sticky="nsew", padx=1, pady=1)
        for j in range(n-1):
            ttk.Label(self.tabla_frame, text=_fmt(zj_vals[j]), font=("Arial", 10),
                     relief="ridge", borderwidth=1, background="#FFFACD").grid(
                     row=current_row, column=j+2, sticky="nsew", padx=1, pady=1)
        z_valor = z_rhs
        ttk.Label(self.tabla_frame, text=_fmt(z_valor), font=("Arial", 10),
                 relief="ridge", borderwidth=1, background="#FFFACD").grid(
                 row=current_row, column=n+1, sticky="nsew", padx=1, pady=1)
        ttk.Label(self.tabla_frame, text="", font=("Arial", 10),
                 relief="ridge", borderwidth=1).grid(row=current_row, column=n+2, sticky="nsew", padx=1, pady=1)
        current_row += 1

        # Fila de diferencia (solo de presentación):
        # Mostrar siempre cⱼ - Wⱼ en Fase I y cⱼ - Zⱼ en Fase II, sin alterar la lógica interna.
        ttk.Label(self.tabla_frame, text="", font=("Arial", 10),
                 relief="ridge", borderwidth=1).grid(row=current_row, column=0, sticky="nsew", padx=1, pady=1)
        etiqueta_fila = "cⱼ - Wⱼ" if self.phase == 1 else "cⱼ - Zⱼ"
        ttk.Label(self.tabla_frame, text=etiqueta_fila, font=("Arial", 10, "bold"),
                 relief="ridge", borderwidth=1, background="#FFE4B5").grid(
                 row=current_row, column=1, sticky="nsew", padx=1, pady=1)
        for j in range(n-1):
            # Internamente usamos rj = Zj-cj para MAX y rj = cj-Zj para MIN.
            # Para mostrar cⱼ - Wⱼ/Zⱼ en ambas fases, invertimos el signo en MAX.
            rj_disp = reduced[j] if self.tipo_original == 'min' else -reduced[j]
            ttk.Label(self.tabla_frame, text=_fmt(rj_disp), font=("Arial", 10),
                     relief="ridge", borderwidth=1, background="#FFE4B5").grid(
                     row=current_row, column=j+2, sticky="nsew", padx=1, pady=1)

        # Ajuste de layout
        total_rows = current_row + 1
        total_cols = n + 3
        for i in range(total_rows):
            self.tabla_frame.grid_rowconfigure(i, weight=1)
        for j in range(total_cols):
            self.tabla_frame.grid_columnconfigure(j, weight=1)
        self.tabla_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        # Título
        self.titulo_label.config(text=f"Fase {self.phase} - Iteración {self.iteracion}")
        # Vista previa de Fase II (Iteración 1) bajo la tabla final de Fase I
        if self.phase == 1 and getattr(self, '_fase1_listo_transicion', False):
            try:
                # Construir vista previa: eliminar artificiales y fijar -c original
                tabla2, nombres2 = _eliminar_artificiales_tabla(self.tabla.copy(), self.nombres, self.indices_art)
                n2 = tabla2.shape[1] - 1
                coefs2 = -np.array([self.c_original[j] if j < len(self.c_original) else 0.0 for j in range(n2)])
                tabla2[0, :-1] = coefs2
                # Encabezado
                sep = ttk.Label(self.tabla_frame, text="", font=("Arial", 4))
                sep.grid(row=total_rows+1, column=0, sticky="nsew")
                ttk.Label(self.tabla_frame, text="Fase 2 - Iteración 0", font=("Arial", 14, "bold")).grid(row=total_rows+2, column=0, columnspan=total_cols, sticky="w", padx=2, pady=(6, 4))
                r0 = total_rows + 3
                headers2 = ["cⱼ", "xᵦ"] + [nombres2[i] for i in range(n2)] + ["bᵢ", "bᵢ/yᵢₖ"]
                for col, header in enumerate(headers2):
                    ttk.Label(self.tabla_frame, text=header, font=("Arial", 11, "bold"), relief="ridge", borderwidth=1, background="#E6E6FA").grid(row=r0, column=col, sticky="nsew", padx=1, pady=1)
                # Fila costos
                ttk.Label(self.tabla_frame, text="", font=("Arial", 10), relief="ridge", borderwidth=1).grid(row=r0+1, column=0, sticky="nsew", padx=1, pady=1)
                ttk.Label(self.tabla_frame, text="", font=("Arial", 10), relief="ridge", borderwidth=1).grid(row=r0+1, column=1, sticky="nsew", padx=1, pady=1)
                for j in range(n2):
                    cj2 = self.c_original[j] if j < len(self.c_original) else 0.0
                    ttk.Label(self.tabla_frame, text=_fmt(cj2), font=("Arial", 10), relief="ridge", borderwidth=1, background="#F0F8FF").grid(row=r0+1, column=j+2, sticky="nsew", padx=1, pady=1)
                ttk.Label(self.tabla_frame, text="", font=("Arial", 10), relief="ridge", borderwidth=1).grid(row=r0+1, column=n2+2, sticky="nsew", padx=1, pady=1)
                ttk.Label(self.tabla_frame, text="", font=("Arial", 10), relief="ridge", borderwidth=1).grid(row=r0+1, column=n2+3, sticky="nsew", padx=1, pady=1)
                # Básicas
                subA2 = tabla2[1:, :-1]
                m2, nsub = subA2.shape
                bas2 = []
                for i in range(m2):
                    vj = None
                    for j in range(nsub):
                        col = subA2[:, j]
                        if abs(col[i]-1.0) < 1e-10 and np.all(np.delete(np.abs(col), i) < 1e-10):
                            vj = j
                            break
                    bas2.append(0 if vj is None else vj)
                # Zj y costos reducidos
                zj2 = []
                for j in range(nsub):
                    acc = 0.0
                    for i in range(m2):
                        var_idx = bas2[i]
                        cB = self.c_original[var_idx] if var_idx < len(self.c_original) else 0.0
                        acc += cB * subA2[i, j]
                    zj2.append(acc)
                red2 = []
                for j in range(nsub):
                    cj = self.c_original[j] if j < len(self.c_original) else 0.0
                    red2.append((zj2[j]-cj) if self.tipo_original=='max' else (cj - zj2[j]))
                # Filas
                rr = r0+2
                for i in range(1, m2+1):
                    var_idx = bas2[i-1]
                    cB_show = self.c_original[var_idx] if var_idx < len(self.c_original) else 0.0
                    ttk.Label(self.tabla_frame, text=_fmt(cB_show), font=("Arial", 10), relief="ridge", borderwidth=1).grid(row=rr, column=0, sticky="nsew", padx=1, pady=1)
                    var_nombre = nombres2[var_idx] if var_idx < len(nombres2) else f"R{i}"
                    ttk.Label(self.tabla_frame, text=var_nombre, font=("Arial", 10, "bold"), relief="ridge", borderwidth=1, background="#FFE4E1").grid(row=rr, column=1, sticky="nsew", padx=1, pady=1)
                    for j in range(nsub):
                        ttk.Label(self.tabla_frame, text=_fmt(tabla2[i, j]), font=("Arial", 10), relief="ridge", borderwidth=1).grid(row=rr, column=j+2, sticky="nsew", padx=1, pady=1)
                    bi2 = tabla2[i, -1]
                    ttk.Label(self.tabla_frame, text=_fmt(bi2), font=("Arial", 10), relief="ridge", borderwidth=1, background="#F0FFF0").grid(row=rr, column=nsub+2, sticky="nsew", padx=1, pady=1)
                    # Razón mínima sugerida
                    tol2 = 1e-10
                    cand = [jj for jj, v in enumerate(red2) if v < -tol2]
                    orden2 = sorted(cand, key=lambda jj: red2[jj])
                    col_e = None
                    for jj in orden2:
                        if np.any(tabla2[1:, jj] > 1e-10):
                            col_e = jj
                            break
                    razon_tx = "-"
                    if col_e is not None and tabla2[i, col_e] > 1e-10:
                        razon_tx = _fmt(bi2 / tabla2[i, col_e])
                    ttk.Label(self.tabla_frame, text=razon_tx, font=("Arial", 10), relief="ridge", borderwidth=1, background="#FFF8DC").grid(row=rr, column=nsub+3, sticky="nsew", padx=1, pady=1)
                    rr += 1
                # Zj/Z y criterio
                ttk.Label(self.tabla_frame, text="", font=("Arial", 10), relief="ridge", borderwidth=1).grid(row=rr, column=0, sticky="nsew", padx=1, pady=1)
                ttk.Label(self.tabla_frame, text="Zⱼ", font=("Arial", 10, "bold"), relief="ridge", borderwidth=1, background="#FFFACD").grid(row=rr, column=1, sticky="nsew", padx=1, pady=1)
                for j in range(nsub):
                    ttk.Label(self.tabla_frame, text=_fmt(zj2[j]), font=("Arial", 10), relief="ridge", borderwidth=1, background="#FFFACD").grid(row=rr, column=j+2, sticky="nsew", padx=1, pady=1)
                z_rhs2 = 0.0
                for i in range(m2):
                    var_idx = bas2[i]
                    cB = self.c_original[var_idx] if var_idx < len(self.c_original) else 0.0
                    z_rhs2 += cB * tabla2[i+1, -1]
                ttk.Label(self.tabla_frame, text=_fmt(z_rhs2), font=("Arial", 10), relief="ridge", borderwidth=1, background="#FFFACD").grid(row=rr, column=nsub+2, sticky="nsew", padx=1, pady=1)
                ttk.Label(self.tabla_frame, text="", font=("Arial", 10), relief="ridge", borderwidth=1).grid(row=rr, column=nsub+3, sticky="nsew", padx=1, pady=1)
                rr += 1
                ttk.Label(self.tabla_frame, text="", font=("Arial", 10), relief="ridge", borderwidth=1).grid(row=rr, column=0, sticky="nsew", padx=1, pady=1)
                lblcrit = "cⱼ - Zⱼ"
                ttk.Label(self.tabla_frame, text=lblcrit, font=("Arial", 10, "bold"), relief="ridge", borderwidth=1, background="#FFE4B5").grid(row=rr, column=1, sticky="nsew", padx=1, pady=1)
                for j in range(nsub):
                    # Mostrar en formato cⱼ - Zⱼ: para MAX invertir signo del rj interno (Zⱼ - cⱼ)
                    rj_disp = red2[j] if self.tipo_original == 'min' else -red2[j]
                    ttk.Label(self.tabla_frame, text=_fmt(rj_disp), font=("Arial", 10), relief="ridge", borderwidth=1, background="#FFE4B5").grid(row=rr, column=j+2, sticky="nsew", padx=1, pady=1)
                # Ajustar layout extra
                for i in range(rr+1):
                    self.tabla_frame.grid_rowconfigure(i, weight=1)
                for j in range(max(total_cols, nsub+4)):
                    self.tabla_frame.grid_columnconfigure(j, weight=1)
                self.tabla_frame.update_idletasks()
                self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            except Exception:
                pass

    def _es_optima(self):
        coefs_objetivo = self.tabla[0, :-1]
        return np.all(coefs_objetivo >= -1e-10)

    def _verificar_optimalidad(self):
        # Evalúa optimalidad usando rⱼ (definición interna). Maneja mensajes y transición entre fases.
        # Evaluar también durante la tabla de transición de Fase II; si ya es óptima, mostrar popup
        # Usar costos reducidos para criterio de optimalidad en ambas fases
        _, _, reduced = self._compute_basics_zj_reduced()
        tol = 1e-10
        # Con nuestra definición de r_j, la optimalidad se detecta si todos r_j >= 0
        es_optima = np.all(np.array(reduced) >= -tol)
        if es_optima:
            if self.phase == 1:
                # Chequear factibilidad
                if abs(self.tabla[0, -1]) > 1e-8:
                    messagebox.showwarning("Resultado", "El problema es infactible (Fase I)")
                    self.btn_siguiente.config(state="disabled")
                    return True
                # Mostrar Fase I factible y esperar un clic para transicionar a Fase II,
                # además renderizar la vista previa de Fase II (Iteración 1) debajo
                self._fase1_mostrada = True
                self._fase1_listo_transicion = True
                try:
                    self.status_label.config(text="Fase I factible (W = 0). Abajo se muestra la entrada de Fase II - Iteración 0. Pulse 'Siguiente Iteración' para continuar")
                except Exception:
                    pass
                # Redibujar para incluir la vista previa
                self._mostrar_tabla()
                return False
            else:
                self._mostrar_solucion_optima()
                return True
        else:
            # Si estamos en transición de Fase II, no pivotear todavía; solo actualizar mensaje y esperar clic
            if self.phase == 2 and getattr(self, '_fase2_mostrar_transicion', False):
                try:
                    self.status_label.config(text="Fase II: Iteración 0 (tabla de transición). Pulse 'Siguiente Iteración' para continuar")
                except Exception:
                    pass
                return False
            # Elegir columna entrante: r_j < 0 con alguna entrada positiva
            tol = 1e-10
            candidatos = [j for j, v in enumerate(reduced) if v < -tol]
            orden = sorted(candidatos, key=lambda jj: reduced[jj])
            chosen = None
            for j in orden:
                if np.any(self.tabla[1:, j] > 1e-10):
                    chosen = j
                    break
            if chosen is None:
                # No hay columna con avance factible.
                # En Fase I: NO transicionar automáticamente; mostrar mensaje y esperar clic.
                if self.phase == 1:
                    if not getattr(self, '_fase1_mostrada', False):
                        self._fase1_mostrada = True
                    self.status_label.config(text="Fase I: sin columna entrante factible. Pulse 'Siguiente Iteración' para pasar a Fase II")
                    return False
                # En Fase II: resolver como óptimo (no hay mejora posible)
                self._mostrar_solucion_optima()
                return True
            col_entrada = int(chosen)
            # Determinar variable saliente por razón mínima
            col_vals = self.tabla[1:, col_entrada]
            rhs = self.tabla[1:, -1]
            razones = [(rhs[i]/col_vals[i], i) for i in range(len(col_vals)) if col_vals[i] > 1e-10]
            if razones:
                _, idx = min(razones)
                fila_salida = idx + 1
                # Identificar nombre de la variable saliente (básica actual en esa fila)
                subA = self.tabla[1:, :-1]
                var_sal_idx = None
                for j in range(subA.shape[1]):
                    col = subA[:, j]
                    if abs(col[idx] - 1.0) < 1e-10 and np.all(np.delete(np.abs(col), idx) < 1e-10):
                        var_sal_idx = j
                        break
                nombre_saliente = self.nombres[var_sal_idx] if var_sal_idx is not None else f"Fila {fila_salida}"
                self.status_label.config(text=f"No es óptima. Entra: {self.nombres[col_entrada]} | Sale: {nombre_saliente}")
            else:
                self.status_label.config(text=f"No es óptima. Entra: {self.nombres[col_entrada]}")
            return False

    def _transicionar_a_fase2(self):
        # Transición clásica (ya no usada para el salto directo); conserva utilidades si se requiere
        # Eliminar artificiales
        self.tabla, self.nombres = _eliminar_artificiales_tabla(self.tabla, self.nombres, self.indices_art)
        # Reset objetivo a -c_original extendido (sin canonicalizar aún para mostrar todas las tablas)
        ncols = self.tabla.shape[1] - 1
        coefs = -np.array([self.c_original[j] if j < len(self.c_original) else 0.0 for j in range(ncols)])
        self.tabla[0, :-1] = coefs
        self.phase = 2
        # Mostrar también la Iteración 0 de Fase II (tabla inicial con costos originales y sin artificiales)
        self.iteracion = 0
        self._fase2_mostrar_transicion = True
        # Actualizar título de ventana para reflejar Fase II (Z)
        try:
            self.title("Método Simplex - Fase II (Z)")
        except Exception:
            pass
        self.status_label.config(text="Fase II: Iteración 0 (tabla de transición). Pulse 'Siguiente Iteración' para continuar")
        self._mostrar_tabla()

    def _mostrar_solucion_optima(self):
        # Construye y muestra el popup de solución óptima, deshabilitando el avance
        # Permitir mostrar popup aún si estamos en la tabla de transición de Fase II
        self.status_label.config(text="¡SOLUCIÓN ÓPTIMA ENCONTRADA!")
        self.btn_siguiente.config(state="disabled")
        # Extraer solución básica
        m2, n2 = self.tabla.shape
        sol = {self.nombres[j]: 0.0 for j in range(n2-1)}
        for i in range(1, m2):
            for j in range(n2-1):
                col = self.tabla[1:, j]
                if abs(col[i-1]-1.0) < 1e-8 and np.all(np.delete(np.abs(col), i-1) < 1e-8):
                    sol[self.nombres[j]] = float(self.tabla[i, -1])
                    break
        # Calcular el valor óptimo como sum(c_B * b) usando la base actual
        subA = self.tabla[1:, :-1]
        rhs_b = self.tabla[1:, -1]
        # Detectar columnas básicas por fila
        basicas_idx = []
        for i in range(subA.shape[0]):
            var_idx = None
            for j in range(subA.shape[1]):
                col = subA[:, j]
                if abs(col[i] - 1.0) < 1e-10 and np.all(np.delete(np.abs(col), i) < 1e-10):
                    var_idx = j
                    break
            basicas_idx.append(0 if var_idx is None else var_idx)
        valor = 0.0
        for i, var_idx in enumerate(basicas_idx):
            if self.phase == 1:
                signo_art = -1.0 if self.tipo_original == 'max' else 1.0
                cB = signo_art if var_idx in set(self.indices_art) else 0.0
            else:
                cB = self.c_original[var_idx] if var_idx < len(self.c_original) else 0.0
            valor += cB * rhs_b[i]
        # Mostrar
        nombres_vars = [f"x{i+1}" for i in range(self.n_original)]
        texto = [
            "=== SOLUCIÓN ÓPTIMA ===",
            f"Tipo: {self.tipo_original.upper()}",
            f"Valor óptimo: {_fmt(valor)}",
            "",
            "Variables:",
        ]
        for nm in nombres_vars:
            texto.append(f"{nm} = {_fmt(sol.get(nm, 0.0))}")
        extras = [k for k in sol.keys() if k not in nombres_vars]
        if extras:
            texto.append("")
            texto.append("Variables adicionales:")
            for nm in sorted(extras):
                texto.append(f"{nm} = {_fmt(sol[nm])}")
        messagebox.showinfo("Solución Óptima", "\n".join(texto))

    def _siguiente_iteracion(self):
        # Avanza el algoritmo una iteración; también aplica la transición Fase I -> Fase II
        # Si Fase I quedó factible y estamos listos para transicionar, haz la transición ahora
        if self.phase == 1 and getattr(self, '_fase1_listo_transicion', False):
            self._fase1_listo_transicion = False
            # Construir Fase II directamente como Iteración 1
            try:
                self.tabla, self.nombres = _eliminar_artificiales_tabla(self.tabla, self.nombres, self.indices_art)
            except Exception:
                pass
            ncols = self.tabla.shape[1] - 1
            coefs = -np.array([self.c_original[j] if j < len(self.c_original) else 0.0 for j in range(ncols)])
            self.tabla[0, :-1] = coefs
            self.phase = 2
            self.iteracion = 0
            self._mostrar_tabla()
            self._verificar_optimalidad()
            return
        # Si estamos en la tabla de transición de Fase II, avanzar a Iteración 1 sin pivotear
        if self.phase == 2 and getattr(self, '_fase2_mostrar_transicion', False):
            self._fase2_mostrar_transicion = False
            self.iteracion = 1
            self._mostrar_tabla()
            # Evaluar optimalidad/pivoteo normalmente después
            self._verificar_optimalidad()
            return
        # Verificar optimalidad; si ya es óptima, se muestra popup y salimos
        before_phase = self.phase
        if self._verificar_optimalidad():
            return
        # Si en esta llamada se produjo la transición de Fase I -> Fase II,
        # detenerse para mostrar la tabla de transición (Iteración 0)
        if before_phase == 1 and self.phase == 2 and getattr(self, '_fase2_mostrar_transicion', False):
            return
    # Selección de pivote: r_j < 0 (definición ya contempla fase/tipo)
        _, _, reduced = self._compute_basics_zj_reduced()
        tol = 1e-10
        candidatos = [j for j, v in enumerate(reduced) if v < -tol]
        orden = sorted(candidatos, key=lambda jj: reduced[jj])
        chosen = None
        for j in orden:
            if np.any(self.tabla[1:, j] > 1e-10):
                chosen = j
                break
        if chosen is None:
            # No hay avance factible
            if self.phase == 1:
                # Interpretar como fin de Fase I con W=0 y pasar a Fase II
                self._transicionar_a_fase2()
            else:
                self._mostrar_solucion_optima()
            return
        col_pivote = int(chosen)
        columna_pivote = self.tabla[1:, col_pivote]
        rhs = self.tabla[1:, -1]
        razones_validas = []
        for i in range(len(columna_pivote)):
            if columna_pivote[i] > 1e-10:
                razon = rhs[i] / columna_pivote[i]
                if razon >= 0:
                    razones_validas.append((razon, i))
        if not razones_validas:
            messagebox.showerror("Error", "Problema no acotado - no hay solución finita")
            return
        _, min_indice = min(razones_validas)
        fila_pivote = min_indice + 1
        _pivotear(self.tabla, fila_pivote, col_pivote)
        self.iteracion += 1
        self._mostrar_tabla()
        self._verificar_optimalidad()


if __name__ == "__main__":
    # Sugerir AppID antes de crear ventanas (mejora uso de icono en barra de tareas en Windows)
    _set_windows_app_id()
    app = SimplizadorGUI()
    app.mainloop()
