import tkinter as tk
from tkinter import ttk, messagebox

from src.simplex_functions import (
    estandarizar_problema,
    resolver_dos_fases,
    _estandarizar_problema_dos_fases,
    _fmt
    )

from src.utilities import _set_window_icon, _resource_path

from src.two_phase_gui import TwoPhaseSimplexWindow

# Ventana principal de la aplicación
class SimplizadorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simplificador")
        self.geometry("800x680")
        _set_window_icon(self)

        ttk.Label(
            self,
            text="SIMPLIZADOR",
            font=("Helvetica", 14, "bold"),
            foreground="gray",
            justify="center"
        ).pack(pady=15)

        self.tipo_var = tk.StringVar(value="max")
        self.n_var = tk.IntVar()
        self.m_var = tk.IntVar()

        # labelframe para selección de tipo, n y m
        config_frame = ttk.LabelFrame(self, text="Configuración problema de P.L.", padding=10)
        config_frame.pack(padx=10, pady=10, fill="x")

        frm_top = ttk.Frame(config_frame)
        frm_top.pack(pady=10)
        ttk.Label(frm_top, text="Clasificación:").grid(row=0, column=0, padx=5)
        ttk.Combobox(frm_top, textvariable=self.tipo_var,
                     values=["max", "min"], width=6).grid(row=0, column=1, padx=5)
        ttk.Label(frm_top, text="N° variables:").grid(row=0, column=2, padx=5)
        ttk.Entry(frm_top, textvariable=self.n_var, width=5).grid(row=0, column=3, padx=5)
        ttk.Label(frm_top, text="N° restricciones:").grid(row=0, column=4, padx=5)
        ttk.Entry(frm_top, textvariable=self.m_var, width=5).grid(row=0, column=5, padx=5)

        # Etiquetas informativas sobre los límites
        frm_limits = ttk.Frame(config_frame)
        frm_limits.pack(pady=(0, 10))
        ttk.Label(frm_limits, text="Límites máximos: 50 variables y 50 restricciones", 
                 font=("Arial", 9), foreground="gray").pack()

        ttk.Button(config_frame, text="Continuar", command=self.crear_campos).pack(pady=10)

        # Labelframe para ingreso de datos del problema
        datos_frame = ttk.LabelFrame(self, text="Ingreso de datos del problema", padding=10)
        datos_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Crear el área con scrollbars
        self.canvas_frame = ttk.Frame(datos_frame)
        self.canvas_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Canvas para el contenido con scroll
        self.canvas = tk.Canvas(self.canvas_frame, bg=self.cget('bg'))
        
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

