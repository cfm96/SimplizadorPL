import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from src.utilities import (_set_window_icon,   _dbg)
from src.simplex_functions import (
    _fmt
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
