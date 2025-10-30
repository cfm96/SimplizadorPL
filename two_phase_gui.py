import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

from simplex_functions import (
    _estandarizar_problema_dos_fases,
    _fmt,
    _pivotear,
    _construir_tabla_simplex,
    _encontrar_variables_basicas,
    _eliminar_artificiales_tabla,
    )

from utilities import _set_window_icon

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

