import os, sys
import numpy as np

""" Formatea números evitando notación científica y eliminando ceros innecesarios """
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
