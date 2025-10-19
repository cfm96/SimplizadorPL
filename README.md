# Simplizador de Programación Lineal (GUI + Simplex 2 Fases)

## Descripción general

Aplicación en Python con interfaz gráfica (tkinter) para:
- Capturar un problema de Programación Lineal (PL) en forma general (MAX o MIN).
- Mostrar el problema estandarizado (igualdades, variables no negativas) evitando notación científica.
- Resolver con el método Simplex de 2 Fases paso a paso, mostrando todas las tablas (Fase I y Fase II), con una vista previa didáctica de la entrada a Fase II.

El archivo principal es `simplizador_simplex.py`.

## Requisitos
- Python 3.8+.
- Paquetes estándar: `tkinter` (incluido en la mayoría de instalaciones de Python) y `numpy`.

## Cómo ejecutar

Desde PowerShell (Windows):

```powershell
# (Opcional) Activar tu entorno virtual si lo usas
# .venv\Scripts\Activate.ps1

python .\simplizador_simplex.py
```

Se abrirá la ventana "Simplizador de P.L.".

## Uso rápido (GUI)
1. Selecciona el tipo de problema: `max` o `min`.
2. Ingresa la cantidad de variables y restricciones (hasta 50 y 50).
3. Pulsa "Continuar" y completa:
   - Coeficientes de la función objetivo.
   - Para cada restricción: coeficientes, operador (<=, >=, =) y RHS (lado derecho).
4. Pulsa "Estandarizar" para ver:
   - El problema estandarizado (igualdades y variables no negativas) con nombres `x1..xK`.
   - Si aplica, un bloque "Problema auxiliar (Fase I)" con las variables artificiales indicadas.
5. Pulsa "Resolver (Simplex)" para abrir el visor de tablas del método de 2 fases.
6. En la ventana del Simplex, usa "Siguiente Iteración" para avanzar paso a paso hasta alcanzar el óptimo.

## Ejemplo completo (MAX con <=, >= y =)

Este ejemplo activa la Fase I (por tener >= y =) y muestra la vista previa “Fase 2 - Iteración 0”.

Problema:

Max Z = 3x1 + 2x2

Sujeto a:
- x1 + x2 <= 4
- x1 + 2x2 >= 2
- x1 - x2 = 1
- x1, x2 >= 0

Ingreso en la GUI:
- Tipo: max
- N° variables: 2
- N° restricciones: 3
- Función objetivo: [3, 2]
- Restricciones:
  - R1: [1, 1] <= 4
  - R2: [1, 2] >= 2
  - R3: [1, -1] = 1

Qué verás:
1) Estandarización (texto):
  - Se agregará una holgura a R1.
  - Se agregará un exceso y una artificial a R2.
  - Se agregará una artificial a R3.
  - Las artificiales aparecerán al final del listado de variables.
2) Al pulsar “Resolver (Simplex)”, se abrirá el visor de 2 Fases:
  - Fase I (W): la fila de costos mostrará ±1 solo en columnas artificiales; la fila de suma se etiqueta Wⱼ, y la fila de criterio se muestra como cⱼ − Wⱼ (presentación homogénea).
  - Al finalizar Fase I con W = 0 (factible), debajo aparecerá “Fase 2 - Iteración 0”, sin artificiales y con costos originales.
  - Un clic en “Siguiente Iteración” adoptará esa tabla como inicio real de Fase II.
  - En Fase II, la fila de suma será Zⱼ y la fila de criterio se mostrará como cⱼ − Zⱼ.

Resultado esperado del ejemplo:
- Óptimo en x1 = 2.5, x2 = 1.5
- Valor óptimo Z = 10.5

Notas del cálculo (intuición):
- De la igualdad: x1 = x2 + 1.
- R1 implica 2x2 + 1 <= 4 ⇒ x2 <= 1.5.
- R2 implica 3x2 + 1 >= 2 ⇒ x2 >= 1/3.
- Sobre la recta x1 = x2 + 1, Z = 3(x2+1) + 2x2 = 5x2 + 3 se maximiza en x2 = 1.5 ⇒ Z = 10.5.

## Interfaz y visualización de tablas
- La tabla muestra encabezados: `cⱼ | xᵦ | x1 x2 ... | bᵢ | bᵢ/yᵢₖ`.
- Segunda fila: los costos `cⱼ` (Fase I: 0 para no artificiales; ±1 para artificiales; Fase II: coeficientes originales).
- Filas de restricciones (básicas), con sus coeficientes y el RHS `bᵢ` formateado sin notación científica.
- Fila de suma: `Wⱼ` en Fase I o `Zⱼ` en Fase II. El valor de W/Z se calcula como Σ(c_B · b).
- Fila de criterio (presentación):
  - Fase I: muestra `cⱼ − Wⱼ`.
  - Fase II: muestra `cⱼ − Zⱼ`.
  - Nota: La lógica interna usa rⱼ coherente con el tipo de problema; para MAX se invierte el signo solo en la presentación para respetar la etiqueta `cⱼ − Zⱼ`.
- Razones `bᵢ / yᵢₖ`: se muestran por fila cuando `yᵢₖ > 0` (regla clásica del simplex).

## Flujo del método (2 Fases)
- Estandarización de restricciones:
  - Convierte RHS negativo multiplicando la fila por (−1) y ajusta el operador.
  - Agrega holguras/excesos según corresponda.
  - Agrega variables artificiales solo cuando son necesarias (>=, =).
  - Reordena las artificiales al final de las columnas.
- Fase I (función objetivo W):
  - MAX: se maximiza −W, lo que en la tabla equivale a costos −1 para artificiales; MIN: costos +1.
  - Canonicalización de la fila objetivo con las filas básicas artificiales.
  - Se pivotea hasta W = 0 (factibilidad) o hasta no haber mejora posible.
  - Al finalizar factible, debajo se muestra una “Vista previa Fase 2 — Iteración 0”: tabla sin artificiales y con costos originales.
- Transición a Fase II:
  - En el siguiente clic, se adopta la vista previa como tabla real de Fase II con `Iteración 0`.
- Fase II (función objetivo Z):
  - Costos originales del problema, sin variables artificiales.
  - Se pivotea hasta que no haya `rⱼ` negativos (según la convención interna); se muestra el popup con la solución óptima.

## Convenciones y detalles numéricos
- Variables nombradas como `x1, x2, …, xK`.
- Evita notación científica; redondeo amigable con `_fmt` y tolerancias ~1e−10.
- Criterio interno de entrada:
  - MAX: rⱼ = Zⱼ − cⱼ.
  - MIN: rⱼ = cⱼ − Zⱼ.
  - Entra alguna columna con rⱼ < 0 que tenga entradas positivas.
- Salida: regla de la razón mínima `bᵢ / yᵢₖ` con `yᵢₖ > 0`.
- Valor de la función objetivo (W o Z) y RHS se calculan como Σ(c_B · b) con los c_B de la fase correspondiente.

## Casos especiales manejados
- Infactibilidad: si al cerrar Fase I el valor no es 0, se notifica que el problema es infactible.
- No acotado: si no existen razones válidas (todas yᵢₖ ≤ 0 en la columna entrante), se notifica que el problema no es acotado.
- Límites de tamaño: hasta 50 variables y 50 restricciones; el creador de restricciones usa lotes para mantener la UI fluida.

## Estructura del código (resumen)
- Funciones núcleo:
  - `_fmt(x)`: formato numérico sin notación científica.
  - `_estandarizar_problema_dos_fases(...)`: construye A, b, costos extendidos, nombres y posiciones de artificiales; reordena columnas.
  - `_construir_tabla_simplex(A, b, c)`: arma una tabla simplex (fila objetivo + restricciones).
  - `_encontrar_variables_basicas(A, indices_artificiales)`: detecta columnas unitarias por fila.
  - `_pivotear(tabla, fila, col)`: operaciones elementales de pivoteo.
  - `_pivot_out_artificiales_en_fase1(...)`: intenta reemplazar artificiales básicas por variables reales cuando es posible.
  - `_fase_uno(...)` y `_fase_dos(...)`: referencia no GUI.
  - `_eliminar_artificiales_tabla(tabla, nombres, indices_art)`: elimina columnas artificiales.
  - `resolver_dos_fases(...)`: resuelve completo (no GUI) y devuelve solución/estado.
  - `estandarizar_problema(...)`: solo para visualización del problema estándar (sin artificiales).
- Clases GUI:
  - `SimplizadorGUI`: ventana principal para capturar datos y lanzar el solver.
  - `TwoPhaseSimplexWindow`: visor paso a paso de Fase I/Fase II, con vista previa Fase 2 — Iteración 0.
  - `SimplexWindow`: ventana demo del simplex canónico (didáctica; opcional en tu flujo principal).

## Depuración
- Bandera global `DEBUG` al inicio del archivo. Si la pones en `True`, verás trazas detalladas en consola (selección de pivote, razones, etc.). Por defecto está en `False`.

## Glosario
- RHS (Right-Hand Side): lado derecho de las restricciones, vector `b`; en la tabla, la última columna.
- Wⱼ / Zⱼ: suma ponderada Σ(c_B · aᵢⱼ) de la columna j con los costos básicos c_B (en Fase I usa costos de artificiales; en Fase II, costos originales).
- cⱼ − Wⱼ / cⱼ − Zⱼ: fila de criterio mostrada. Para MAX, la lógica interna usa Zⱼ − cⱼ y se multiplica por (−1) para mostrar `cⱼ − Zⱼ` al usuario.
- Razones: `bᵢ / yᵢₖ` (yᵢₖ > 0), criterio de salida.

## Notas
- La UI utiliza scroll vertical y horizontal; la creación de restricciones se hace por lotes y con indicador de progreso para mejorar la experiencia en problemas grandes.
- Los mensajes de error y óptimo se muestran con `messagebox`.