# Perceptrón Monocapa - Compuerta AND

Aplicación web interactiva hecha con **Python y Streamlit** que entrena un perceptrón monocapa para aprender la compuerta lógica AND y muestra, paso a paso, todas las operaciones matemáticas que realiza.

Integrantes: Ortega Plaza diego, Valdez Curiel Diana

---

## ¿Qué hace?

- Dibuja el esquema de la red: entradas `x1` y `x2`, pesos `w1` y `w2`, bias `b`, suma ponderada, función escalón y salida `y`.
- Permite elegir los pesos iniciales, el bias, la tasa de aprendizaje (η) y el límite de épocas.
- Entrena el perceptrón con la tabla de verdad del AND y detiene el entrenamiento cuando una época completa no tiene errores.
- Muestra una tabla resumen por época (pesos, bias y errores).
- Desglosa en LaTeX cada operación de cada muestra: suma ponderada, escalón, error y actualización de pesos.
- Incluye un módulo de prueba: se ingresa cualquier `(x1, x2)` y la salida se calcula en tiempo real sobre el esquema de la red.
- Genera un reporte en PDF descargable con la tabla de épocas y el procedimiento completo.

## Estructura del proyecto

El código está separado por responsabilidades:

```
perceptron/
├── perceptron.py      # Logica del aprendizaje y almacenamiento del historial
├── pdf_generator.py   # Generacion del reporte en PDF
├── app.py             # Interfaz web (Streamlit)
├── .gitignore
└── README.md
```

| Archivo | Qué contiene |
|---|---|
| `perceptron.py` | Clase `Perceptron` con `predecir()` y `entrenar()`. No depende de Streamlit ni del PDF. |
| `pdf_generator.py` | Función `generar_pdf(p)`, que recibe un perceptrón ya entrenado y devuelve el PDF en bytes. |
| `app.py` | Controles, esquema de la red, tabla de épocas, procedimiento paso a paso, módulo de prueba y botón de descarga. |

## Requisitos

- Python 3.13.9
- Las librerías `streamlit` y `fpdf2`

## Instalación y ejecución

Desde la carpeta del proyecto (PowerShell en Windows):

```
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install streamlit fpdf2
python -m streamlit run app.py
```



## Cómo usarla

1. En la barra lateral, escribir los pesos iniciales, el bias, la tasa de aprendizaje y el límite de épocas.
2. Presionar **Entrenar**.
3. Revisar el mensaje de resultado (si convergió o no), la tabla por épocas y el procedimiento paso a paso (cada época se abre con un clic).
4. En la sección de prueba, cambiar `x1` y `x2` para ver la salida del perceptrón entrenado.
5. Presionar **Descargar reporte PDF** para obtener el informe.

## Cómo aprende

Para cada muestra `(x1, x2)` con salida deseada `t`:

```
z = (x1 * w1) + (x2 * w2) + b        suma ponderada
y = f(z)                             escalon: 1 si z >= 0, 0 en otro caso
e = t - y                            error

w1 = w1 + eta * e * x1               regla delta
w2 = w2 + eta * e * x2
b  = b  + eta * e                    el bias tiene entrada fija 1
```

Una **época** es una pasada por las cuatro muestras del AND. Si en una época no hay errores, el perceptrón aprendió y el entrenamiento se detiene.

## Ejemplo

Con `w1 = 0`, `w2 = 0`, `b = 0`, `η = 0.1` y 20 épocas como límite, converge en 6 épocas:

| Época | w1 | w2 | b | Errores |
|---|---|---|---|---|
| 1 | 0.1 | 0.1 | 0 | 2 |
| 2 | 0.2 | 0.1 | -0.1 | 3 |
| 3 | 0.2 | 0.1 | -0.2 | 3 |
| 4 | 0.2 | 0.2 | -0.2 | 2 |
| 5 | 0.2 | 0.1 | -0.3 | 1 |
| 6 | 0.2 | 0.1 | -0.3 | 0 |

Pesos finales: `w1 = 0.2`, `w2 = 0.1`, `b = -0.3`. Solo la entrada (1, 1) activa la neurona, como en el AND.
