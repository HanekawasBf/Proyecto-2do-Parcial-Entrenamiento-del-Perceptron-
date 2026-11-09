import streamlit as st
from perceptron import Perceptron
from pdf_generator import generar_pdf

st.set_page_config(page_title="Perceptrón AND", layout="wide")
st.title("Perceptrón - Compuerta AND")

# Redondea para mostrar (evita 0.20000000000000004 y -0).
    # round(x, 4) deja 4 decimales como maximo.
    # El "+ 0.0" convierte un -0.0 en 0.0 para que nunca se muestre "-0".
    # El formato ":g" quita ceros sobrantes (0.10 -> 0.1, 2.0 -> 2).
def n(x):
    return f"{round(x, 4) + 0.0:g}"


def dibujar_red(w1, w2, b, x1=None, x2=None, y=None):
    # Si no se pasan x1, x2 o y, se dibuja solo el nombre del nodo.
    # Si se pasan, el nodo muestra el valor
    et_x1 = "x1" if x1 is None else f"x1 = {n(x1)}"
    et_x2 = "x2" if x2 is None else f"x2 = {n(x2)}"
    et_y = "y" if y is None else f"y = {y}"

    # Esto es texto en lenguaje DOT (el de Graphviz), no Python.
    # Las llaves se escriben dobles {{ }} porque en un f-string
    # una sola llave se usa para insertar variables.
    # rankdir=LR dibuja el grafo de izquierda a derecha.
    dot = f"""
    digraph {{
        rankdir=LR;
        node [shape=circle, fontsize=14];
        x1 [label="{et_x1}"];
        x2 [label="{et_x2}"];
        bias [label="b = {n(b)}", shape=box];
        suma [label="Σ"];
        f [label="Escalón", shape=box];
        y [label="{et_y}"];
        x1 -> suma [label="w1 = {n(w1)}"];
        x2 -> suma [label="w2 = {n(w2)}"];
        bias -> suma;
        suma -> f -> y;
    }}
    """
    st.graphviz_chart(dot)


# Hiperparametros (barra lateral)
# Cada uno devuelve el valor que tiene en este momento.
st.sidebar.header("Hiperparámetros")
w1 = st.sidebar.number_input("Peso inicial w1", value=0.0, step=0.1)
w2 = st.sidebar.number_input("Peso inicial w2", value=0.0, step=0.1)
b = st.sidebar.number_input("Bias inicial b", value=0.0, step=0.1)
eta = st.sidebar.slider("Tasa de aprendizaje (η)", 0.01, 1.0, 0.1, 0.01)
max_epocas = st.sidebar.number_input("Límite de épocas", min_value=1,
                                     max_value=1000, value=20, step=1)

#las variables normales se
# "olvidan". st.session_state es un diccionario que SOBREVIVE entre
# ejecuciones y aqui guarda el perceptron
# ya entrenado para poder usarlo en la prueba, la tabla y el PDF.
if st.sidebar.button("Entrenar"):          # el boton es True solo al hacer clic
    p = Perceptron(w1, w2, b, eta)
    p.entrenar(int(max_epocas))
    st.session_state["perceptron"] = p

# Esquema de la red y prueba
st.header("1. Esquema de la red")
if "perceptron" in st.session_state:       # solo si ya se entreno
    p = st.session_state["perceptron"]

    st.subheader("Prueba del perceptrón entrenado")
    c1, c2 = st.columns(2)                 
    # key="..." le da un nombre unico al widget para que Streamlit
    # recuerde su valor entre ejecuciones
    x1_prueba = c1.number_input("x1", value=1.0, step=1.0, key="x1_prueba")
    x2_prueba = c2.number_input("x2", value=1.0, step=1.0, key="x2_prueba")

    # Usa los pesos ya entrenados y NO los modifica (solo predice)
    z_prueba, y_prueba = p.predecir(x1_prueba, x2_prueba)

    dibujar_red(p.w1, p.w2, p.b, x1_prueba, x2_prueba, y_prueba)

    # st.latex dibuja formulas matematicas. El prefijo rf"..." combina:
    #   r = texto "crudo"
    #   f = f-string (para insertar variables con {})
    st.latex(rf"z = ({n(x1_prueba)} \cdot {n(p.w1)}) + "
             rf"({n(x2_prueba)} \cdot {n(p.w2)}) + ({n(p.b)}) = "
             rf"{n(z_prueba)} \;\Rightarrow\; y = {y_prueba}")
else:
    dibujar_red(w1, w2, b)                 # antes de entrenar: pesos iniciales

# Resultados 
if "perceptron" in st.session_state:
    p = st.session_state["perceptron"]
    ultima = p.epocas[-1]                  # [-1] = ultimo elemento de la lista

    # Si la ultima epoca tuvo 0 errores, el perceptron aprendio
    if ultima["errores"] == 0:
        st.success(f"Convergió en {len(p.epocas)} épocas")
    else:
        st.warning(f"No convergió en {len(p.epocas)} épocas")

    # Exportacion a PDF 
    # generar_pdf(p) viene de pdf_generator.py y devuelve el PDF en bytes.
    # El boton lo ofrece como descarga.
    st.download_button("Descargar reporte PDF",
                       data=generar_pdf(p),
                       file_name="reporte_perceptron.pdf",
                       mime="application/pdf")

    # Tabla resumen por epocas 
    st.header("2. Tabla resumen por épocas")
    # arma una lista de diccionarios, uno por epoca.
    # Cada diccionario se convierte en una fila de la tabla.
    # Se usa n() para mostrar los numeros limpios.
    
    # Tabla en formato Markdown
    tabla = "| Época | w1 | w2 | b | Errores |\n|---|---|---|---|---|\n"
    for e in p.epocas:
        tabla += (f"| {e['epoca']} | {n(e['w1'])} | {n(e['w2'])} | "
                  f"{n(e['b'])} | {e['errores']} |\n")
    st.markdown(tabla)

    # Procedimiento paso a paso
    st.header("3. Procedimiento paso a paso")
    st.latex(r"f(z) = 1 \text{ si } z \geq 0, \quad f(z) = 0 \text{ en otro caso}")

    for e in p.epocas:
        with st.expander(f"Época {e['epoca']}  (errores: {e['errores']})"):
            # p.pasos tiene las muestras de TODAS las epocas mezcladas;
            # aqui se filtran solo las de la epoca actual
            pasos_epoca = [s for s in p.pasos if s["epoca"] == e["epoca"]]

            # enumerate numera las muestras: 1, 2, 3, 4
            for i, s in enumerate(pasos_epoca, start=1):
                st.markdown(f"**Muestra {i}:** "
                            f"$(x_1, x_2) = ({s['x1']}, {s['x2']})$, "
                            f"$t = {s['t']}$")
                # 1 suma ponderada, con los pesos de ANTES de actualizar
                st.latex(rf"z = ({s['x1']} \cdot {n(s['w1_ant'])}) + "
                         rf"({s['x2']} \cdot {n(s['w2_ant'])}) + "
                         rf"({n(s['b_ant'])}) = {n(s['z'])}")
                # 2 salida del escalon y error
                st.latex(rf"y = f(z) = {s['y']}, \quad "
                         rf"e = t - y = {s['t']} - {s['y']} = {s['e']}")
                # 3 regla delta: peso nuevo = peso anterior + eta * e * x
                st.latex(rf"w_1 = {n(s['w1_ant'])} + {n(p.eta)} \cdot "
                         rf"({s['e']}) \cdot {s['x1']} = {n(s['w1_nuevo'])}")
                st.latex(rf"w_2 = {n(s['w2_ant'])} + {n(p.eta)} \cdot "
                         rf"({s['e']}) \cdot {s['x2']} = {n(s['w2_nuevo'])}")
                # el bias se multiplica por 1 porque su entrada es fija
                st.latex(rf"b = {n(s['b_ant'])} + {n(p.eta)} \cdot "
                         rf"({s['e']}) \cdot 1 = {n(s['b_nuevo'])}")
                st.divider()               # linea separadora entre muestras
else:
    st.info("Configura los hiperparámetros y presiona **Entrenar**.")