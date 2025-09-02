import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Pronóstico", layout="wide")

st.title("Pronóstico")
st.write("""
Ingresa tus datos históricos para generar la **tabla y el gráfico**.
Contiene los siguientes métodos:
""")

# Crear 6 columnas (una por método)
cols = st.columns(6)

# Insertar un método en cada columna
metodos = [
    "Promedio simple",
    "Línea con tendencia",
    "Curva cíclica (promedio)",
    "Cíclica con tendencia",
    "Geométrica (a·t^b)",
    "Exponencial (a·b^t)",
    # Si quieres más, crea más columnas
]

# Mostrar los métodos en horizontal
for col, metodo in zip(cols, metodos):
    col.markdown(f"✅ **{metodo}**")

# ========= utilidades matemáticas =========
def solve_least_squares(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
    return beta

def mse(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return np.mean((yhat - y) ** 2)

# ========= modelos =========
def modelo_promedio_simple(y):
    m = float(np.mean(y))
    return {"params": {"a": m},
            "fit": np.full_like(y, m, dtype=float),
            "predict": lambda tt: float(m)}

def modelo_linea(y, t):
    X = np.column_stack([np.ones_like(t, dtype=float), t])
    beta = solve_least_squares(X, y)  # a, b
    a, b = beta
    fit = a + b * t
    return {"params": {"a": float(a), "b": float(b)},
            "fit": fit,
            "predict": lambda tt: float(a + b * tt)}

def modelo_cuadratica(y, t):
    X = np.column_stack([np.ones_like(t, dtype=float), t, t**2])
    beta = solve_least_squares(X, y)  # a, b, c
    a, b, c = beta
    fit = a + b * t + c * (t**2)
    return {"params": {"a": float(a), "b": float(b), "c": float(c)},
            "fit": fit,
            "predict": lambda tt: float(a + b*tt + c*tt*tt)}

def modelo_exponencial(y, t):
    y_pos = np.maximum(y, 1e-9)
    ly = np.log(y_pos)
    X = np.column_stack([np.ones_like(t, dtype=float), t])
    alpha, beta = solve_least_squares(X, ly)
    a = math.exp(alpha); b = math.exp(beta)
    fit = a * (b ** t)
    return {"params": {"a": float(a), "b": float(b)},
            "fit": fit,
            "predict": lambda tt: float(a * (b ** tt))}

def modelo_geometrica(y, t):
    lt = np.log(np.maximum(t, 1e-9))
    ly = np.log(np.maximum(y, 1e-9))
    X = np.column_stack([np.ones_like(lt, dtype=float), lt])
    lna, b = solve_least_squares(X, ly)
    a = math.exp(lna)
    fit = a * (t ** b)
    return {"params": {"a": float(a), "b": float(b)},
            "fit": fit,
            "predict": lambda tt: float(a * (tt ** b))}

def modelo_ciclica_promedio(y, t, N):
    cos = np.cos(2*np.pi*t/N); sin = np.sin(2*np.pi*t/N)
    X = np.column_stack([np.ones_like(t, dtype=float), cos, sin])
    a, u, v = solve_least_squares(X, y)
    fit = a + u*cos + v*sin
    return {
        "params": {"a": float(a), "u": float(u), "v": float(v), "N": int(N)},
        "fit": fit,
        "predict": lambda tt: float(a + u*math.cos(2*math.pi*tt/N) + v*math.sin(2*math.pi*tt/N))
    }

    # return {
    #     "params": {"a": float(a), "u": float(u), "v": float(v), "N": int(N)},
    #     "fit": fit,
    #     "predict": lambda tt: float(a + u*math.cos(2*math.pi*tt/N) + v*math.sin(2*math.pi*tt/N))}

def modelo_ciclica_tendencia(y, t, N):
    cos = np.cos(2*np.pi*t/N); sin = np.sin(2*np.pi*t/N)
    X = np.column_stack([np.ones_like(t, dtype=float), t, cos, sin])
    a, b, u, v = solve_least_squares(X, y)
    fit = a + b*t + u*cos + v*sin
    return {"params": {"a": float(a), "b": float(b), "u": float(u), "v": float(v), "N": int(N)},
            "fit": fit,
            "predict": lambda tt: float(a + b*tt + u*math.cos(2*math.pi*tt/N) + v*math.sin(2*math.pi*tt/N))}

def calcular_todo(df, horizon, N, pesos=None):
    years = df["Año"].to_numpy(dtype=int)
    y = df["Soles"].to_numpy(dtype=float)
    t = np.arange(1, len(y)+1, dtype=float)
    if N is None or N <= 0:
        N = len(df)

    # ===== modelos =====
    m_prom = modelo_promedio_simple(y)
    m_lin  = modelo_linea(y, t)
    m_cpr  = modelo_ciclica_promedio(y, t, N)
    m_ct   = modelo_ciclica_tendencia(y, t, N)
    m_geo  = modelo_geometrica(y, t)
    m_exp  = modelo_exponencial(y, t)
    m_pot  = modelo_cuadratica(y, t)

    # ===== tabla histórico =====
    hist = pd.DataFrame({
        "Año": years,
        "Periodo": t.astype(int),
        "Soles": y,
        "Promedio Simple": m_prom["fit"],
        "Línea con Tendencia": m_lin["fit"],
        "Curva Cíclica Promedio": m_cpr["fit"],
        "Cíclica con Tendencia": m_ct["fit"],
        "Curva Geométrica": m_geo["fit"],
        "Curva Exponencial": m_exp["fit"],
        "Curva Potencial": m_pot["fit"],
    })

    # ===== errores (sobre histórico) =====
    def custom_error(y_real, y_fit, N):
        return np.sqrt(np.sum(np.abs(y_real - y_fit)) / N)

    mses = {
        "Promedio Simple": custom_error(y, m_prom["fit"], N),
        "Línea con Tendencia": custom_error(y, m_lin["fit"], N),
        "Curva Cíclica Promedio": custom_error(y, m_cpr["fit"], N),
        "Cíclica con Tendencia": custom_error(y, m_ct["fit"], N),
        "Curva Geométrica": custom_error(y, m_geo["fit"], N),
        "Curva Exponencial": custom_error(y, m_exp["fit"], N),
        "Curva Potencial": custom_error(y, m_pot["fit"], N),
    }

    # ===== PESOS =====
    # SOLO si no recibes pesos desde afuera, calcula automáticos 0.5 / 0.3 / 0.2
    if pesos is None:
        sorted_models = sorted(mses.items(), key=lambda x: x[1])  # menor error primero
        top_3 = [model for model, _ in sorted_models[:3]]
        pesos = {model: 0.0 for model in mses.keys()}
        if len(top_3) >= 3:
            pesos[top_3[0]] = 0.5
            pesos[top_3[1]] = 0.3
            pesos[top_3[2]] = 0.2

    # ===== predicciones futuras =====
    last_year = int(years[-1])
    future_years = np.arange(last_year+1, last_year+horizon+1, dtype=int)
    tt_future = np.arange(len(y)+1, len(y)+horizon+1, dtype=float)

    preds = pd.DataFrame({
        "Año": future_years,
        "Periodo": tt_future.astype(int),
        "Soles": [np.nan]*horizon,
        "Promedio Simple": [m_prom["predict"](tt) for tt in tt_future],
        "Línea con Tendencia": [m_lin["predict"](tt) for tt in tt_future],
        "Curva Cíclica Promedio": [m_cpr["predict"](tt) for tt in tt_future],
        "Cíclica con Tendencia": [m_ct["predict"](tt) for tt in tt_future],
        "Curva Geométrica": [m_geo["predict"](tt) for tt in tt_future],
        "Curva Exponencial": [m_exp["predict"](tt) for tt in tt_future],
        "Curva Potencial": [m_pot["predict"](tt) for tt in tt_future],
    })

    tabla = pd.concat([hist, preds], ignore_index=True)

    # ===== Pre Ideal (mezcla ponderada) =====
    tabla["Pre Ideal"] = (
        pesos["Promedio Simple"]       * tabla["Promedio Simple"] +
        pesos["Línea con Tendencia"]   * tabla["Línea con Tendencia"] +
        pesos["Curva Cíclica Promedio"]* tabla["Curva Cíclica Promedio"] +
        pesos["Cíclica con Tendencia"] * tabla["Cíclica con Tendencia"] +
        pesos["Curva Geométrica"]      * tabla["Curva Geométrica"] +
        pesos["Curva Exponencial"]     * tabla["Curva Exponencial"] +
        pesos["Curva Potencial"]       * tabla["Curva Potencial"]
    )

    # ===== Ideal con bias =====
    ultimo_historico = y[-1]
    mask_futuro = tabla["Soles"].isna()
    if mask_futuro.any():
        primer_preideal_futuro = tabla.loc[mask_futuro, "Pre Ideal"].iloc[0]
        vfijo = primer_preideal_futuro - ultimo_historico
    else:
        vfijo = 0.0

    tabla["Ideal"] = tabla["Soles"].where(~mask_futuro, tabla["Pre Ideal"] + vfijo)

    return tabla, mses, pesos

def format_number(v):
    try:
        return f"{int(round(v)):,}".replace(",", ",")
    except:
        return v

# ========= carga de datos =========
with st.expander("1) Cargar datos", expanded=True):
    # st.write("Sube un Excel o pega la tabla. Puedes ingresar solo una columna con valores de 'Soles' o las clásicas dos columnas 'Año, Soles'.")
    st.write("Puedes ingresar solo una columna con valores")

    col1, col2 = st.columns([1, 2])  # Izquierda: carga / Derecha: vista previa

    # Estado compartido
    st.session_state.setdefault("_need_start_year", False)
    st.session_state.pop("df_input_ready", None)  # se regenerará en este paso

    df_input = None
    soles_series = None

    with col1:
        # uploaded = st.file_uploader("Excel opcional (.xlsx)", type=["xlsx"])
        uploaded = None

        if uploaded is not None:
            try:
                tmp = pd.read_excel(uploaded)
                tmp.columns = [str(c).strip() for c in tmp.columns]

                if {"Año", "Soles"}.issubset(tmp.columns):
                    df_input = tmp[["Año", "Soles"]].dropna()
                    st.session_state["_need_start_year"] = False
                elif tmp.shape[1] == 1:
                    soles_series = tmp.iloc[:, 0].dropna().astype(float).reset_index(drop=True)
                    st.session_state["_need_start_year"] = True
                else:
                    df_input = tmp.iloc[:, :2]
                    df_input.columns = ["Año", "Soles"]
                    df_input = df_input.dropna()
                    st.session_state["_need_start_year"] = False
            except Exception as e:
                st.error(f"No pude leer el Excel: {e}")
        else:
            text = st.text_area(
                "Pega tus datos",
                value="41683851922\n45846527653\n47200000000\n50123456789",
                height=300
            )
            if text.strip():
                try:
                    text = text.replace(",", "")  # 👈 elimina separadores de miles
                    tmp = pd.read_csv(io.StringIO(text), header=None)
                    if tmp.shape[1] >= 2:
                        df_input = tmp.iloc[:, :2]
                        df_input.columns = ["Año", "Soles"]
                        df_input = df_input.dropna()
                        st.session_state["_need_start_year"] = False
                    else:
                        soles_series = tmp.iloc[:, 0].dropna().astype(float).reset_index(drop=True)
                        st.session_state["_need_start_year"] = True
                except Exception:
                    try:
                        tmp = pd.read_csv(io.StringIO(text), header=None, sep=r"\s+")
                        if tmp.shape[1] == 1:
                            soles_series = tmp.iloc[:, 0].dropna().astype(float).reset_index(drop=True)
                            st.session_state["_need_start_year"] = True
                    except Exception as e:
                        st.error(f"No pude parsear el texto: {e}")

    with col2:
        st.write("Vista previa de los datos cargados:")

        # ⚙️ Caso 1: 1 sola columna (Soles) → pedir Año inicial y construir df_input aquí
        if st.session_state.get("_need_start_year", False) and (soles_series is not None):
            start_year = st.number_input("Año inicial", min_value=0, value=2012, step=1)
            years = np.arange(int(start_year), int(start_year) + len(soles_series), dtype=int)
            df_preview = pd.DataFrame({"Año": years, "Soles": soles_series.astype(float)})
            # st.caption("Vista previa construida desde 1 columna (ajusta el Año inicial si hace falta).")
            st.dataframe(df_preview.style.format({"Soles": format_number}), use_container_width=True)

            # Guarda el df listo para pasos siguientes
            st.session_state["df_input_ready"] = df_preview.copy()

        # ⚙️ Caso 2: ya hay 2 columnas → normalizar y mostrar
        elif df_input is not None:
            df_input = df_input.rename(columns=lambda c: str(c).strip())
            if not {"Año", "Soles"}.issubset(df_input.columns):
                st.warning("Asegúrate de que existan las columnas 'Año' y 'Soles'.")
            else:
                df_input = df_input[["Año", "Soles"]].dropna()
                df_input["Año"] = df_input["Año"].astype(int)
                df_input["Soles"] = df_input["Soles"].astype(float)
                st.dataframe(df_input, use_container_width=True)
                st.session_state["df_input_ready"] = df_input.copy()
        else:
            # Sin datos → muestra plantilla vacía con años por defecto
            default_df = pd.DataFrame({
                "Año": list(range(2012, 2012 + 14)),
                "Soles": [np.nan] * 14
            })
            st.dataframe(df_input.style.format({"Soles": format_number}), use_container_width=True)
            st.info("Pega una columna de Soles o sube un Excel.")

with st.expander("2) Parámetros de pronóstico", expanded=True):
    col1, col3 = st.columns(2)

    # Primero: Año inicial si el usuario ingresó 1 columna (Soles)
    default_start = 2012
    # start_year = col2.number_input("Año inicial (si ingresaste 1 columna)", min_value=0, value=default_start, step=1)

    # Si venía 1 columna, construimos df_input aquí
    if st.session_state.get("_need_start_year", False) and ("_soles_series" in st.session_state):
        s = st.session_state["_soles_series"]
        years = np.arange(int(start_year), int(start_year) + len(s), dtype=int)
        df_input = pd.DataFrame({"Año": years, "Soles": s.astype(float)})
        st.dataframe(df_input, use_container_width=True)

    # Calculamos default_N según df_input ACTUAL
    df_input = st.session_state.get("df_input_ready", None)

    if df_input is not None and len(df_input) > 0:
        default_N = len(df_input)
    else:
        default_N = 4  # fallback por si no hay datos aún


    # ⚠️ Definir N ANTES del horizonte
    N = col3.number_input("Periodo N p/ términos cíclicos", min_value=2, value=default_N, step=1)

    # Ahora que sabemos N, podemos calcular el límite de horizonte
    max_horizon = max(1, N // 2)

    # ✅ Aquí sí usamos max_horizon
    horizon = col1.number_input("Horizonte (periodos a pronosticar)", min_value=1, max_value=max_horizon, value=min(6, max_horizon), step=1)

with st.expander("3) Pesos de mezcla “Pre Ideal”", expanded=True):
    c1, c2, c3, c4 = st.columns(4)

    auto_pesos = st.checkbox("Asignar pesos automáticamente según MSE", value=True, key="auto_pesos_chk")

    # Recupera df_input ya armado
    df_input = st.session_state.get("df_input_ready", None)

    # Si es automático y tenemos datos suficientes, calcula y guarda los pesos
    if auto_pesos and df_input is not None and len(df_input) >= 3:
        try:
            # Ojo: N y horizon ya vienen del Expander 2
            _tabla_tmp, _mses_tmp, pesos_auto = calcular_todo(df_input, horizon=int(horizon), N=int(N), pesos=None)
            st.session_state["auto_pesos_vals"] = pesos_auto
        except Exception:
            # Si algo falla, al menos no revientes la UI
            st.session_state["auto_pesos_vals"] = None

    # Lee defaults (auto si existen, si no ceros)
    defaults = st.session_state.get("auto_pesos_vals", None) if auto_pesos else None
    def dget(name, fallback=0.0):
        return float(defaults.get(name, fallback)) if isinstance(defaults, dict) else float(fallback)

    # Inputs (si auto, solo muestran y están deshabilitados)
    w_prom = c1.number_input("Promedio simple", value=dget("Promedio Simple", 0.0), step=0.1, format="%.2f", disabled=auto_pesos, key="w_prom")
    w_lin  = c1.number_input("Línea con tendencia", value=dget("Línea con Tendencia", 0.0), step=0.1, format="%.2f", disabled=auto_pesos, key="w_lin")
    w_cpr  = c2.number_input("Curva cíclica (promedio)", value=dget("Curva Cíclica Promedio", 0.0), step=0.1, format="%.2f", disabled=auto_pesos, key="w_cpr")
    w_ct   = c2.number_input("Cíclica con tendencia", value=dget("Cíclica con Tendencia", 0.0), step=0.1, format="%.2f", disabled=auto_pesos, key="w_ct")
    w_geo  = c3.number_input("Curva geométrica", value=dget("Curva Geométrica", 0.0), step=0.1, format="%.2f", disabled=auto_pesos, key="w_geo")
    w_exp  = c3.number_input("Curva exponencial", value=dget("Curva Exponencial", 0.0), step=0.1, format="%.2f", disabled=auto_pesos, key="w_exp")
    w_pot  = c4.number_input("Curva potencial (cuadrática)", value=dget("Curva Potencial", 0.0), step=0.1, format="%.2f", disabled=auto_pesos, key="w_pot")

    # Construye 'pesos' para pasar a calcular_todo en el bloque de Resultados
    if auto_pesos and isinstance(defaults, dict):
        # usar los automáticos recién calculados
        pesos = defaults
    else:
        # usar los manuales ingresados por el usuario
        pesos = {
            "Promedio Simple": w_prom,
            "Línea con Tendencia": w_lin,
            "Curva Cíclica Promedio": w_cpr,
            "Cíclica con Tendencia": w_ct,
            "Curva Geométrica": w_geo,
            "Curva Exponencial": w_exp,
            "Curva Potencial": w_pot
        }

    # Deja los pesos en session_state para que el bloque de "Resultados" los use
    st.session_state["pesos_seleccionados"] = pesos

st.markdown("---")
st.subheader("Resultados")

df_input = st.session_state.get("df_input_ready", None)
pesos_ui = st.session_state.get("pesos_seleccionados", None)

if df_input is None or len(df_input) < 10:
    st.warning("Ingresa al menos 10 observaciones para ajustar los modelos.")
else:
    N = int(max(2, N))
    tabla, mses, pesos_usados = calcular_todo(df_input, horizon=horizon, N=N, pesos=pesos_ui)

    # Formatear tabla para mostrar en Streamlit
    tabla_fmt = tabla.copy()
    for col in tabla_fmt.columns[2:]:  # todas excepto Año y Periodo
        tabla_fmt[col] = tabla_fmt[col].map(format_number)

    # Altura dinámica basada en número de filas (30px por fila aprox.)
    num_filas = len(tabla_fmt)
    altura_dinamica = min(800, max(300, 38 * num_filas))  # Entre 300 y 800 píxeles

    st.dataframe(tabla_fmt, use_container_width=True, height=altura_dinamica)


    # MSEs
    mses_df = pd.DataFrame([mses]).T
    mses_df.columns = ["MSE"]
    best = mses_df["MSE"].idxmin()
    st.caption(f"Mejor MSE (histórico): **{best}**")

    # Colorear MSEs con escala de verdes (mejor = verde oscuro)
    styled_mses = mses_df.style.background_gradient(
        cmap="Greens", axis=0, subset=["MSE"]
    ).format({"MSE": lambda v: format_number(v)})

    st.dataframe(styled_mses, use_container_width=True)

    # Gráfico
    import matplotlib.ticker as mticker
    fig, ax = plt.subplots(figsize=(20, 5.5))

    # 1) Particiones histórico / futuro
    mask_futuro = tabla["Soles"].isna()

    xh = tabla.loc[~mask_futuro, "Año"].to_numpy()
    yh = tabla.loc[~mask_futuro, "Ideal"].to_numpy()

    xf = tabla.loc[mask_futuro, "Año"].to_numpy()
    yf = tabla.loc[mask_futuro, "Ideal"].to_numpy()

    # 2) Ideal histórico (línea continua)
    if len(xh) > 0:
        ax.plot(xh, yh, marker="o", linewidth=2, color="#FF2C2C", linestyle="-", label="Historico", zorder=10)

    # 3) Ideal pronóstico (línea punteada que ARRANCA en el último histórico)
    if len(xf) > 0 and len(xh) > 0:
        xfd = np.concatenate(([xh[-1]], xf))
        yfd = np.concatenate(([yh[-1]], yf))
        ax.plot(xfd, yfd, marker="o", linewidth=1.5, color="red", linestyle="--", label="Pronóstico", zorder=9)

    # Modelos (opcionales)
    ax.plot(tabla["Año"], tabla["Línea con Tendencia"], linewidth=1, linestyle="--", label="Línea con tendencia")
    ax.plot(tabla["Año"], tabla["Curva Exponencial"], linewidth=1, linestyle="--", label="Exponencial")
    ax.plot(tabla["Año"], tabla["Curva Potencial"], linewidth=1, linestyle="--", label="Potencial (cuadrática)")
    ax.plot(tabla["Año"], tabla["Cíclica con Tendencia"], linewidth=1, linestyle="--", label="Cíclica con tendencia")

    ax.set_xlabel("Año")
    ax.set_ylabel("Soles")
    ax.set_xticks(tabla["Año"].unique())

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", ".")))

    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)
    st.pyplot(fig)


    # Descargar CSV
    csv = tabla.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar tabla en CSV", data=csv, file_name="consolidado.csv", mime="text/csv")