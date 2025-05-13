import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def resolver_placa(n, top, bottom, left, right, tol=1e-4, max_iter=10000):
    # Inicializar la matriz de temperaturas
    T = np.zeros((n+2, n+2))
    T[0, :] = top
    T[-1, :] = bottom
    T[:, 0] = left
    T[:, -1] = right

    # Generar ecuaciones de diferencias finitas
    ecuaciones = []
    for i in range(1, n+1):
        for j in range(1, n+1):
            ecuaciones.append(f"T[{i},{j}] = 0.25 * (T[{i+1},{j}] + T[{i-1},{j}] + T[{i},{j+1}] + T[{i},{j-1}])")

    # Resolver el sistema de ecuaciones utilizando diferencias finitas
    for _ in range(max_iter):
        T_old = T.copy()
        for i in range(1, n+1):
            for j in range(1, n+1):
                T[i, j] = 0.25 * (T[i+1, j] + T[i-1, j] + T[i, j+1] + T[i, j-1])
        if np.max(np.abs(T - T_old)) < tol:
            break
    return T, ecuaciones

# Configuración de Streamlit
st.set_page_config(layout="wide", page_title="Simulador EDP Elíptica")
st.title("🌡️ Simulación de Calor — Ecuaciones Diferenciales Parciales Elípticas")

# Mostrar las ecuaciones con explicación
st.markdown("""

""")

# Entradas de usuario
col1, col2 = st.columns(2)

with col1:
    magnitud = st.selectbox("🔢 Selecciona la magnitud a simular:", 
                            ["Temperatura (°C)", "Potencial Eléctrico (V)", "Presión (Pa)", "Concentración (mol/L)"])

    n = st.slider("📐 Tamaño de la placa (n x n)", min_value=1, max_value=50, value=3)
    top = st.number_input("T en el borde superior", value=100.0)
    bottom = st.number_input("T en el borde inferior", value=100.0)
    left = st.number_input("T en el borde izquierdo", value=0.0)
    right = st.number_input("T en el borde derecho", value=0.0)

with col2:
    st.markdown("### 📘 Ecuaciones generadas por diferencias finitas")
    T, ecuaciones = resolver_placa(n, top, bottom, left, right)
    st.code("\n".join(ecuaciones), language="text")

# Mostrar la matriz resultante en formato tabla
st.markdown("---")
st.markdown("### 📝 Tabla de Resultados (Matriz de Magnitudes)")

tabla_resultados = [[f"{T[i+1, j+1]:.2f}" for j in range(n)] for i in range(n)]
df_resultados = pd.DataFrame(tabla_resultados, columns=[f"Columna {i+1}" for i in range(n)])
st.table(df_resultados)

# Mostrar gráficos con y sin valores numéricos
col3, col4 = st.columns(2)

with col3:
    st.markdown("### 🎨 Mapa con valores numéricos")
    fig1, ax1 = plt.subplots()
    data = T[1:-1, 1:-1]
    cmap = plt.cm.get_cmap('coolwarm')
    im = ax1.imshow(data, cmap=cmap, origin='upper')
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax1.text(j, i, f"{data[i,j]:.1f}", ha='center', va='center', color='black', fontsize=8)
    plt.colorbar(im, ax=ax1, shrink=0.8, label="T")
    ax1.set_title("Distribución de T con valores")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    st.pyplot(fig1)

with col4:
    st.markdown("### 🌈 Solo colores (sin números)")
    fig2, ax2 = plt.subplots()
    im2 = ax2.imshow(data, cmap=cmap, origin='upper')
    plt.colorbar(im2, ax=ax2, shrink=0.8, label="T")
    ax2.set_title("Distribución de T")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    st.pyplot(fig2)