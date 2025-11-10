import streamlit as st
import pandas as pd
import os
import re
from RandomForest import RandomForest
from Batallas import predecir_batalla, predecir_equipo

# === Rutas ===
POKEMON_CSV = r"C:\\CodigosVisual\\InteligenciaArtificial\\ProyectoFinal\\data\\pokemon_full.csv"
IMG_DIR = r"C:\\CodigosVisual\\InteligenciaArtificial\\ProyectoFinal\\data\\archive\\pokemon\\pokemon"

# === Cargar datos base ===
pokemon_df = pd.read_csv(POKEMON_CSV)

# === Función para buscar imágenes ===
def get_pokemon_image_path(pokedex_number):
    """Busca una imagen cuyo nombre sea exactamente el ID (O número en Pokédex) (ej: '6.png' o '6.jpg')."""
    pattern = re.compile(rf"^{pokedex_number}(\.png|\.jpg)$", re.IGNORECASE)
    for file in os.listdir(IMG_DIR):
        if pattern.match(file):
            return os.path.join(IMG_DIR, file)
    return None

# === Configuración de página ===
st.set_page_config(page_title="Pokémon Battle Predictor", page_icon=":D", layout="wide")
st.title("*** Pokémon Battle Predictor ***")
st.write("Selecciona tus Pokémon y predice la probabilidad de victoria")

# === Columnas por generación ===
generaciones = sorted(pokemon_df["generation"].unique())

col1, col2 = st.columns(2)

# === EQUIPO 1 ===
with col1:
    st.subheader("Equipo 1")
    gen1 = st.selectbox("Generación del Equipo 1", generaciones, key="gen1")
    poke_list_1 = pokemon_df[pokemon_df["generation"] == gen1]
    selected_1 = st.multiselect("Selecciona hasta 6 Pokémon", poke_list_1["name"], key="poke1")

    if len(selected_1) > 6:
        st.warning("⚠️ Solo puedes seleccionar hasta 6 Pokémon. Se tomarán los primeros 6.")
        selected_1 = selected_1[:6]

    imgs_col1 = st.container()
    equipo1_ids = []
    for name in selected_1:
        row = poke_list_1[poke_list_1["name"] == name].iloc[0]
        pid = row["pokedex_number"]
        equipo1_ids.append(pid)
        img_path = get_pokemon_image_path(pid)
        if img_path:
            imgs_col1.image(img_path, caption=f"{name} (ID {pid})", width=150)
        else:
            imgs_col1.warning(f"No se encontró imagen para {name} (ID {pid})")

# === EQUIPO 2 ===
with col2:
    st.subheader("Equipo 2")
    gen2 = st.selectbox("Generación del Equipo 2", generaciones, key="gen2")
    poke_list_2 = pokemon_df[pokemon_df["generation"] == gen2]
    selected_2 = st.multiselect("Selecciona hasta 6 Pokémon", poke_list_2["name"], key="poke2")

    if len(selected_1) > 6:
        st.warning("⚠️ Solo puedes seleccionar hasta 6 Pokémon. Se tomarán los primeros 6.")
        selected_1 = selected_1[:6]

    imgs_col2 = st.container()
    equipo2_ids = []
    for name in selected_2:
        row = poke_list_2[poke_list_2["name"] == name].iloc[0]
        pid = row["pokedex_number"]
        equipo2_ids.append(pid)
        img_path = get_pokemon_image_path(pid)
        if img_path:
            imgs_col2.image(img_path, caption=f"{name} (ID {pid})", width=150)
        else:
            imgs_col2.warning(f"No se encontró imagen para {name} (ID {pid})")

# === BOTÓN DE COMBATE ===
st.divider()
if st.button("🔥 Iniciar combate!"):
    if not equipo1_ids or not equipo2_ids:
        st.error("Selecciona al menos un Pokémon por equipo.")
    else:
        st.info("Calculando probabilidades con el modelo Random Forest... ⏳")

        try:
            prob = predecir_equipo(RandomForest.model, RandomForest.df, equipo1_ids, equipo2_ids)
            if prob is not None:
                st.success(f"Probabilidad de victoria del Equipo 1 sobre el Equipo 2: **{prob:.2%}**")
                st.progress(prob)
            else:
                st.warning("No se pudo calcular la probabilidad. Revisa los IDs seleccionados.")
        except Exception as e:
            st.error(f"Ocurrió un error durante la predicción: {e}")
