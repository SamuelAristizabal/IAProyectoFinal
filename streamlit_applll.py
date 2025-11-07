
import streamlit as st
import pandas as pd
import joblib
import json
import os
import warnings

# Filtrar advertencias de versiones (son solo warnings, no errores)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Configuración de rutas - USANDO LA RUTA DE KAGGLEHUB QUE FUNCIONÓ
KAGGLE_DATA_PATH = r'C:\Users\usuario\.cache\kagglehub\datasets\rounakbanik\pokemon\versions\1'
CSV_PATH = os.path.join(KAGGLE_DATA_PATH, 'Pokemon.csv')

# Verificar que los archivos existen antes de cargarlos
st.write("🔍 Verificando archivos necesarios...")

if not os.path.exists('artifacts/pokemon_rf_model.joblib'):
    st.error("❌ No se encontró el modelo en 'artifacts/pokemon_rf_model.joblib'")
    st.stop()
else:
    st.success("✅ Modelo encontrado")

if not os.path.exists('artifacts/meta.json'):
    st.error("❌ No se encontró el metadata en 'artifacts/meta.json'")
    st.stop()
else:
    st.success("✅ Metadata encontrada")

if not os.path.exists(CSV_PATH):
    st.error(f"❌ No se encontró el dataset Pokémon en: {CSV_PATH}")
    st.write("📁 Archivos en la ruta de kagglehub:")
    if os.path.exists(KAGGLE_DATA_PATH):
        try:
            files = os.listdir(KAGGLE_DATA_PATH)
            for file in files:
                st.write(f"   - {file}")
        except:
            st.write("   No se pudo listar archivos")
    st.stop()
else:
    st.success("✅ Dataset Pokémon encontrado")

# Cargar modelo y metadata
try:
    with st.spinner('Cargando modelo...'):
        model = joblib.load('artifacts/pokemon_rf_model.joblib')
    with st.spinner('Cargando metadata...'):
        with open('artifacts/meta.json','r') as f:
            meta = json.load(f)
    st.success("✅ Modelo y metadata cargados correctamente")
except Exception as e:
    st.error(f"❌ Error cargando modelo o metadata: {e}")
    st.stop()

# Cargar datos Pokémon
try:
    with st.spinner('Cargando dataset Pokémon...'):
        pokemon_df = pd.read_csv(CSV_PATH)
        pokemon_df.columns = [c.strip() for c in pokemon_df.columns]

    # Verificar columnas necesarias
    usable = ['name'] + meta['numeric_stats'] + meta['against_cols']
    missing_cols = [c for c in usable if c not in pokemon_df.columns]
    if missing_cols:
        st.error(f"❌ Columnas faltantes en el CSV: {missing_cols}")
        st.write("📊 Columnas disponibles en el dataset:")
        for col in pokemon_df.columns:
            st.write(f"   - {col}")
        st.stop()

    pokemon_df.set_index('name', inplace=True)
    st.success(f"✅ Dataset cargado: {len(pokemon_df)} Pokémon disponibles")

except Exception as e:
    st.error(f"❌ Error procesando el dataset: {e}")
    st.stop()

# INTERFAZ DE LA APLICACIÓN
st.title('⚔️ Pokémon Battle Predictor')
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    mode = st.radio('Modo de Batalla', ['1v1', 'Equipo vs Equipo'])

if mode == '1v1':
    st.subheader("🏅 Modo 1 vs 1")

    col1, col2 = st.columns(2)
    with col1:
        p1 = st.selectbox('Pokémon 1', pokemon_df.index.tolist(), index=0)
    with col2:
        p2 = st.selectbox('Pokémon 2', pokemon_df.index.tolist(), index=1)

    if st.button('🎯 Predecir Batalla 1v1', type='primary'):
        try:
            # calcular features localmente
            p1_row = pokemon_df.loc[p1]
            p2_row = pokemon_df.loc[p2]

            # construir las mismas features que en entrenamiento
            X = []
            for stn in meta['numeric_stats']:
                X.append(float(p1_row[stn] - p2_row[stn]))
            X_dict = {f'{stn}_diff':X[i] for i,stn in enumerate(meta['numeric_stats'])}

            # ventajas tipo
            def type_adv_local(pa,pb):
                attacker_types = [t for t in [pa['type1'], pa.get('type2')] if pd.notna(t)]
                defender_types = [t for t in [pb['type1'], pb.get('type2')] if pd.notna(t)]
                s = 0.0
                count = 0
                for atk in attacker_types:
                    for d in defender_types:
                        col = f'against_{str(d).lower()}'
                        if col in pb.index:
                            s += float(pb[col])
                        else:
                            s += 1.0
                        count += 1
                return s/count if count>0 else 1.0

            X_dict['adv_p1_on_p2'] = type_adv_local(p1_row, p2_row)
            X_dict['adv_p2_on_p1'] = type_adv_local(p2_row, p1_row)
            Xdf = pd.DataFrame([X_dict])
            prob = model.predict_proba(Xdf)[0][1]

            # Mostrar resultados
            st.markdown("---")
            st.subheader("📊 Resultados de la Batalla")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"{p1}", f"{prob*100:.1f}%", delta="Ganador" if prob > 0.5 else None)
            with col2:
                st.metric("VS", "")
            with col3:
                st.metric(f"{p2}", f"{(1-prob)*100:.1f}%", delta="Ganador" if prob < 0.5 else None)

            # Barra de progreso
            st.progress(prob)

            # Mostrar información adicional
            with st.expander("🔍 Ver detalles del enfrentamiento"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**{p1}**")
                    st.write(f"Tipo: {p1_row['type1']}" + (f", {p1_row['type2']}" if pd.notna(p1_row.get('type2')) else ""))
                    st.write(f"HP: {p1_row['hp']} | Ataque: {p1_row['attack']}")
                    st.write(f"Defensa: {p1_row['defense']} | Velocidad: {p1_row['speed']}")

                with col2:
                    st.write(f"**{p2}**")
                    st.write(f"Tipo: {p2_row['type1']}" + (f", {p2_row['type2']}" if pd.notna(p2_row.get('type2')) else ""))
                    st.write(f"HP: {p2_row['hp']} | Ataque: {p2_row['attack']}")
                    st.write(f"Defensa: {p2_row['defense']} | Velocidad: {p2_row['speed']}")

        except Exception as e:
            st.error(f"❌ Error en la predicción: {e}")

else:
    st.subheader("👥 Modo Equipo vs Equipo")
    st.write("Ingresa los nombres de los Pokémon separados por comas:")

    colA, colB = st.columns(2)
    with colA:
        textA = st.text_area('Equipo A', value='Charizard, Blastoise, Pikachu', height=100)
        st.caption("Ejemplo: Charizard, Blastoise, Pikachu")
    with colB:
        textB = st.text_area('Equipo B', value='Venusaur, Gyarados, Raichu', height=100)
        st.caption("Ejemplo: Venusaur, Gyarados, Raichu")

    if st.button('🎯 Predecir Batalla de Equipos', type='primary'):
        teamA = [t.strip() for t in textA.split(',') if t.strip()]
        teamB = [t.strip() for t in textB.split(',') if t.strip()]

        if not teamA or not teamB:
            st.error('❌ Ambos equipos deben contener al menos 1 Pokémon')
        else:
            try:
                # usar el enfoque de sumatoria de probabilidades
                def make_features_local(n1,n2):
                    p1 = pokemon_df.loc[n1]
                    p2 = pokemon_df.loc[n2]
                    row = {}
                    for stn in meta['numeric_stats']:
                        row[f'{stn}_diff'] = float(p1[stn] - p2[stn])
                    # type adv
                    def type_adv_simple(attacker, defender):
                        s = 0.0
                        count = 0
                        for atk in [attacker.get('type1'), attacker.get('type2')]:
                            for d in [defender.get('type1'), defender.get('type2')]:
                                if pd.isna(atk) or pd.isna(d):
                                    continue
                                col = f'against_{str(d).lower()}'
                                if col in defender.index:
                                    s += float(defender[col])
                                else:
                                    s += 1.0
                                count += 1
                        return s/(count if count>0 else 1)
                    row['adv_p1_on_p2'] = type_adv_simple(p1, p2)
                    row['adv_p2_on_p1'] = type_adv_simple(p2, p1)
                    return pd.DataFrame([row])

                scoresA = 0.0
                scoresB = 0.0
                total = 0
                invalid_names = []

                with st.spinner('Calculando enfrentamientos...'):
                    for a in teamA:
                        for b in teamB:
                            if a not in pokemon_df.index or b not in pokemon_df.index:
                                invalid_names.extend([name for name in [a, b] if name not in pokemon_df.index])
                                continue
                            Xdf = make_features_local(a,b)
                            prob = model.predict_proba(Xdf)[0][1]
                            scoresA += prob
                            scoresB += (1-prob)
                            total += 1

                if invalid_names:
                    st.warning(f'⚠️ Nombres no válidos ignorados: {list(set(invalid_names))}')

                if total == 0:
                    st.error('❌ No se pudo evaluar ningún enfrentamiento. Revisa los nombres.')
                else:
                    probA = (scoresA/total)*100
                    probB = (scoresB/total)*100

                    st.markdown("---")
                    st.subheader("📊 Resultados de la Batalla")

                    # Mostrar resultados en métricas
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Equipo A", f"{probA:.1f}%", delta="Ganador" if probA > probB else None)
                    with col2:
                        st.metric("Equipo B", f"{probB:.1f}%", delta="Ganador" if probB > probA else None)

                    # Barra de progreso
                    st.progress(probA/100)

                    # Mostrar equipos
                    with st.expander("👥 Ver composición de equipos"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Equipo A:**")
                            for pokemon in teamA:
                                status = "✅" if pokemon in pokemon_df.index else "❌"
                                st.write(f"{status} {pokemon}")
                        with col2:
                            st.write("**Equipo B:**")
                            for pokemon in teamB:
                                status = "✅" if pokemon in pokemon_df.index else "❌"
                                st.write(f"{status} {pokemon}")

            except Exception as e:
                st.error(f"❌ Error en la predicción de equipos: {e}")

# Footer
st.markdown("---")
st.caption("Pokémon Battle Predictor - Usando Machine Learning para predecir batallas Pokémon")
