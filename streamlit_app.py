
import streamlit as st
import pandas as pd
import joblib
import json
import os
import warnings
import numpy as np

# Filtrar advertencias de versiones (son solo warnings, no errores)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)

# Configuración de rutas - USANDO LA RUTA DE KAGGLEHUB QUE FUNCIONÓ
KAGGLE_DATA_PATH = r'C:\Users\usuario\.cache\kagglehub\datasets\rounakbanik\pokemon\versions\1'
CSV_PATH = os.path.join(KAGGLE_DATA_PATH, 'Pokemon.csv')

# Verificar que los archivos existen antes de cargarlos
if not st.session_state.get('files_checked'):
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

    st.session_state.files_checked = True

# Cargar modelo y metadata
if 'model' not in st.session_state:
    try:
        with st.spinner('Cargando modelo...'):
            st.session_state.model = joblib.load('artifacts/pokemon_rf_model.joblib')
        with st.spinner('Cargando metadata...'):
            with open('artifacts/meta.json','r') as f:
                st.session_state.meta = json.load(f)
        st.success("✅ Modelo y metadata cargados correctamente")
    except Exception as e:
        st.error(f"❌ Error cargando modelo o metadata: {e}")
        st.stop()

# Cargar datos Pokémon
if 'pokemon_df' not in st.session_state:
    try:
        with st.spinner('Cargando dataset Pokémon...'):
            st.session_state.pokemon_df = pd.read_csv(CSV_PATH)
            st.session_state.pokemon_df.columns = [c.strip() for c in st.session_state.pokemon_df.columns]

        # Verificar columnas necesarias
        usable = ['name'] + st.session_state.meta['numeric_stats'] + st.session_state.meta['against_cols']
        missing_cols = [c for c in usable if c not in st.session_state.pokemon_df.columns]
        if missing_cols:
            st.error(f"❌ Columnas faltantes en el CSV: {missing_cols}")
            st.write("📊 Columnas disponibles en el dataset:")
            for col in st.session_state.pokemon_df.columns:
                st.write(f"   - {col}")
            st.stop()

        st.session_state.pokemon_df.set_index('name', inplace=True)
        st.success(f"✅ Dataset cargado: {len(st.session_state.pokemon_df)} Pokémon disponibles")

    except Exception as e:
        st.error(f"❌ Error procesando el dataset: {e}")
        st.stop()

# Acceso rápido a las variables
model = st.session_state.model
meta = st.session_state.meta
pokemon_df = st.session_state.pokemon_df

# Función para calcular importancia de características
def explain_prediction(model, features_df, feature_names):
    """Explica la predicción mostrando las características más importantes"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = list(zip(feature_names, importances, features_df.iloc[0].values))
        feature_importance.sort(key=lambda x: abs(x[1] * x[2]), reverse=True)
        return feature_importance[:5]  # Top 5 características más influyentes
    return []

# Función para analizar ventajas de equipo - CORREGIDA
def analyze_team_advantages(teamA, teamB, pokemon_df, meta):
    """Analiza las ventajas generales de un equipo sobre otro"""
    advantages = {
        'stats_avg': {'teamA': {}, 'teamB': {}},
        'type_coverage': {'teamA': set(), 'teamB': set()},
        'total_stats': {'teamA': 0, 'teamB': 0}
    }

    # Calcular promedios de stats por equipo - CORREGIDO: asegurar que son números
    valid_teamA = [p for p in teamA if p in pokemon_df.index]
    valid_teamB = [p for p in teamB if p in pokemon_df.index]

    for stat in meta['numeric_stats']:
        if valid_teamA:
            teamA_stats = [float(pokemon_df.loc[p][stat]) for p in valid_teamA]
            advantages['stats_avg']['teamA'][stat] = np.mean(teamA_stats)
        else:
            advantages['stats_avg']['teamA'][stat] = 0

        if valid_teamB:
            teamB_stats = [float(pokemon_df.loc[p][stat]) for p in valid_teamB]
            advantages['stats_avg']['teamB'][stat] = np.mean(teamB_stats)
        else:
            advantages['stats_avg']['teamB'][stat] = 0

    # Calcular cobertura de tipos
    for team in ['teamA', 'teamB']:
        team_list = valid_teamA if team == 'teamA' else valid_teamB
        for p in team_list:
            p_data = pokemon_df.loc[p]
            advantages['type_coverage'][team].add(str(p_data['type1']))
            if pd.notna(p_data.get('type2')):
                advantages['type_coverage'][team].add(str(p_data['type2']))

    # Calcular stats totales - CORREGIDO: asegurar que son números
    if valid_teamA:
        advantages['total_stats']['teamA'] = sum([
            sum([float(pokemon_df.loc[p][stat]) for stat in meta['numeric_stats']]) 
            for p in valid_teamA
        ])
    else:
        advantages['total_stats']['teamA'] = 0

    if valid_teamB:
        advantages['total_stats']['teamB'] = sum([
            sum([float(pokemon_df.loc[p][stat]) for stat in meta['numeric_stats']]) 
            for p in valid_teamB
        ])
    else:
        advantages['total_stats']['teamB'] = 0

    return advantages

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
                X.append(float(p1_row[stn]) - float(p2_row[stn]))  # CORREGIDO: asegurar floats
            X_dict = {f'{stn}_diff': X[i] for i, stn in enumerate(meta['numeric_stats'])}

            # ventajas tipo
            def type_adv_local(pa, pb):
                attacker_types = [str(t) for t in [pa['type1'], pa.get('type2')] if pd.notna(t)]
                defender_types = [str(t) for t in [pb['type1'], pb.get('type2')] if pd.notna(t)]
                s = 0.0
                count = 0
                for atk in attacker_types:
                    for d in defender_types:
                        col = f'against_{d.lower()}'
                        if col in pb.index:
                            s += float(pb[col])  # CORREGIDO: asegurar float
                        else:
                            s += 1.0
                        count += 1
                return s / count if count > 0 else 1.0

            X_dict['adv_p1_on_p2'] = type_adv_local(p1_row, p2_row)
            X_dict['adv_p2_on_p1'] = type_adv_local(p2_row, p1_row)
            Xdf = pd.DataFrame([X_dict])
            prob = model.predict_proba(Xdf)[0][1]

            # ANÁLISIS DE LA PREDICCIÓN
            feature_names = Xdf.columns.tolist()
            top_features = explain_prediction(model, Xdf, feature_names)

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
            st.progress(float(prob))

            # ANÁLISIS DETALLADO
            with st.expander("🔍 Análisis detallado del enfrentamiento"):
                st.subheader("📈 Factores clave en la decisión:")

                if top_features:
                    st.write("**Características más influyentes:**")
                    for feature, importance, value in top_features:
                        effect = "FAVORABLE" if (importance * value) > 0 else "DESFAVORABLE"
                        color = "🟢" if effect == "FAVORABLE" else "🔴"
                        st.write(f"{color} **{feature}**: {value:.2f} (Impacto: {importance*100:.1f}%)")

                # Comparación de stats
                st.subheader("📊 Comparación de Estadísticas:")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**{p1}**")
                    for stat in ['hp', 'attack', 'defense', 'speed']:
                        val1 = float(p1_row[stat])
                        val2 = float(p2_row[stat])
                        diff = val1 - val2
                        arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
                        st.write(f"{stat.capitalize()}: {val1} ({arrow} {abs(diff):.0f})")
                with col2:
                    st.write(f"**{p2}**")
                    for stat in ['hp', 'attack', 'defense', 'speed']:
                        val1 = float(p2_row[stat])
                        val2 = float(p1_row[stat])
                        diff = val1 - val2
                        arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
                        st.write(f"{stat.capitalize()}: {val1} ({arrow} {abs(diff):.0f})")

                # Ventajas de tipo
                st.subheader("🎯 Ventajas de Tipo:")
                st.write(f"**{p1} → {p2}**: Multiplicador {X_dict['adv_p1_on_p2']:.2f}x")
                st.write(f"**{p2} → {p1}**: Multiplicador {X_dict['adv_p2_on_p1']:.2f}x")

                if X_dict['adv_p1_on_p2'] > X_dict['adv_p2_on_p1']:
                    st.info(f"**{p1} tiene mejor ventaja de tipo**")
                elif X_dict['adv_p2_on_p1'] > X_dict['adv_p1_on_p2']:
                    st.info(f"**{p2} tiene mejor ventaja de tipo**")
                else:
                    st.info("**Ventajas de tipo equilibradas**")

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
                        row[f'{stn}_diff'] = float(p1[stn]) - float(p2[stn])  # CORREGIDO: asegurar floats
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
                                    s += float(defender[col])  # CORREGIDO: asegurar float
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
                all_predictions = []

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
                            all_predictions.append({
                                'pokemonA': a,
                                'pokemonB': b,
                                'probA': prob,
                                'probB': 1-prob
                            })

                if invalid_names:
                    st.warning(f'⚠️ Nombres no válidos ignorados: {list(set(invalid_names))}')

                if total == 0:
                    st.error('❌ No se pudo evaluar ningún enfrentamiento. Revisa los nombres.')
                else:
                    probA = (scoresA/total)*100
                    probB = (scoresB/total)*100

                    # ANÁLISIS DE VENTAJAS DEL EQUIPO
                    team_advantages = analyze_team_advantages(teamA, teamB, pokemon_df, meta)

                    st.markdown("---")
                    st.subheader("📊 Resultados de la Batalla")

                    # Mostrar resultados en métricas
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Equipo A", f"{probA:.1f}%", delta="Ganador" if probA > probB else None)
                    with col2:
                        st.metric("Equipo B", f"{probB:.1f}%", delta="Ganador" if probB > probA else None)

                    # Barra de progreso
                    st.progress(float(probA/100))

                    # ANÁLISIS DETALLADO DE EQUIPOS
                    with st.expander("🔍 Análisis detallado de los equipos"):

                        # Estadísticas promedio
                        st.subheader("📈 Estadísticas Promedio por Equipo")
                        stats_data = []
                        for stat in meta['numeric_stats']:
                            stats_data.append({
                                'Stat': stat.upper(),
                                'Equipo A': float(team_advantages['stats_avg']['teamA'][stat]),
                                'Equipo B': float(team_advantages['stats_avg']['teamB'][stat])
                            })

                        stats_df = pd.DataFrame(stats_data)
                        # CORREGIDO: usar width en lugar de use_container_width
                        st.dataframe(stats_df.style.highlight_max(axis=1, color='lightgreen'), width='stretch')

                        # Cobertura de tipos
                        st.subheader("🎯 Cobertura de Tipos")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Equipo A tipos:**")
                            for tipo in team_advantages['type_coverage']['teamA']:
                                st.write(f"• {tipo}")
                        with col2:
                            st.write("**Equipo B tipos:**")
                            for tipo in team_advantages['type_coverage']['teamB']:
                                st.write(f"• {tipo}")

                        # Mejores matchups
                        st.subheader("🔥 Mejores Matchups Individuales")
                        all_predictions.sort(key=lambda x: x['probA'], reverse=True)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Mejores para Equipo A:**")
                            for matchup in all_predictions[:3]:
                                st.write(f"• {matchup['pokemonA']} vs {matchup['pokemonB']}: {matchup['probA']*100:.1f}%")
                        with col2:
                            st.write("**Mejores para Equipo B:**")
                            for matchup in sorted(all_predictions, key=lambda x: x['probB'], reverse=True)[:3]:
                                st.write(f"• {matchup['pokemonB']} vs {matchup['pokemonA']}: {matchup['probB']*100:.1f}%")

                        # Razones principales de la victoria
                        st.subheader("💡 Factores Decisivos")
                        if probA > probB:
                            winning_team = "Equipo A"
                            # Analizar por qué gana el equipo A
                            better_stats = []
                            for stat in meta['numeric_stats']:
                                if team_advantages['stats_avg']['teamA'][stat] > team_advantages['stats_avg']['teamB'][stat]:
                                    better_stats.append(stat)

                            if better_stats:
                                st.write(f"**{winning_team} domina en:** {', '.join(better_stats)}")

                            st.write(f"**{winning_team} tiene mejor cobertura de tipos** ({len(team_advantages['type_coverage']['teamA'])} tipos vs {len(team_advantages['type_coverage']['teamB'])} tipos)")

                        else:
                            winning_team = "Equipo B"
                            better_stats = []
                            for stat in meta['numeric_stats']:
                                if team_advantages['stats_avg']['teamB'][stat] > team_advantages['stats_avg']['teamA'][stat]:
                                    better_stats.append(stat)

                            if better_stats:
                                st.write(f"**{winning_team} domina en:** {', '.join(better_stats)}")

                            st.write(f"**{winning_team} tiene mejor cobertura de tipos** ({len(team_advantages['type_coverage']['teamB'])} tipos vs {len(team_advantages['type_coverage']['teamA'])} tipos)")

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
