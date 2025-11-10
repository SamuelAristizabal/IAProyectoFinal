import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from RandomForest import RandomForest

rd = RandomForest()

def predecir_batalla(modelo, df, id1, id2, quiet = False):
    p1 = df[df["first_id"] == id1].iloc[0]
    p2 = df[df["second_id"] == id2].iloc[0]

    fila = pd.DataFrame([{
        "first_Attack": p1["first_Attack"],
        "first_Defense": p1["first_Defense"],
        "first_Sp. Atk": p1["first_Sp_Atk"],
        "first_Sp. Def": p1["first_Sp_Def"],
        "first_Speed": p1["first_Speed"],
        "first_HP": p1["first_HP"],
        "second_Attack": p2["second_Attack"],
        "second_Defense": p2["second_Defense"],
        "second_Sp. Atk": p2["second_Sp_Atk"],
        "second_Sp. Def": p2["second_Sp_Def"],
        "second_Speed": p2["second_Speed"],
        "second_HP": p2["second_HP"],
        "first_type_advantage": p1["first_type_advantage"],
        "second_type_advantage": p2["second_type_advantage"]
    }])

    fila.columns = fila.columns.str.replace('.', '', regex=False)
    fila.columns = fila.columns.str.replace(' ', '_', regex=False)

    proba = modelo.predict_proba(fila)[0, 1]
    if not quiet:
        print(f"Probabilidad de victoria de {p1['first_Name']} sobre {p2['second_Name']}: {proba:.2%}")
    return proba


def predecir_equipo(modelo, df, equipo1_ids, equipo2_ids):

    probabilidades = []

    for id1, id2 in itertools.product(equipo1_ids, equipo2_ids):
        try:
            p = predecir_batalla(modelo, df, id1, id2, quiet=True)
            probabilidades.append(p)
        except Exception as e:
            print(f"Error comparando {id1} vs {id2}: {e}")

    if not probabilidades:
        return None

    prom = sum(probabilidades) / len(probabilidades)
    print(f"Probabilidad de victoria del equipo 1 sobre el 2: {prom:.2%}")
    return prom

if __name__ == "__main__":
    equipo1 = [6, 9, 25, 3, 130, 131]     # Charizard, Blastoise, Pikachu, Venusaur, Gyarados, Lapras
    equipo2 = [94, 143, 65, 149, 6, 150]  # Gengar, Snorlax, Alakazam, Dragonite, Charizard, Mewtwo

    predecir_equipo(RandomForest.model, RandomForest.df, equipo1, equipo2)
