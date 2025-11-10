import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

class RandomForest():
    # Cargo mi dataset ya modificado
    df = pd.read_csv(r"C:\CodigosVisual\InteligenciaArtificial\ProyectoFinal\data\pokemon_battles_full.csv")

    # Definir target, variable binaria, si gana el primer pokemon es 1 si no es 0
    df["first_wins"] = (df["Winner"] == df["first_id"]).astype(int)

    # Columnas que sí vamos a usar como features
    features = [
        "first_Attack", "first_Defense", "first_Sp_Atk", "first_Sp_Def", "first_Speed", "first_HP",
        "second_Attack", "second_Defense", "second_Sp_Atk", "second_Sp_Def", "second_Speed", "second_HP",
        "first_type_advantage", "second_type_advantage"
    ]

    # Normalizar nombres de columnas
    df.columns = df.columns.str.replace('.', '', regex=False)
    df.columns = df.columns.str.replace(' ', '_', regex=False)

    # Aquí ya definimos cuál es X y cual es y
    X = df[features]
    y = df["first_wins"]

    # Divide en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


    ## Entrenar modelo
    model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1] #Probabilidad de que gane el primer pokemon

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))

