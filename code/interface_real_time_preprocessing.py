import shap
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import mutual_info_classif

import paths
import constants as const

# Carico il training set per informazioni sulle feature
train_set = pd.read_csv(paths.SMOTE20_PREP_TRAIN_PATH)

X_train = train_set.drop(columns=["isFraud"])
y_train = train_set["isFraud"]

# Carico i preprocessori salvati
print("Loading preprocessors...")
imputer = joblib.load(paths.IMPUTER_PATH)
scaler = joblib.load(paths.SCALER_PATH)
var_thresh = joblib.load(paths.VAR_THRESH_PATH)
selected_features = joblib.load(paths.SELECTED_FEATURES_PATH)

# Maschera per colonne discrete
discrete_mask = [np.issubdtype(X_train[c].dtype, np.integer) for c in selected_features]


def mi_score(X, y, discrete_mask=discrete_mask, random_state=const.RANDOM_STATE):
    return mutual_info_classif(X, y, discrete_features=discrete_mask, random_state=random_state)

selector = joblib.load(paths.SELECTOR_PATH)


def preprocess_user_input(df):
    # --- Rimuovo colonne non utilizzate dal modello ---
    for col in ["TransactionID_x", "TransactionID_y"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # --- Feature temporali ---
    df["TransactionDT_days"] = (df["TransactionDT"] / (24*60*60)).astype(int)
    df["hour"] = (df["TransactionDT"] // 3600) % 24
    df["dayofweek"] = (df["TransactionDT"] // (24*3600)) % 7

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    # --- Imputazione ---
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    df[num_cols] = imputer.transform(df[num_cols])

    # --- Scaling ---
    df[num_cols] = scaler.transform(df[num_cols])

    # --- Feature selection ---
    df_var = var_thresh.transform(df)
    df_sel = selector.transform(df_var)

    df_final = pd.DataFrame(df_sel, columns=selected_features)
    return df_final



# Carico i modelli salvati
print("Loading models...")
model_rf = joblib.load(paths.RF_PATH)
model_dt = joblib.load(paths.DT_PATH)
model_nb = joblib.load(paths.NB_PATH)
model_knn = joblib.load(paths.KNN_PATH)
model_ada = joblib.load(paths.ADA_PATH)
model_xgb = joblib.load(paths.XGB_PATH)


st.title("Credit Card Fraud Detection")

st.subheader("Inserisci manualmente una transazione")

# Percorso del CSV
file_csv = paths.RAW_TEST_PATH

# Apri il file e leggi le righe
with open(file_csv, "r") as f:
    # La prima riga è l'header
    header = f.readline().strip().split(",")
    # La seconda riga sono i valori
    values = f.readline().strip().split(",")

row_dict = {}
for col, val in zip(header, values):
    try:
        row_dict[col] = float(val)
    except ValueError:
        row_dict[col] = val

user_input = pd.DataFrame([row_dict])

if "isFraud" in user_input.columns:
    user_input = user_input.drop(columns=["isFraud"])

user_input_dict = {}
for col in user_input.columns:
    val = user_input[col].iloc[0]
    if np.issubdtype(user_input[col].dtype, np.number):
        user_input_dict[col] = st.number_input(col, value=float(val))
    else:
        user_input_dict[col] = st.text_input(col, value=str(val))

# Ricostruisco il DataFrame dall'input modificabile
user_input_df = pd.DataFrame([user_input_dict])


if st.button("Predict"):
    df_input = preprocess_user_input(user_input_df)

   # Lista dei modelli con nomi
    models = {
        "Random Forest": model_rf,
        "XGBoost": model_xgb,
        "Decision Tree": model_dt,
        "Gaussian NB": model_nb,
        "KNN": model_knn,
        "AdaBoost": model_ada
    }

    # Predizioni
    results = {}
    for name, model in models.items():
        pred = model.predict(df_input)[0]
        results[name] = "Fraudulent" if pred else "Legitimate"

    # Tabella dei risultati colorata
    df_results = pd.DataFrame(list(results.items()), columns=["Model", "Prediction"])
    def color_pred(val):
        return 'color: red' if val == "Fraudulent" else 'color: green'
    st.subheader("Predizioni dei modelli:")
    st.dataframe(df_results.style.map(color_pred, subset=["Prediction"]))


    # Voto di maggioranza
    fraud_votes = list(results.values()).count("Fraudulent")
    legit_votes = list(results.values()).count("Legitimate")
    st.subheader("Voto finale dei modelli:")
    if fraud_votes > legit_votes:
        st.error(f"Transazione segnalata come Fraudulent da {fraud_votes}/{len(models)} modelli!")
    else:
        st.success(f"Transazione segnalata come Legitimate da {legit_votes}/{len(models)} modelli!")

    # --- SHAP explanations ---
    st.subheader("Explanations (per singolo modello)")

    for name, model in models.items():
        st.subheader(f"Spiegazione per {name}")
        if name in ["Random Forest", "Decision Tree", "XGBoost"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df_input)

            # Se shape è (1, n_features, 2) -> prendi esempio 0 e classe 1
            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_vals = shap_values[0, :, 1]  # primo esempio, tutte le feature, classe 1
            elif isinstance(shap_values, list):
                shap_vals = shap_values[1][0]  # vecchia lista, primo esempio classe 1
            else:
                shap_vals = shap_values[0]  # caso array 2D semplice

            # Base value
            base_val = explainer.expected_value
            if isinstance(base_val, (list, np.ndarray)):
                base_val_array = np.array(base_val).ravel()
                if len(base_val_array) > 1:
                    base_val = float(base_val_array[1])  # classe 1
                else:
                    base_val = float(base_val_array[0])
            else:
                base_val = float(base_val)

            # Waterfall plot
            fig, ax = plt.subplots()  
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_vals,
                    base_values=base_val,
                    data=df_input_aligned.iloc[0] if name == "AdaBoost" else df_input.iloc[0],
                    feature_names=df_input_aligned.columns if name == "AdaBoost" else df_input.columns
                ),
                show=False
            )
            st.pyplot(fig)

        elif name == "AdaBoost":
            # Background set: usa un piccolo campione del training set
            background = train_set.sample(50, random_state=42)
            background = background[model_ada.feature_names_in_] 

            df_input_aligned = df_input[model_ada.feature_names_in_]

            explainer = shap.KernelExplainer(model_ada.predict_proba, background.sample(50, random_state=42))
            shap_values = explainer.shap_values(df_input_aligned, nsamples=100)

            # Se shape è (1, n_features, 2) -> prendi esempio 0 e classe 1
            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_vals = shap_values[0, :, 1]  # primo esempio, tutte le feature, classe 1
            elif isinstance(shap_values, list):
                shap_vals = shap_values[1][0]  # vecchia lista, primo esempio classe 1
            else:
                shap_vals = shap_values[0]  # caso array 2D semplice

            # Base value
            base_val = explainer.expected_value
            if isinstance(base_val, (list, np.ndarray)):
                base_val_array = np.array(base_val).ravel()
                if len(base_val_array) > 1:
                    base_val = float(base_val_array[1])  # classe 1
                else:
                    base_val = float(base_val_array[0])
            else:
                base_val = float(base_val)

            fig, ax = plt.subplots()  
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_vals,
                    base_values=base_val,
                    data=df_input_aligned.iloc[0] if name == "AdaBoost" else df_input.iloc[0],
                    feature_names=df_input_aligned.columns if name == "AdaBoost" else df_input.columns
                ),
                show=False
            )
            st.pyplot(fig)

        elif name == "Gaussian NB":
            proba = model.predict_proba(df_input)[0]
            st.write(f"Posterior probabilities (Legitimate vs Fraudulent): {proba}")
            st.write(f"La transazione è stata classificata come **{results[name]}** con probabilità {max(proba):.2f}")

        elif name == "KNN":
            neighbors = model.kneighbors(df_input, n_neighbors=3, return_distance=True)
            neighbor_indices = neighbors[1][0]
            neighbor_labels = y_train.iloc[neighbor_indices].values
            st.write("I 3 vicini più simili e le loro etichette reali:")
            st.table(pd.DataFrame({
                "Index": neighbor_indices,
                "Label": ["Fraudulent" if l==1 else "Legitimate" for l in neighbor_labels]
            }))
            st.write(f"La transazione è stata classificata come **{results[name]}** perché {sum(neighbor_labels)} dei 3 vicini erano Fraudulent.")
