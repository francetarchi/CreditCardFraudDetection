import shap
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

import paths

# Carico il training set per informazioni sulle feature
print("Loading balanced preprocessed training set...")
smote_train_set = pd.read_csv(paths.SMOTE20_PREP_TRAIN_PATH)

X_train = smote_train_set.drop(columns=["isFraud"])
y_train = smote_train_set["isFraud"]

# Carico i preprocessori salvati
print("Loading preprocessors...")
imputer = joblib.load(paths.IMPUTER_PATH)
scaler = joblib.load(paths.SCALER_PATH)
encoder = joblib.load(paths.ENCODER_PATH)

# Carico le feature selezionate durante il preprocessing
print("Loading selected features...")
selected_features = pd.read_csv(paths.SELECTED_FEATURES_PATH)

# Carico i modelli salvati
print("Loading models...")
model_rf = joblib.load(paths.RF_PATH)
model_dt = joblib.load(paths.DT_PATH)
model_nb = joblib.load(paths.NB_PATH)
model_knn = joblib.load(paths.KNN_PATH)
model_ada = joblib.load(paths.ADA_PATH)
model_xgb = joblib.load(paths.XGB_PATH)


# Funzione di preprocessing per l'input utente
def preprocess_user_input(df):
    print("Preprocessing user input...")
    
    # --- Feature temporali ---
    df["TransactionDT_days"] = df["TransactionDT"] / (24*60*60)
    hour = (df["TransactionDT"] // 3600) % 24
    dayofweek = (df["TransactionDT"] // (24*3600)) % 7
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dayofweek_sin"] = np.sin(2 * np.pi * dayofweek / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * dayofweek / 7)

    # --- Imputazione ---
    df = imputer.transform(df)

    # --- Scaling ---
    df = scaler.transform(df)

    # --- Encoding ---
    df = encoder.transform(df)

    # --- Feature selection ---
    df_final = pd.DataFrame(df, columns=selected_features.columns)

    return df_final


# Streamlit app
st.title("Credit Card Fraud Detection")
st.subheader("Insert a transaction to classify:")

file_csv = paths.RAW_TEST_PATH
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

edited_df = st.data_editor(user_input, num_rows="fixed", use_container_width=True)

# Ricostruisco il DataFrame dall'input modificabile
user_input_df = pd.DataFrame([edited_df.iloc[0].to_dict()])

if st.button("Predict"):
    df_input = preprocess_user_input(user_input_df)

   # Lista dei modelli con nomi
    models = {
        "KNN": model_knn,
        "Gaussian NB": model_nb,
        "Decision Tree": model_dt,
        "Random Forest": model_rf,
        "AdaBoost": model_ada,
        "XGBoost": model_xgb
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
    st.subheader("Model Predictions:")
    st.dataframe(df_results.style.map(color_pred, subset=["Prediction"]))

    # Voto di maggioranza
    fraud_votes = list(results.values()).count("Fraudulent")
    legit_votes = list(results.values()).count("Legitimate")
    st.subheader("Final Model Vote:")
    if fraud_votes > legit_votes:
        st.error(f"Transaction classified as Fraudulent by {fraud_votes}/{len(models)} models!")
    else:
        st.success(f"Transaction classified as Legitimate by {legit_votes}/{len(models)} models!")

    # SHAP explanations
    st.subheader("Explanations (per model):")
    for name, model in models.items():
        st.subheader(f"Explanation for {name}")
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
                base_val = float(base_val) # type: ignore

            # Waterfall plot
            fig, ax = plt.subplots()  
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_vals,
                    base_values=base_val,
                    data=df_input.iloc[0],
                    feature_names=df_input.columns
                ),
                show=False
            )
            st.pyplot(fig)

        elif name == "AdaBoost":
            # Background set: usa un piccolo campione del training set
            background = smote_train_set.sample(50, random_state=42)
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
                    data=df_input_aligned.iloc[0],
                    feature_names=df_input_aligned.columns
                ),
                show=False
            )
            st.pyplot(fig)

        elif name == "Gaussian NB":
            proba = model.predict_proba(df_input)[0]
            st.write(f"Posterior probabilities (Legitimate vs Fraudulent): {proba}")
            st.write(f"Transaction classified as **{results[name]}** with probability {max(proba):.2f}")

        elif name == "KNN":
            neighbors = model.kneighbors(df_input, n_neighbors=3, return_distance=True)
            neighbor_indices = neighbors[1][0]
            neighbor_labels = y_train.iloc[neighbor_indices].values
            st.write("The 3 most similar neighbors and their true labels:")
            st.table(pd.DataFrame({
                "Index": neighbor_indices,
                "Label": ["Fraudulent" if l==1 else "Legitimate" for l in neighbor_labels]
            }))
            st.write(f"Transaction classified as **{results[name]}** because {sum(neighbor_labels)} of the 3 neighbors were Fraudulent.")
