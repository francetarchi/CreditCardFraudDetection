import shap
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

import paths
import constants as const

# Carico i modelli salvati
print("Loading models...")
model_rf = joblib.load(paths.RF_PATH)
model_dt = joblib.load(paths.DT_PATH)
model_nb = joblib.load(paths.NB_PATH)
model_knn = joblib.load(paths.KNN_PATH)
model_ada = joblib.load(paths.ADA_PATH)
model_xgb = joblib.load(paths.XGB_PATH)

st.title("Credit Card Fraud Detection")


train_set = pd.read_csv(paths.SMOTE20_PREP_TRAIN_PATH)
test_set = pd.read_csv(paths.PREP_TEST_PATH)

X_train = train_set.drop(columns=["isFraud"])
y_train = train_set["isFraud"]

X_test = test_set.drop(columns=["isFraud"])
y_test = test_set["isFraud"]

# Mostro 5 transazioni del test set
sample_transactions = X_test.sample(5)
st.write("Select a transaction to analyze:")
st.dataframe(sample_transactions)

# Per selezionare una riga
selected_index = st.selectbox("Select transaction index:", sample_transactions.index)

# Bottone per predire
if st.button("Predict"):
    df_input = sample_transactions.loc[[selected_index]]

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
    st.dataframe(df_results.style.applymap(color_pred, subset=["Prediction"]))

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