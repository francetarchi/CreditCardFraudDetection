import streamlit as st
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

import constants as const

# Carico i modelli salvati
model_rf = joblib.load("C:\\Users\\vale\\OneDrive - University of Pisa\\File di Francesco Tarchi - DMML\\Trained models\\RandomForest.pkl")
model_dt = joblib.load("C:\\Users\\vale\\OneDrive - University of Pisa\\File di Francesco Tarchi - DMML\\Trained models\\DecisionTree.pkl")
model_nb = joblib.load("C:\\Users\\vale\\OneDrive - University of Pisa\\File di Francesco Tarchi - DMML\\Trained models\\GaussianNB.pkl")
model_knn = joblib.load("C:\\Users\\vale\\OneDrive - University of Pisa\\File di Francesco Tarchi - DMML\\Trained models\\KNN.pkl")
model_ab = joblib.load("C:\\Users\\vale\\OneDrive - University of Pisa\\File di Francesco Tarchi - DMML\\Trained models\\AdaBoost.pkl")
model_xgb = joblib.load("C:\\Users\\vale\\OneDrive - University of Pisa\\File di Francesco Tarchi - DMML\\Trained models\\XGBoost.pkl")

st.title("Credit Card Fraud Detection")

df = pd.read_csv("C:\\Users\\vale\\OneDrive - University of Pisa\\File di Francesco Tarchi - DMML\\Dataset\\dataset_preprocessed.csv")

X = df.drop(columns=["isFraud"])
y = df["isFraud"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=const.DIM_TEST, random_state=42, stratify=y
)

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
        "AdaBoost": model_ab
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
    st.subheader("SHAP Explanations (per singolo modello)")

    for name, model in models.items():
        if name in ["Random Forest", "Decision Tree", "AdaBoost"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df_input)

            # Per classificatori binari, shap_values è una lista [classe0, classe1]
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            fig, ax = plt.subplots()
            shap.waterfall_plot(shap.Explanation(values=shap_values[0],
                                                 base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, (list, tuple)) else explainer.expected_value,
                                                 data=df_input.iloc[0],
                                                 feature_names=df_input.columns))
            st.pyplot(fig)

        elif name == "XGBoost":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(df_input)

            fig, ax = plt.subplots()
            shap.waterfall_plot(shap_values[0])
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

