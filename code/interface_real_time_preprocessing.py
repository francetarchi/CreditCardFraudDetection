import shap
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import paths

# Carico il training set per informazioni sulle feature
print("Loading balanced preprocessed training set...")
smote_train_set = pd.read_csv(paths.SMOTE20_PREP_TRAIN_PATH)

X_train = smote_train_set.drop(columns=["isFraud"])
y_train = smote_train_set["isFraud"]

def mi_score(X, y):
    if hasattr(X, "toarray"):
        X = X.toarray()
    return mutual_info_classif(X, y, random_state=42)

# Carico i preprocessori salvati
print("Loading preprocessors...")
imputer = joblib.load(paths.IMPUTER_PATH)
scaler = joblib.load(paths.SCALER_PATH)
encoder = joblib.load(paths.ENCODER_PATH)
var_thresh = joblib.load(paths.VAR_THRESH_PATH)
selector = joblib.load(paths.SELECTOR_PATH)

# Carico le feature selezionate durante il preprocessing
print("Loading selected features...")
selected_features = np.load(paths.SELECTED_FEATURES_PATH, allow_pickle=True)


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

    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df.infer_objects()
    
    # Feature Engineering Temporale
    if "TransactionDT" in df.columns:
        df["TransactionDT_days"] = df["TransactionDT"] / (24*60*60)
        hour = (df["TransactionDT"] // 3600) % 24
        dayofweek = (df["TransactionDT"] // (24*3600)) % 7
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        df["dayofweek_sin"] = np.sin(2 * np.pi * dayofweek / 7)
        df["dayofweek_cos"] = np.cos(2 * np.pi * dayofweek / 7)

    # Imputazione
    # Recuperiamo i nomi delle colonne dall'imputer per mantenere coerenza
    cat_cols = imputer.transformers_[0][2]
    num_cols = imputer.transformers_[1][2]
    imputed_cols = list(cat_cols) + list(num_cols)
    
    # Transform ritorna numpy array -> riconvertiamo in DataFrame
    imputed_array = imputer.transform(df)
    df_imputed = pd.DataFrame(imputed_array, columns=imputed_cols)
    
    # Convertiamo le colonne numeriche in float per lo scaler
    df_imputed[num_cols] = df_imputed[num_cols].astype(float)

    # Scaling
    scaled_array = scaler.transform(df_imputed)
    df_scaled = pd.DataFrame(scaled_array, columns=imputed_cols)

    # Encoding
    encoded_array = encoder.transform(df_scaled)
    
    # Variance Threshold
    var_array = var_thresh.transform(encoded_array)
    
    # Feature Selection
    selected_array = selector.transform(var_array)
    
    # Creazione DataFrame Finale
    df_final = pd.DataFrame(selected_array.toarray(), columns=selected_features)

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

edited_df = st.data_editor(user_input, num_rows="fixed", width="stretch")

# Ricostruisco il DataFrame dall'input modificabile
user_input_df = pd.DataFrame([edited_df.iloc[0].to_dict()])

if st.button("Predict"):
    df_input = preprocess_user_input(user_input_df)

   # Lista dei modelli con nomi
    models = {
        "Gaussian NB": model_nb,
        "Decision Tree": model_dt,
        "Random Forest": model_rf,
        "AdaBoost": model_ada,
        "XGBoost": model_xgb,
        "KNN": model_knn
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
        # Usiamo 50 campioni per mantenere il calcolo veloce
        background = smote_train_set.sample(50, random_state=42)
        
        # Allineamento colonne (sicurezza per evitare errori di feature mismatch)
        if hasattr(model, "feature_names_in_"):
            cols = model.feature_names_in_
            background = background[cols]
            df_input_aligned = df_input[cols]
        else:
            df_input_aligned = df_input

    
        explainer = shap.Explainer(model.predict_proba, background)
        
        explanation = explainer(df_input_aligned, max_evals=200)
        
        fig, ax = plt.subplots()
        shap.plots.waterfall(explanation[0, :, 1], show=False, max_display=14)
        st.pyplot(fig)