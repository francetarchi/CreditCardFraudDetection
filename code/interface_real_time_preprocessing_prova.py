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



############# AUXILIARY FUNCTIONS #############
# Funzione per il colore della predizione
def color_pred(val):
        return 'color: red' if val == "Fraudulent" else 'color: green'


# Funzione per il calcolo dei mutual_information score
def mi_score(X, y):
    if hasattr(X, "toarray"):
        X = X.toarray()
    return mutual_info_classif(X, y, random_state=42)


#Funzione per caricare il dataset
@st.cache_data(show_spinner=False)  # salvo il dataset in memoria, altrimenti Streamlit lo ricarica ogni volta che clicco "Predict"
def load_dataset():
    print("--- SETUP")
    print("Loading balanced preprocessed training set...")
    smote_train_set = pd.read_csv(paths.SMOTE20_PREP_TRAIN_PATH)

    X_train = smote_train_set.drop(columns=["isFraud"])
    y_train = smote_train_set["isFraud"]

    return smote_train_set, X_train, y_train


# Funzione per caricare la pipeline di preprocessing
@st.cache_resource(show_spinner=False)  # salvo i preprocessors e le selected_features in memoria, altrimenti Streamlit li ricarica ogni volta che clicco "Predict"
def load_preprocessing_pipeline():
    print("Loading preprocessors...")
    imputer = joblib.load(paths.IMPUTER_PATH)
    scaler = joblib.load(paths.SCALER_PATH)
    encoder = joblib.load(paths.ENCODER_PATH)
    var_thresh = joblib.load(paths.VAR_THRESH_PATH)
    selector = joblib.load(paths.SELECTOR_PATH)

    print("Loading selected features...")
    selected_features = np.load(paths.SELECTED_FEATURES_PATH, allow_pickle=True)

    return imputer, scaler, encoder, var_thresh, selector, selected_features


# Funzione per caricare i modelli salvati
@st.cache_resource(show_spinner=False)  # salvo i modelli in memoria, altrimenti Streamlit li ricarica ogni volta che clicco "Predict"
def load_models():
    print("Loading models...")
    loaded_models = {
        "KNN": joblib.load(paths.KNN_PATH),
        "Gaussian NB": joblib.load(paths.NB_PATH),
        "DecisionTree": joblib.load(paths.DT_PATH),
        "RandomForest": joblib.load(paths.RF_PATH),
        "AdaBoost": joblib.load(paths.ADA_PATH),
        "XGBoost": joblib.load(paths.XGB_PATH)
    }

    return loaded_models


# Funzione per caricare gli SHAP explainers
@st.cache_resource(show_spinner=False)  # salvo gli SHAP explainers in memoria, altrimenti Streamlit li ricarica ogni volta che clicco "Predict"
def load_shap_explainers(_models, _smote_train_set):
    print("Building SHAP explainers...")
    explainers = {}

    # Creo il background data di SHAP (per utilizzare solamente 50 campioni)
    background_data = _smote_train_set.sample(50, random_state=42)

    for name, model in _models.items():
        # Allineamento colonne per sicurezza
        if hasattr(model, "feature_names_in_"):
            cols = model.feature_names_in_
            bg = background_data[cols]
        else:
            bg = background_data
            
        explainers[name] = shap.Explainer(model.predict_proba, bg)
    
    print("--- SETUP DONE: you can use the application.\n")
    return explainers


# Funzione di preprocessing per l'input utente
def preprocess_user_input(df):
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
    
    # Creazione DataFrame finale
    df_final = pd.DataFrame(selected_array.toarray(), columns=selected_features)

    return df_final



############# LOADING ELEMENTS #############
with st.spinner('Loading system resources...'):
    smote_train_set, X_train, y_test = load_dataset()
    imputer, scaler, encoder, var_thresh, selector, selected_features = load_preprocessing_pipeline()
    models = load_models()
    shap_explainers = load_shap_explainers(models, smote_train_set)



############# STREAMLIT APPLICATION #############
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

# Definisco il bottone "Predict" per effettuare la predizione e le spiegazioni
if st.button("Predict"):
    print("--- PREDICT")

    ## Preprocessing ##
    print("Preprocessing user input...")
    df_input = preprocess_user_input(user_input_df)
    print("  --> DONE.")

    ## Predizioni ##
    print("Predicting user input...")
    results = {}
    for name, model in models.items():
        pred = model.predict(df_input)[0]
        results[name] = "Fraudulent" if pred else "Legitimate"
    print("  --> DONE.")

    ## Risultati ##
    print("Displaying results...")

    # Tabella dei risultati (colorata)
    df_results = pd.DataFrame(list(results.items()), columns=["Model", "Prediction"])
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
    
    print("  --> DONE.")

    ## SHAP explanations ##
    print("Computing SHAP explanations...")
    st.subheader("Explanations (per model):")
    for name, explainer in shap_explainers.items():
        with st.expander(f"See explanation for {name}"):
            # Allineamento input
            model = models[name]
            if hasattr(model, "feature_names_in_"):
                df_input_aligned = df_input[model.feature_names_in_]
            else:
                df_input_aligned = df_input
            
            # Calcolo SHAP values
            explanation = explainer(df_input_aligned, max_evals=200)    # max_evals controlla la precisione del calcolo
            
            fig, ax = plt.subplots()
            shap.plots.waterfall(explanation[0, :, 1], show=False, max_display=14)
            st.pyplot(fig)
    print("  --> DONE.")

    print("--- PREDICT DONE: you can now ask for a new prediction.\n")
