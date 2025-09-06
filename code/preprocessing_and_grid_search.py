import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

### CLASSIFICATORI ###
# Decision Tree
from sklearn.tree import DecisionTreeClassifier
# Naïve Bayes
from sklearn.naive_bayes import GaussianNB
# K-NN
from sklearn.neighbors import KNeighborsClassifier
# Random Forest
from sklearn.ensemble import RandomForestClassifier
# AdaBoost
from sklearn.ensemble import AdaBoostClassifier
# XGBoost
from xgboost import XGBClassifier


### INIZIALIZATION OPERATIONS ###
print("\nINIZIALIZING OPERATIONS:")

# Dichiarazione costanti
DIM_TRAIN = 0.75
DIM_TEST = 0.25
DIM_TRAIN_SMALL = 0.3
DIM_TEST_SMALL = 0.7

# Instanziazione dei modelli con relativi parametri
print("Instantiating models...")
param_grids = {
    "DecisionTree": {
        "model": DecisionTreeClassifier(),
        "params": {
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [2, 5, 10],
            "criterion": ["gini", "entropy", "log_loss"],
            "splitter": ["best", "random"],
            "max_leaf_nodes": [None, 10, 20, 30],   
            "min_samples_leaf": [1, 2, 5], 
        }
    },
    "NaiveBayes": {
        "model": GaussianNB(),
        "params": {
            "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6] 
        }
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7, 11],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, None],
            "criterion": ['entropy', 'gini'],
            'max_leaf_nodes':  [None, 10, 20, 30],
            'max_samples': [None, 0.5, 0.9],
            'min_samples_split': [2, 5, 10], 
            'min_impurity_decrease': [0.0, 0.1, 0.2],
            'bootstrap': [False, True]
        }
    },
    "AdaBoost": {
        "model": AdaBoostClassifier(),
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.5, 1.0],
            "algorithm": ["SAMME", "SAMME.R"]
        }
    },
    "XGBoost": {
        "model": XGBClassifier(),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7, 10],
            "learning_rate": [0.01, 0.1, 0.2, 0.5, 1.0],
            "min_child_weight": [1, 3, 5],
            "gamma": [0, 0.1, 0.2]
        }
    }
}


### DATASET ###
print("Loading imbalanced dataset...")
df = pd.read_csv("C:\\Users\\berte\\OneDrive - University of Pisa\\File di Francesco Tarchi - DMML\\Dataset\\dataset.csv")

# Caricamento del training set già bilanciato
print("Loading resampled training set...")
train_resampled = pd.read_csv('C:\\Users\\berte\\OneDrive - University of Pisa\\File di Francesco Tarchi - DMML\\Dataset\\train_smote_10.csv')
y_train_res = train_resampled["isFraud"]
X_train_res = train_resampled.drop(columns=["isFraud"])

# Riduzione del training set già bilanciato per la GridSearch (per trovare i migliori ipermarametri per ogni modello)
print("Splitting data into train and test sets...")
X_train_res_small, _, y_train_res_small, _ = train_test_split(
    X_train_res, y_train_res, train_size=DIM_TRAIN_SMALL, stratify=y_train_res, random_state=42
)

# # Conta i valori mancanti per colonna
# missing_counts = df.isnull().sum()

# # Ordina le colonne dal più “vuoto” al meno “vuoto”
# missing_counts = missing_counts.sort_values(ascending=False)

# # Mostra il risultato
# print(missing_counts)

# # colonne che sembrano categoriche (tipo object o pochi valori unici)
# cat_cols = [col for col in df.columns if df[col].dtype == "object" or df[col].nunique() < 50]

# for col in cat_cols:
#     print(f"{col}: {df[col].nunique()} categorie")


### PREPROCESSING ###
print("\nPREPROCESSING:")
# La colonna TransactionDT è in secondi: non usiamo il timestamp grezzo perché sono secondi cumulativi che non hanno
# un significato immediato. Lo trasformiamo in features che catturino i pattern temporali.
# -------------------- Feature temporali --------------------
print("Creating temporal features...")
df["TransactionDT_days"] = (df["TransactionDT"] / (24*60*60)).astype(int)
df["hour"] = (df["TransactionDT"] // 3600) % 24
df["dayofweek"] = (df["TransactionDT"] // (24*3600)) % 7

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

# -------------------- Colonne numeriche --------------------
print("Processing numerical features...")
num_cols = df.select_dtypes(include=np.number).columns.tolist()
num_cols.remove("isFraud")  # escludiamo target

imputer = SimpleImputer(strategy="mean")
df[num_cols] = imputer.fit_transform(df[num_cols])
df[num_cols] = df[num_cols].fillna(-1)

scaler = RobustScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# -------------------- X e y --------------------
print("Creating feature matrix (X) and target vector (y)...")
X = df.drop(columns=["TransactionID_x", "TransactionID_y", "isFraud"])
y = df["isFraud"]



# -------------------- Train/Test split --------------------
print("Splitting data into train and test sets...")
_, X_test, _, y_test = train_test_split(
    X, y, test_size=DIM_TEST, random_state=42, stratify=y
)

# # -------------------- SMOTE --------------------
# print("Applying SMOTE to balance the training set...")
# smote = SMOTE(random_state=42, sampling_strategy=1.0)
# X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# # Unisco X e y in un unico DataFrame -- Se voglio evitarmi ogni volta di fare eseguire SMOTE da capo
# # --- SOLO LA PRIMA VOLTA ---
# train_resampled = X_train_res.copy()
# train_resampled["isFraud"] = y_train_res
# train_resampled.to_csv("train_smote_.csv", index=False)

# print("Prima di SMOTE:", y_train.value_counts())
# print("Dopo SMOTE:", y_train_res.value_counts())

# print("Training set size before SMOTE: ", X_train.shape)
# print("Training set size after SMOTE: ", X_train_res.shape)


### TRAINING AND TESTING ###
print("\nTRAINING AND TESTING:")
# Addestramento e valutazione dei modelli
print("Training and evaluating models...")
results = []
name = ""
for name, cfg in param_grids.items():
    print(f"\nModel: {name}")
    
    print("Instantiating grid for GridSearch...")
    grid = GridSearchCV(cfg["model"], cfg["params"], cv=5, scoring="f1", n_jobs=4, verbose=2)
    
    print(f"Finding best hyper-parameters (on small rebalanced training set)...")
    grid.fit(X_train_res_small, y_train_res_small)
    best_params = grid.best_params_
    best_model = cfg["model"].set_params(**best_params)
    
    print("Training best model (on complete rebalanced training set)...")
    best_model.fit(X_train_res, y_train_res)

    print(f"Testing best model (on imbalanced test set)...")
    y_pred = best_model.predict(X_test)
    
    # Calcolo metriche
    print(f"Evaluating best model...")
    
    print(f"Confusion matrix: {confusion_matrix(y_test, y_pred)}")
    print(f"[0,0]: {confusion_matrix(y_test, y_pred)[0,0]}")
    print(f"[0,1]: {confusion_matrix(y_test, y_pred)[0,1]}")
    print(f"[1,0]: {confusion_matrix(y_test, y_pred)[1,0]}")
    print(f"[1,1]: {confusion_matrix(y_test, y_pred)[1,1]}")

    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    metrics = {
        "Model": name,
        "Best Params": grid.best_params_,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Specificity": specificity,        
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "Precision_weighted": precision_score(y_test, y_pred, average="weighted"),
        "Recall_weighted": recall_score(y_test, y_pred, average="weighted"),
        "F1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "Balanced_Accuracy": (specificity + sensitivity) / 2,
        "Confusion Matrix": cm.tolist(),
    }
    results.append(metrics)


### RESULTS ###
# Salvo i risultati su un file CSV
print("Saving results to CSV...")
# Percorso del file
csv_file = f"model_results/{name}_{DIM_TRAIN_SMALL*100}.csv"
# Creo il DataFrame dei risultati correnti
df_results = pd.DataFrame(results)
# Controllo se il file esiste già
if os.path.exists(csv_file):
    # Apro in append, senza scrivere l'header
    df_results.to_csv(csv_file, mode='a', index=False, header=False)
else:
    # Se non esiste, scrivo normalmente con header
    df_results.to_csv(csv_file, index=False)

# Mostro i risultati
print("\nRESULTS:")
print(df_results)
