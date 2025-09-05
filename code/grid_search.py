import os
import datetime
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import constants as const


### CLASSIFICATORI ###
# # Decision Tree
# from sklearn.tree import DecisionTreeClassifier
# # Naïve Bayes
# from sklearn.naive_bayes import GaussianNB
# # K-NN
# from sklearn.neighbors import KNeighborsClassifier
# # Random Forest
# from sklearn.ensemble import RandomForestClassifier
# AdaBoost
from sklearn.ensemble import AdaBoostClassifier
# # XGBoost
# from xgboost import XGBClassifier


### TIMING ###
now = datetime.datetime.now()
print("\n--- Start of execution:", now.strftime("%Y-%m-%d %H:%M:%S"))


### INIZIALIZATION OPERATIONS ###
print("\nINIZIALIZING OPERATIONS:")
# Instanziazione dei modelli con relativi parametri
print("Instantiating models...")
param_grids = {
    # "DecisionTree": {
    #     "model": DecisionTreeClassifier(),
    #     "params": {
    #         "max_depth": [3, 5, 10, None],
    #         "min_samples_split": [2, 5, 10],
    #         "criterion": ["gini", "entropy", "log_loss"],
    #         "splitter": ["best", "random"],
    #         "max_leaf_nodes": [None, 10, 20, 30],
    #         "min_samples_leaf": [1, 2, 5]
    #     }
    # }
    # "NaiveBayes": {
    #     "model": GaussianNB(),
    #     "params": {
    #         "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6] 
    #     }
    # }
    # "KNN": {
    #     "model": KNeighborsClassifier(),
    #     "params": {
    #         "n_neighbors": [3, 5, 7, 11],
    #         # "n_neighbors": [3],
    #         # "weights": ["uniform", "distance"],
    #         "weights": ["distance"],
    #         # "p": [1, 2],
    #         "p": [1],
    #         # "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
    #         "algorithm": ["auto"]
    #     }
    # }
    # "RandomForest": {
    #     "model": RandomForestClassifier(),
    #     "params": {
    #         "n_estimators": [50, 100, 200],
    #         "max_depth": [5, 10, None],
    #         "criterion": ['entropy', 'gini'],
    #         'max_leaf_nodes':  [None, 10, 20, 30],
    #         'max_samples': [None, 0.5, 0.9],
    #         'min_samples_split': [2, 5, 10], 
    #         'min_impurity_decrease': [0.0, 0.1, 0.2],
    #         'bootstrap': [False, True]
    #     }
    # }
    "AdaBoost": {
        "model": AdaBoostClassifier(),
        "params": {
            "n_estimators": [50, 100, 200],
            # "learning_rate": [0.01, 0.1, 0.5, 0.75, 1.0]
        }
    }
    # "XGBoost": {
    #     "model": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    #     "params": {
    #         "n_estimators": [50, 100, 200],
    #         "max_depth": [3, 5, 7, 10],
    #         "learning_rate": [0.01, 0.1, 0.2, 0.5, 0.75, 1.0],
    #         "min_child_weight": [1, 3, 5],
    #         "gamma": [0, 0.1, 0.2]
    #     }
    # }
}


### DATASET ###
print("\nDATASETS LOADING:")
# Caricamento del dataset grezzo
print("Loading imbalanced preprocessed dataset...")
# df = pd.read_csv("C:\\Users\\berte\\OneDrive - University of Pisa\\File di Francesco Tarchi - DMML\\Dataset\\dataset_preprocessed.csv")
df = pd.read_csv("C:\\Users\\franc\\OneDrive - University of Pisa\\Documenti\\_Progetti magistrale\\DMML\\Dataset\\dataset_preprocessed.csv")

# -------------------- X e y --------------------
print("Creating feature matrix (X) and target vector (y)...")
X = df.drop(columns=["TransactionID_x", "TransactionID_y", "isFraud"])
y = df["isFraud"]


# -------------------- Train/Test split --------------------
print("Splitting data into train and test sets...")
_, X_test, _, y_test = train_test_split(
    X, y, test_size=const.DIM_TEST, random_state=42, stratify=y
)

# Caricamento del training set già bilanciato
print("Loading balanced preprocessed training set...")
# train_resampled = pd.read_csv('C:\\Users\\berte\\OneDrive - University of Pisa\\File di Francesco Tarchi - DMML\\Dataset\\train_smote_10.csv')
train_resampled = pd.read_csv('C:\\Users\\franc\\OneDrive - University of Pisa\\Documenti\\_Progetti magistrale\\DMML\\Dataset\\train_smote_10.csv')
y_train_res = train_resampled["isFraud"]
X_train_res = train_resampled.drop(columns=["isFraud"])

# Riduzione del training set già bilanciato per la GridSearch (per trovare i migliori ipermarametri per ogni modello)
print("Reducing balanced training set for GridSearch...")
X_train_res_small, _, y_train_res_small, _ = train_test_split(
    X_train_res, y_train_res, train_size=const.DIM_TRAIN_SMALL, stratify=y_train_res, random_state=42
)


### TRAINING AND TESTING ###
print("\nTRAINING AND TESTING:")
# Addestramento e valutazione dei modelli
print("Training and evaluating models...")
results = []
name = ""
for name, cfg in param_grids.items():
    print(f"\nModel: {name}")
    
    print("Instantiating grid for GridSearch...")
    grid = GridSearchCV(cfg["model"], cfg["params"], cv=5, scoring="f1", n_jobs=-1, verbose=2)
    
    print(f"Finding best hyper-parameters (on small rebalanced training set)...")
    grid.fit(X_train_res_small, y_train_res_small)
    best_params = grid.best_params_
    best_model = cfg["model"].set_params(**best_params)
    
    print("Training best model (on complete rebalanced training set)...")
    best_model.fit(X_train_res, y_train_res, verbose=2)

    print(f"Testing best model (on imbalanced test set)...")
    y_pred = best_model.predict(X_test, verbose=2)
    
    # Calcolo metriche
    print(f"Evaluating best model...")

    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion matrix:\n{cm}\n")

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
print("\nRESULTS:")
# Creo un DataFrame con i risultati
df_results = pd.DataFrame(results)

# Salvo i risultati su un file CSV
print("Saving results to CSV...")
file_path = f"model_results/{name}_{const.DIM_TRAIN_SMALL*100}.csv"
if os.path.exists(file_path):
    # Apro in append, senza scrivere l'header
    df_results.to_csv(file_path, mode='a', index=False, header=False)
else:
    # Se non esiste, scrivo normalmente con header
    df_results.to_csv(file_path, index=False)

# Mostro i risultati a schermo
print(df_results)


### TIMING ###
now = datetime.datetime.now()
print("\n--- End of execution:", now.strftime("%Y-%m-%d %H:%M:%S"))
