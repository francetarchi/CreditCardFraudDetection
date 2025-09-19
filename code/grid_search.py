import os
import datetime
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
import joblib
import json  # <--- aggiunto

import paths
import constants as const


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


### TIMING ###
now = datetime.datetime.now()
print("\n--- Start of execution:", now.strftime("%Y-%m-%d %H:%M:%S"))


### INIZIALIZATION OPERATIONS ###
print("\nINIZIALIZING OPERATIONS:")
# Instanziazione dei modelli con relativi parametri
print("Instantiating models...")
param_grids = {
    "DecisionTree": {
        "model": DecisionTreeClassifier(random_state=const.RANDOM_STATE),
        "params": {
            # "max_depth": [3, 5, 10, None],
            # "min_samples_split": [2, 5, 10],
            # "criterion": ["gini", "entropy", "log_loss"],
            # "splitter": ["best", "random"],
            # "max_leaf_nodes": [None, 10, 20, 30],
            # "min_samples_leaf": [1, 2, 5]
            "max_depth": [None],
            "min_samples_split": [5],
            "criterion": ["entropy"],
            "splitter": ["best"],
            "max_leaf_nodes": [None],
            "min_samples_leaf": [1]
        }
    },
    "NaiveBayes": {
        "model": GaussianNB(),
        "params": {
            # "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]
            "var_smoothing": [1e-7]
        }
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            # "n_neighbors": [3, 5, 7, 11],
            "n_neighbors": [3],
            "weights": ["distance"],
            "p": [1],
            "algorithm": ["auto"]
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=const.RANDOM_STATE),
        "params": {
            # "max_depth": [5, 10, None],
            # "n_estimators": [50, 100, 200],
            # "criterion": ['entropy', 'gini'],
            # 'max_leaf_nodes':  [None, 10, 20, 30],
            # 'max_samples': [None, 0.5, 0.9],
            # 'min_samples_split': [2, 5, 10],
            # 'min_impurity_decrease': [0.0, 0.1, 0.2],
            # "bootstrap": [False, True]
            "n_estimators": [200],
            "max_depth": [None],
            "criterion": ['gini'],
            'max_leaf_nodes':  [None],
            'max_samples': [None],
            'min_samples_split': [2],
            'min_impurity_decrease': [0.0],
            "bootstrap": [False]
        }
    },
    "AdaBoost": {
        "model": AdaBoostClassifier(random_state=const.RANDOM_STATE),
        "params": {
            # "n_estimators": [50, 100, 200],
            # "learning_rate": [0.01, 0.1, 0.5, 0.75, 1.0]
            "n_estimators": [200],
            "learning_rate": [1.0]
        }
    },
    "XGBoost": {
        "model": XGBClassifier(random_state=const.RANDOM_STATE),
        "params": {
            # "n_estimators": [50, 100, 200],
            # "max_depth": [3, 5, 7, 10],
            # "learning_rate": [0.01, 0.1, 0.2, 0.5, 0.75, 1.0],
            # "min_child_weight": [1, 3, 5],
            # "gamma": [0, 0.1, 0.2]
            "n_estimators": [200],
            "max_depth": [10],
            "learning_rate": [0.75],
            "min_child_weight": [1],
            "gamma": [0]
        }
    }
}


### DATASET ###
print("\nDATASETS LOADING:")
SMOTE_PREP_TRAIN_PATH = paths.SMOTE20_PREP_TRAIN_PATH
SMOTE_DIRECTORY_PATH = paths.SMOTE20_DIRECTORY_PATH

# Caricamento del training set preprocessato (bilanciato al 20% con SMOTE)
print("Loading balanced preprocessed training set...")
train_resampled = pd.read_csv(SMOTE_PREP_TRAIN_PATH)
y_train_res = train_resampled["isFraud"]
X_train_res = train_resampled.drop(columns=["isFraud"])

# Caricamento del test set preprocessato (sbilanciato)
print("Loading preprocessed testing set...")
test_set = pd.read_csv(paths.PREP_TEST_PATH)
y_test = test_set["isFraud"]
X_test = test_set.drop(columns=["isFraud"])


### TRAINING AND TESTING ###
print("\nTRAINING AND TESTING:")
# Addestramento e valutazione dei modelli
print("Training and evaluating models...")
results = []
for name, cfg in param_grids.items():
    print(f"\nModel: {name}")
    
    print("Instantiating grid for GridSearch...")
    grid = GridSearchCV(cfg["model"], cfg["params"], cv=5, scoring="f1", n_jobs=-1, verbose=2)
    
    print(f"Finding best hyper-parameters (on small rebalanced training set)...")
    grid.fit(X_train_res, y_train_res)
    best_model = grid.best_estimator_

    print(f"Testing best model (on imbalanced test set)...")
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Calcolo metriche
    print(f"Evaluating best model...")

    raw_cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion matrix:\n{raw_cm}\n")

    conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = conf_matrix.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    conf_matrix_list = conf_matrix.tolist()

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Specificity": specificity,
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "Precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "Balanced_Accuracy": (specificity + sensitivity) / 2,
        "ROC_AUC": roc_auc_score(y_test, y_pred_proba),
        "PR_AUC": average_precision_score(y_test, y_pred_proba),
        "Confusion Matrix": json.dumps(conf_matrix_list)
    }
    results.append(metrics)

    ### RESULTS ###
    print("\nRESULTS:")
    # Creo un DataFrame con i risultati
    df_metrics = pd.DataFrame(metrics, index=[0])

    # Salvo i risultati su un file CSV
    print("Saving results to CSV...")
    file_path = f"model_results/{name}_{const.TARGET_MINORITY_RATIO_1_5*100}.csv"
    if os.path.exists(file_path):
        # Apro in append, senza scrivere l'header
        df_metrics.to_csv(file_path, mode='a', index=False, header=False)
    else:
        # Se non esiste, scrivo normalmente con header
        df_metrics.to_csv(file_path, index=False)

    # Mostro i risultati a schermo
    print(df_metrics)

    # Creo la cartella se non esiste
    os.makedirs(SMOTE_DIRECTORY_PATH, exist_ok=True)

    # Salvo il modello migliore in un file pickle
    path = os.path.join(SMOTE_DIRECTORY_PATH, f"{name}.pkl")
    joblib.dump(best_model, path)
    print(f"{name} salvato in {path}")


# Creo un DataFrame unico con tutti i risultati
df_results = pd.DataFrame(results)

# Memorizzo tutti i risultati in un unico file CSV
df_results.to_csv("model_results/unique.csv", index=False)

### TIMING ###
now = datetime.datetime.now()
print("\n--- End of execution:", now.strftime("%Y-%m-%d %H:%M:%S"))
