import os
import joblib
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# ----------------- dataset -----------------
# Caricamento del training set già bilanciato
print("Loading resampled training set...")
train = pd.read_csv('C:\\Users\\berte\\OneDrive - University of Pisa\\File di Francesco Tarchi - DMML\\Dataset\\train_smote_10.csv')
y_train = train["isFraud"]
X_train = train.drop(columns=["isFraud"])

# ----------------- iperparametri trovati -----------------
best_params = {
    "decision_tree": {"max_depth": None, "min_samples_split": 5, "criterion": "log_loss", "splitter": "best", "max_leaf_nodes": None, "min_samples_leaf": 1},
    "naive_bayes": {"var_smoothing": 1e-9},
    "knn": {"n_neighbors": 3, "weights": "distance", "p": 1, "algorithm": "auto"},
    "random_forest": {"n_estimators": 200, "max_depth": 10, "criterion": "entropy", "max_leaf_nodes": None, "max_samples": None, "min_samples_split": 2, "min_impurity_decrease": 0.0, "bootstrap": False},
    "adaboost": {"n_estimators": 200, "learning_rate": 0.5, "algorithm": "SAMME"},
    "xgboost": {"n_estimators": 200, "max_depth": 10, "learning_rate": 0.75, "min_child_weight": 1, "gamma": 0},
}

# ----------------- modelli -----------------
models = {
    "decision_tree": DecisionTreeClassifier(**best_params["decision_tree"]),
    "random_forest": RandomForestClassifier(**best_params["random_forest"]),
    "adaboost": AdaBoostClassifier(**best_params["adaboost"]),
    "naive_bayes": GaussianNB(**best_params["naive_bayes"]),
    "knn": KNeighborsClassifier(**best_params["knn"]),
    "xgboost": XGBClassifier(**best_params["xgboost"]),
}


# Percorso OneDrive
onedrive_dir = r"C:\\Users\\berte\\OneDrive - University of Pisa\\File di Francesco Tarchi - DMML\\Trained models"

# Creo la cartella se non esiste
os.makedirs(onedrive_dir, exist_ok=True)

# ----------------- training e salvataggio -----------------
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    path = os.path.join(onedrive_dir, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"{name} salvato in {path}")

