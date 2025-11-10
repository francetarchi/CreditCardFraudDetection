import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shap
import joblib
from typing import Tuple
from sklearn.utils import shuffle
from sklearn.inspection import permutation_importance
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

import paths
import constants as const


# Carico dataset bilanciato preprocessato
print("Loading balanced preprocessed train set...")
smote_train_set = pd.read_csv(paths.SMOTE20_PREP_TRAIN_PATH)
X_train = smote_train_set.drop(columns=["isFraud"])
y_train = smote_train_set["isFraud"]

# Carico il test set preprocessato
print("Loading preprocessed test set...")
df_test = pd.read_csv(paths.PREP_TEST_PATH)
X_test = df_test.drop(columns=["isFraud"])
y_test = df_test["isFraud"]
features = X_test.columns

# Carico i modelli
models = {
    "KNN": joblib.load(paths.KNN_PATH),
    "GaussianNB": joblib.load(paths.NB_PATH),
    "DecisionTree": joblib.load(paths.DT_PATH),
    "RandomForest": joblib.load(paths.RF_PATH),
    "AdaBoost": joblib.load(paths.ADA_PATH),
    "XGBoost": joblib.load(paths.XGB_PATH)
}

# Directory per salvare i grafici
output_dir = "graphs/explanations"
os.makedirs(output_dir, exist_ok=True)

def make_stratified_sample(X: pd.DataFrame, y: pd.Series,
                           max_samples: int,
                           min_minority: int) -> Tuple[pd.DataFrame, pd.Series]:
    minority_mask = y == 1

    # righe con isFraud = 1
    X_minority = X[minority_mask]
    y_minority = y[minority_mask]
    
    # righe con isFraud = 0
    X_majority = X[~minority_mask]
    y_majority = y[~minority_mask]

    # tengo tutte le minoranze (o limito se sono troppe)
    if len(X_minority) > min_minority:
        X_minority = X_minority.sample(n=min_minority, random_state=const.RANDOM_STATE)
        y_minority = y_minority.loc[X_minority.index] # ricampiono y di conseguenza (ricollego le etichette corrette)

    # calcolo quante maggioranze (isFraud = 0) posso tenere
    remaining = max_samples - len(X_minority)
    if remaining < 0:
        remaining = 0

    if remaining < len(X_majority):
        X_majority = X_majority.sample(n=remaining, random_state=const.RANDOM_STATE)
        y_majority = y_majority.loc[X_majority.index]

    X_sample = pd.concat([X_minority, X_majority])
    y_sample = pd.concat([y_minority, y_majority])
    X_sample, y_sample = shuffle(X_sample, y_sample, random_state=const.RANDOM_STATE) # type: ignore
    return X_sample, y_sample # type: ignore

# Campione per spiegazioni globali
X_test_sample, y_test_sample = make_stratified_sample(X_test, y_test, const.MAX_SAMPLES_GLOBAL, const.MIN_MINORITY_KEEP)
X_train_sample, y_train_sample = make_stratified_sample(X_train, y_train, const.MAX_SAMPLES_GLOBAL, const.MIN_MINORITY_KEEP)

print("Calculating explanations...")

for name, model in models.items():
    print(f"\n================ {name} =================")

    # -------- Feature Importances --------
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = importances.argsort()[::-1][:const.TOP_N]

        plt.figure(figsize=(8, 5))
        plt.barh(range(len(indices)), importances[indices][::-1])
        plt.yticks(range(len(indices)), features[indices][::-1])
        plt.title(f"Feature Importances - {name}")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name}_feature_importances.svg"))
        plt.close()
    else:
        print(f"{name} does not support feature_importances_")

    # -------- SHAP --------
    try:
        shap_values = None
        if name in ["RandomForest", "XGBoost", "DecisionTree"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_test_sample)
        else:
            explainer = shap.KernelExplainer(model.predict_proba, X_train_sample)
            shap_values = explainer(X_test_sample)

        output_to_explain = 1  # classe 1

        plt.figure()
        shap.summary_plot(shap_values[:, :, output_to_explain], X_test_sample, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name}_shap_summary.svg"))
        plt.close()
        print(f"SHAP summary plot saved for {name} (sample size={len(X_test_sample)})")

    except Exception as e:
        print(f"SHAP not available for {name}: {e}")


    # -------- Permutation Feature Importances --------
    try:
        result = permutation_importance(model, X_test_sample, y_test_sample, n_repeats=10, random_state=const.RANDOM_STATE)
        perm_importances = result.importances_mean # type: ignore
        indices = perm_importances.argsort()[::-1][:const.TOP_N]

        plt.figure(figsize=(8, 5))
        plt.barh(range(len(indices)), perm_importances[indices][::-1])
        plt.yticks(range(len(indices)), features[indices][::-1])
        plt.title(f"Permutation Feature Importances - {name}")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name}_permutation_importances.svg"))
        plt.close()
    except Exception as e:
        print(f"Permutation importance not available for {name}: {e}")

    # -------- Rule Extraction (Surrogate Decision Tree) --------
    try:
        surrogate = DecisionTreeClassifier(max_depth=3)
        surrogate.fit(X_train_sample, y_train_sample)

        tree.plot_tree(surrogate, proportion=True)
        plt.title(f"Surrogate Decision Tree - {name}")
        plt.savefig(os.path.join(output_dir, f"{name}_surrogate_tree.svg"))
        plt.close()

        # Estrazione delle regole testuali
        rules = tree.export_text(surrogate)
        rules_file = os.path.join(output_dir, f"{name}_surrogate_rules.txt")
        with open(rules_file, "w") as f:
            f.write(rules)
        print(f"Rule extraction saved for {name} in {rules_file}")
    except Exception as e:
        print(f"Rule extraction not available for {name}: {e}")
