import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap
import os
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier, export_text
from typing import Tuple
import numpy as np

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
    "Gaussian NB": joblib.load(paths.NB_PATH),
    "Decision Tree": joblib.load(paths.DT_PATH),
    "Random Forest": joblib.load(paths.RF_PATH),
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
    X_min = X[minority_mask]
    y_min = y[minority_mask]
    X_maj = X[~minority_mask]
    y_maj = y[~minority_mask]

    # tengo tutte le minoranze (o limito se enorme)
    if len(X_min) > min_minority:
        X_min = X_min.sample(n=min_minority, random_state=const.RANDOM_STATE)
        y_min = y.loc[X_min.index]

    remaining = max_samples - len(X_min)
    if remaining < 0:
        remaining = 0

    if remaining < len(X_maj):
        X_maj = X_maj.sample(n=remaining, random_state=const.RANDOM_STATE)
        y_maj = y.loc[X_maj.index]

    X_s = pd.concat([X_min, X_maj])
    y_s = pd.concat([y_min, y_maj])
    idx = np.random.RandomState(const.RANDOM_STATE).permutation(len(X_s))
    return X_s.iloc[idx], y_s.iloc[idx]

# Campione per spiegazioni globali
X_expl, y_expl = make_stratified_sample(X_test, y_test, const.MAX_SAMPLES_GLOBAL, const.MIN_MINORITY_KEEP)
print(f"Sample per spiegazioni: {X_expl.shape[0]} righe (fraud={y_expl.sum()})")

# Loop sui modelli che hanno feature_importances_
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
        plt.savefig(os.path.join(output_dir, f"{name}_feature_importances.png"))
        plt.close()
    else:
        print(f"{name} non supporta feature_importances_")

    # -------- SHAP --------
    try:
        shap_values = None
        if name in ["Random Forest", "XGBoost", "Decision Tree"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_expl)
        else:
            explainer = shap.KernelExplainer(model.predict_proba, X_expl)
            shap_values = explainer.shap_values(X_expl, nsamples=100)

        plt.figure()
        shap.summary_plot(shap_values[:, :, 1], X_expl, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name}_shap_summary.png"))
        plt.close()
        print(f"SHAP summary plot salvato per {name} (sample size={len(X_expl)})")

    except Exception as e:
        print(f"SHAP non disponibile per {name}: {e}")


    # -------- Permutation Feature Importances --------
    try:
        result = permutation_importance(model, X_expl, y_expl, n_repeats=10, random_state=const.RANDOM_STATE)
        perm_importances = result.importances_mean # type: ignore
        indices = perm_importances.argsort()[::-1][:const.TOP_N]

        plt.figure(figsize=(8, 5))
        plt.barh(range(len(indices)), perm_importances[indices][::-1])
        plt.yticks(range(len(indices)), features[indices][::-1])
        plt.title(f"Permutation Feature Importances - {name}")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name}_permutation_importances.png"))
        plt.close()
    except Exception as e:
        print(f"Permutation importance non disponibile per {name}: {e}")

    # -------- Rule Extraction (Surrogate Decision Tree) --------
    try:
        X_sub = X_train.sample(1000, random_state=42)
        y_sub = model.predict(X_sub)

        surrogate = DecisionTreeClassifier(max_depth=3)
        surrogate.fit(X_sub, y_sub)
        rules = export_text(surrogate, feature_names=list(features))
        rules_file = os.path.join(output_dir, f"{name}_surrogate_rules.txt")
        with open(rules_file, "w") as f:
            f.write(rules)
        print(f"Rule extraction salvata per {name} in {rules_file}")
    except Exception as e:
        print(f"Rule extraction non disponibile per {name}: {e}")
