import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap
import os
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier, export_text

import paths

# Carico dataset preprocessato
df_train = pd.read_csv(paths.SMOTE20_PREP_TRAIN_PATH)

# Uso un sottoinsieme per velocizzare i calcoli
df_train_sample = df_train.sample(n=1000, random_state=42)
X_train = df_train_sample.drop(columns=["isFraud"])
y_train = df_train_sample["isFraud"]

# Lavoro su un sottoinsieme del train set

df_test = pd.read_csv(paths.PREP_TEST_PATH).sample(n=1000, random_state=42)
X_test = df_test.drop(columns=["isFraud"])
y_test = df_test["isFraud"]
features = X_test.columns

# Carico i modelli
models = {
    "Decision Tree": joblib.load(paths.DT_PATH),
    "XGBoost": joblib.load(paths.XGB_PATH),
    "Gaussian NB": joblib.load(paths.NB_PATH),
    "Random Forest": joblib.load(paths.RF_PATH),
    "AdaBoost": joblib.load(paths.ADA_PATH),
    "KNN": joblib.load(paths.KNN_PATH)
}

# Directory per salvare i grafici
output_dir = "graphs/explanations"
os.makedirs(output_dir, exist_ok=True)

top_n = 20

# Loop sui modelli che hanno feature_importances_
print("Calculating explanations...")

for name, model in models.items():
    print(f"\n================ {name} =================")

    # -------- Feature Importances --------
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = importances.argsort()[::-1][:top_n]

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
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_train)
        plt.figure()
        shap.summary_plot(shap_values, X_train, show=False)
        plt.savefig(os.path.join(output_dir, f"{name}_shap_summary.png"))
        plt.close()
        print(f"SHAP summary plot salvato per {name}")
    except Exception as e:
        print(f"SHAP non disponibile per {name}: {e}")

    # -------- Permutation Feature Importance --------
    try:
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        perm_importances = result.importances_mean
        indices = perm_importances.argsort()[::-1][:top_n]

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
        surrogate = DecisionTreeClassifier(max_depth=3)
        surrogate.fit(X_train, model.predict(X_train))
        rules = export_text(surrogate, feature_names=list(features))
        rules_file = os.path.join(output_dir, f"{name}_surrogate_rules.txt")
        with open(rules_file, "w") as f:
            f.write(rules)
        print(f"Rule extraction salvata per {name} in {rules_file}")
    except Exception as e:
        print(f"Rule extraction non disponibile per {name}: {e}")