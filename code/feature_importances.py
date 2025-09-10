import joblib
import pandas as pd
import matplotlib.pyplot as plt

import paths

# Carico dataset preprocessato solo per avere le colonne
df = pd.read_csv(paths.PREP_ALL_PATH)
X = df.drop(columns=["isFraud"])
features = X.columns

# Carico i modelli
models = {
    "KNN": joblib.load(paths.KNN_PATH),
    "Gaussian NB": joblib.load(paths.NB_PATH),
    "Decision Tree": joblib.load(paths.DT_PATH),
    "Random Forest": joblib.load(paths.RF_PATH),
    "AdaBoost": joblib.load(paths.ADA_PATH),
    "XGBoost": joblib.load(paths.XGB_PATH)
}

# Loop sui modelli che hanno feature_importances_
for name, model in models.items():
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

        plt.figure(figsize=(8, 5))
        plt.barh(features, importances)
        plt.title(f"Feature Importances - {name}")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.show()
    else:
        print(f"{name} non supporta feature_importances_")
