import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Carico dataset preprocessato solo per avere le colonne
df = pd.read_csv("C:\\Users\\vale\\OneDrive - University of Pisa\\File di Francesco Tarchi - DMML\\Dataset\\dataset_preprocessed.csv")
X = df.drop(columns=["isFraud"])
features = X.columns

# Carico i modelli
models = {
    "Random Forest": joblib.load("C:\\Users\\vale\\OneDrive - University of Pisa\\File di Francesco Tarchi - DMML\\Trained models\\RandomForest.pkl"),
    "Decision Tree": joblib.load("C:\\Users\\vale\\OneDrive - University of Pisa\\File di Francesco Tarchi - DMML\\Trained models\\DecisionTree.pkl"),
    "AdaBoost": joblib.load("C:\\Users\\vale\\OneDrive - University of Pisa\\File di Francesco Tarchi - DMML\\Trained models\\AdaBoost.pkl"),
    "XGBoost": joblib.load("C:\\Users\\vale\\OneDrive - University of Pisa\\File di Francesco Tarchi - DMML\\Trained models\\XGBoost.pkl"),
    "KNN": joblib.load("C:\\Users\\vale\\OneDrive - University of Pisa\\File di Francesco Tarchi - DMML\\Trained models\\KNN.pkl"),
    "Gaussian NB": joblib.load("C:\\Users\\vale\\OneDrive - University of Pisa\\File di Francesco Tarchi - DMML\\Trained models\\GaussianNB.pkl")
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
