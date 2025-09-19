import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from scipy.stats import ttest_rel
import paths

# Carico il training set preprocessato con SMOTE
df = pd.read_csv(paths.SMOTE20_PREP_TRAIN_PATH)
X = df.drop(columns=["isFraud"])
y = df["isFraud"]

# Carico i modelli già salvati
models = {
    "DecisionTree": joblib.load(paths.DT_PATH),
    "GaussianNB": joblib.load(paths.NB_PATH),
    "KNN": joblib.load(paths.KNN_PATH),
    "RandomForest": joblib.load(paths.RF_PATH),
    "AdaBoost": joblib.load(paths.ADA_PATH),
    "XGBoost": joblib.load(paths.XGB_PATH),
}

# Cross-validation con 10 fold
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = {name: [] for name in models}

for train_idx, test_idx in kf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        results[name].append(f1)

# Confronto: esempio RF vs XGB
f1_rf = results["RandomForest"]
f1_xgb = results["XGBoost"]

t_stat, p_val = ttest_rel(f1_rf, f1_xgb)
print("RF vs XGB - t:", t_stat, "p:", p_val)

# Appendo i risultati a un file txt
with open("model_results/t_test_results.txt", "a") as f:
    f.write(f"RF vs XGB - t: {t_stat}, p: {p_val}\n")