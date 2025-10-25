import csv
import pandas as pd
import itertools
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import paths


# Carico il training set preprocessato con SMOTE
df = pd.read_csv(paths.SMOTE20_PREP_TRAIN_PATH)
X = df.drop(columns=["isFraud"])
y = df["isFraud"]

best_params = {}

# Modelli da valutare (con iperparametri ottimizzati da prendere)
models = {
    "DecisionTree": DecisionTreeClassifier(),
    "GaussianNB": GaussianNB(),
    "KNN": KNeighborsClassifier(),
    "RandomForest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "XGBoost": XGBClassifier()
}

# Cross-validation con 10 fold
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = {name: [] for name in models}

for train_idx, test_idx in kf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1] # Probabilità per la classe positiva
        # y_pred = model.predict(X_test)
        # f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        results[name].append(roc_auc)
        print(f"{name} ROC AUC: {roc_auc:.4f}")

df_results = pd.DataFrame(results)
df_results.to_csv("model_results/cv_results_roc_auc.csv", index=False)

all_possible_pairs = list(itertools.combinations(models.keys(), 2))

t_test_results = []

# t-test sui risultati ROC AUC per ogni coppia di modelli
for model_a, model_b in all_possible_pairs:
    roc_auc_a = results[model_a]
    roc_auc_b = results[model_b]
    t_stat, p_val = ttest_rel(roc_auc_a, roc_auc_b)
    t_test_results.append((model_a, model_b, t_stat, p_val))
    print(f"{model_a} vs {model_b} - t: {t_stat}, p: {p_val}")

# Salvo i risultati del t-test in un file CSV
with open("model_results/t_test_results_roc_auc.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Model_A", "Model_B", "T_statistic", "P_value"])
    writer.writerows(t_test_results)
