import csv
import pandas as pd
import itertools
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import paths
import constants as const


# Carico il training set preprocessato con SMOTE
df = pd.read_csv(paths.PREP_TRAIN_PATH)
X = df.drop(columns=["isFraud"])
y = df["isFraud"]


# Modelli da valutare (con iperparametri ottimizzati)
models = {
    "DecisionTree": DecisionTreeClassifier(criterion="entropy", max_depth=None, max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, splitter="random"),
    "GaussianNB": GaussianNB(var_smoothing=1e-05),
    "KNN": KNeighborsClassifier(algorithm="auto", n_neighbors=3, p=1, weights="distance"),
    "RandomForest": RandomForestClassifier(bootstrap=False, criterion="gini", max_depth=None, max_leaf_nodes=None, max_samples=None, min_impurity_decrease=0.0, min_samples_split=5, n_estimators=1000),
    "AdaBoost": AdaBoostClassifier(learning_rate=1.0, n_estimators=5000),
    "XGBoost": XGBClassifier(gamma=1.0, learning_rate=0.05, max_depth=15, min_child_weight=1, n_estimators=1000, scale_pos_weight=4, subsample=0.8)
}

# Cross-validation con 10 fold
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

smote = SMOTE(random_state=const.RANDOM_STATE, sampling_strategy=const.TARGET_MINORITY_RATIO_1_5)

results_f1 = {name: [] for name in models}
results_roc_auc = {name: [] for name in models}

for train_idx, test_idx in kf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Applico SMOTE solo sul training set
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    for name, model in models.items():
        print("Fold:", len(results_f1[name]) + 1, "Model:", name)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] # Probabilità per la classe positiva

        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)

        results_f1[name].append(f1)
        results_roc_auc[name].append(roc_auc)

        print(f"{name} F1 Score: {f1}")
        print(f"{name} ROC AUC: {roc_auc}")

df_results_f1 = pd.DataFrame(results_f1)
df_results_f1.index = [f"Fold_{i+1}" for i in range(1, len(df_results_f1) + 1)]
df_results_f1.to_csv("model_results/cv_results_f1.csv", index=False)

df_results_roc_auc = pd.DataFrame(results_roc_auc)
df_results_roc_auc.index = [f"Fold_{i+1}" for i in range(1, len(df_results_roc_auc) + 1)]
df_results_roc_auc.to_csv("model_results/cv_results_roc_auc.csv", index=False)

results_f1 = pd.read_csv("model_results/cv_results_f1.csv").to_dict(orient="list")
results_roc_auc = pd.read_csv("model_results/cv_results_roc_auc.csv").to_dict(orient="list")

# Genero tutte le possibili coppie di modelli per il t-test
all_possible_pairs = list(itertools.combinations(models.keys(), 2))

t_test_results_f1 = []
t_test_results_roc_auc = []

# t-test sui risultati F1 per ogni coppia di modelli
for model_a, model_b in all_possible_pairs:
    f1_a = results_f1[model_a]
    f1_b = results_f1[model_b]
    t_stat, p_val = ttest_rel(f1_a, f1_b)
    t_test_results_f1.append((model_a, model_b, t_stat, p_val))
    print(f"{model_a} vs {model_b} - t: {t_stat}, p: {p_val}")

# Salvo i risultati del t-test in un file CSV
with open("model_results/t_test_results_f1.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Model_A", "Model_B", "T_statistic", "P_value"])
    writer.writerows(t_test_results_f1)

# t-test sui risultati ROC AUC per ogni coppia di modelli
for model_a, model_b in all_possible_pairs:
    roc_auc_a = results_roc_auc[model_a]
    roc_auc_b = results_roc_auc[model_b]
    t_stat, p_val = ttest_rel(roc_auc_a, roc_auc_b)
    t_test_results_roc_auc.append((model_a, model_b, t_stat, p_val))
    print(f"{model_a} vs {model_b} - t: {t_stat}, p: {p_val}")

# Salvo i risultati del t-test in un file CSV
with open("model_results/t_test_results_roc_auc.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Model_A", "Model_B", "T_statistic", "P_value"])
    writer.writerows(t_test_results_roc_auc)


