import csv
import pandas as pd
import itertools
# from scipy.stats import ttest_rel
from scipy.stats import t
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import paths
import constants as const


# Carico il training set preprocessato con SMOTE
df = pd.read_csv(paths.PREP_TRAIN_PATH)
X = df.drop(columns=["isFraud"])
y = df["isFraud"]

# Modelli da valutare (con iperparametri ottimizzati)
models = {
    "KNN": KNeighborsClassifier(algorithm="auto", n_neighbors=3, p=1, weights="distance"),
    "GaussianNB": GaussianNB(var_smoothing=1e-05),
    "DecisionTree": DecisionTreeClassifier(criterion="entropy", max_depth=None, max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, splitter="random"),
    "RandomForest": RandomForestClassifier(bootstrap=False, criterion="gini", max_depth=None, max_leaf_nodes=None, max_samples=None, min_impurity_decrease=0.0, min_samples_split=5, n_estimators=1000),
    "AdaBoost": AdaBoostClassifier(learning_rate=1.0, n_estimators=5000),
    "XGBoost": XGBClassifier(gamma=1.0, learning_rate=0.05, max_depth=15, min_child_weight=1, n_estimators=1000, scale_pos_weight=4, subsample=0.8)
}

# Calcolo dimensioni per la correzione statistica
n_total = len(X)
n_splits = 10
n_test_avg = n_total / n_splits       # Dimensione media test set (1/10)
n_train_avg = n_total - n_test_avg    # Dimensione media train set (9/10)

def corrected_paired_ttest(scores_a, scores_b, n_train, n_test):
    # Differenza punto a punto
    diff = np.array(scores_a) - np.array(scores_b)
    n = len(diff)
    
    mean_diff = np.mean(diff)
    var_diff = np.var(diff, ddof=1)
    
    # Fattore di correzione per la dipendenza tra training set
    correction = (1/n) + (n_test/n_train)
    
    # Si evitano divisioni per zero se i modelli sono identici
    if var_diff == 0:
        return 0.0, 1.0
        
    # Calcolo t-statistica corretta
    t_stat = mean_diff / np.sqrt(correction * var_diff)
    
    # Calcolo p-value
    df = n - 1
    p_val = 2 * (1 - t.cdf(abs(t_stat), df))
    
    return t_stat, p_val

# Cross-validation con 10 fold
# kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# smote = SMOTE(random_state=const.RANDOM_STATE, sampling_strategy=const.TARGET_MINORITY_RATIO_1_5)

# results_f1 = {name: [] for name in models}
# results_roc_auc = {name: [] for name in models}

# for train_idx, test_idx in kf.split(X, y):
#     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#     y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

#     # Applico SMOTE solo sul training set
#     X_train, y_train = smote.fit_resample(X_train, y_train)
    
#     for name, model in models.items():
#         print("Fold:", len(results_f1[name]) + 1, "Model:", name)
#         model.fit(X_train, y_train)

#         y_pred = model.predict(X_test)
#         y_prob = model.predict_proba(X_test)[:, 1] # Probabilità per la classe positiva

#         f1 = f1_score(y_test, y_pred, zero_division=0)
#         roc_auc = roc_auc_score(y_test, y_prob)

#         results_f1[name].append(f1)
#         results_roc_auc[name].append(roc_auc)

#         print(f"{name} F1 Score: {f1}")
#         print(f"{name} ROC AUC: {roc_auc}")

# df_results_f1 = pd.DataFrame(results_f1)
# df_results_f1.index = [f"Fold_{i+1}" for i in range(1, len(df_results_f1) + 1)]
# df_results_f1.to_csv("model_results/cv_results_f1.csv", index=False)

# df_results_roc_auc = pd.DataFrame(results_roc_auc)
# df_results_roc_auc.index = [f"Fold_{i+1}" for i in range(1, len(df_results_roc_auc) + 1)]
# df_results_roc_auc.to_csv("model_results/cv_results_roc_auc.csv", index=False)

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
    # t_stat, p_val = ttest_rel(f1_a, f1_b)
    t_stat, p_val = corrected_paired_ttest(f1_a, f1_b, n_train_avg, n_test_avg)
    t_test_results_f1.append((model_a, model_b, t_stat, p_val))
    print(f"{model_a} vs {model_b} - t: {t_stat}, p: {p_val}")

# Salvo i risultati del t-test in un file CSV
with open("model_results/corrected_t_test_results_f1.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Model_A", "Model_B", "T_statistic", "P_value"])
    writer.writerows(t_test_results_f1)

# t-test sui risultati ROC AUC per ogni coppia di modelli
for model_a, model_b in all_possible_pairs:
    roc_auc_a = results_roc_auc[model_a]
    roc_auc_b = results_roc_auc[model_b]
    # t_stat, p_val = ttest_rel(roc_auc_a, roc_auc_b)
    t_stat, p_val = corrected_paired_ttest(roc_auc_a, roc_auc_b, n_train_avg, n_test_avg)
    t_test_results_roc_auc.append((model_a, model_b, t_stat, p_val))
    print(f"{model_a} vs {model_b} - t: {t_stat}, p: {p_val}")

# Salvo i risultati del t-test in un file CSV
with open("model_results/corrected_t_test_results_roc_auc.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Model_A", "Model_B", "T_statistic", "P_value"])
    writer.writerows(t_test_results_roc_auc)

t_test_results_f1 = pd.read_csv("model_results/corrected_t_test_results_f1.csv").to_dict(orient="list")
t_test_results_roc_auc = pd.read_csv("model_results/corrected_t_test_results_roc_auc.csv").to_dict(orient="list")

# Visualizzo i risultati del t-test F1 e ROC AUC con due heatmap dei p-value
models_names = list(models.keys())

p_matrix_f1 = pd.DataFrame(np.ones((len(models_names), len(models_names))), index=models_names, columns=models_names)
p_matrix_roc_auc = pd.DataFrame(np.ones((len(models_names), len(models_names))), index=models_names, columns=models_names)

for i in range(len(t_test_results_f1["Model_A"])):
    model_a = t_test_results_f1["Model_A"][i]
    model_b = t_test_results_f1["Model_B"][i]
    p_value_f1 = t_test_results_f1["P_value"][i]
    p_value_roc_auc = t_test_results_roc_auc["P_value"][i]
    
    p_matrix_f1.loc[model_a, model_b] = p_value_f1
    p_matrix_f1.loc[model_b, model_a] = p_value_f1
    
    p_matrix_roc_auc.loc[model_a, model_b] = p_value_roc_auc
    p_matrix_roc_auc.loc[model_b, model_a] = p_value_roc_auc

plt.figure(figsize=(25, 10))
plt.subplot(1, 2, 1)
np.fill_diagonal(p_matrix_f1.values, np.nan)
sns.heatmap(p_matrix_f1, annot=True, fmt=".4f", cmap="RdYlGn_r", mask=p_matrix_f1.isnull(), cbar_kws={"label": "P-value"}, vmin=0, vmax=0.1)
# plt.title("T-test P-values for F1 Score")
plt.title("Corrected t-test P-values for F1 Score")
plt.xlabel("Model B")
plt.ylabel("Model A")

plt.subplot(1, 2, 2)
np.fill_diagonal(p_matrix_roc_auc.values, np.nan)
sns.heatmap(p_matrix_roc_auc, annot=True, fmt=".4f", cmap="RdYlGn_r", mask=p_matrix_roc_auc.isnull(), cbar_kws={"label": "P-value"}, vmin=0, vmax=0.1)
# plt.title("T-test P-values for ROC AUC")
plt.title("Corrected t-test P-values for ROC AUC")
plt.xlabel("Model B")
plt.ylabel("Model A")

plt.savefig("model_results/corrected_t_test_p_value_heatmaps.svg", bbox_inches='tight')
plt.show()

np.fill_diagonal(p_matrix_f1.values, 1)
np.fill_diagonal(p_matrix_roc_auc.values, 1)
# Mask per evidenziare solo i valori significativi
mask_f1 = p_matrix_f1 >= 0.05
mask_roc_auc = p_matrix_roc_auc >= 0.05 
t_matrix_f1 = pd.DataFrame(np.zeros((len(models_names), len(models_names))), index=models_names, columns=models_names)
t_matrix_roc_auc = pd.DataFrame(np.zeros((len(models_names), len(models_names))), index=models_names, columns=models_names)

for i in range(len(t_test_results_f1["Model_A"])):
    model_a = t_test_results_f1["Model_A"][i]
    model_b = t_test_results_f1["Model_B"][i]
    t_stat_f1 = t_test_results_f1["T_statistic"][i]
    t_stat_roc_auc = t_test_results_roc_auc["T_statistic"][i]
    
    t_matrix_f1.loc[model_a, model_b] = t_stat_f1
    t_matrix_f1.loc[model_b, model_a] = -t_stat_f1
    
    t_matrix_roc_auc.loc[model_a, model_b] = t_stat_roc_auc
    t_matrix_roc_auc.loc[model_b, model_a] = -t_stat_roc_auc

plt.figure(figsize=(25, 10))
plt.subplot(1, 2, 1)
sns.heatmap(t_matrix_f1, annot=True, fmt=".4f", cmap="RdYlGn", mask=mask_f1, cbar_kws={"label": "T-statistic"}, vmin=-130, vmax=110)
# plt.title("T-test T-statistics for F1 Score (p-value < 0.05)")
plt.title("Corrected t-test T-statistics for F1 Score (p-value < 0.05)")
plt.xlabel("Model B")
plt.ylabel("Model A")

plt.subplot(1, 2, 2)
sns.heatmap(t_matrix_roc_auc, annot=True, fmt=".4f", cmap="RdYlGn", mask=mask_roc_auc, cbar_kws={"label": "T-statistic"}, vmin=-130, vmax=110)
# plt.title("T-test T-statistics for ROC AUC (p-value < 0.05)")
plt.title("Corrected t-test T-statistics for ROC AUC (p-value < 0.05)")
plt.xlabel("Model B")
plt.ylabel("Model A")

plt.savefig("model_results/corrected_t_test_t_statistic_heatmaps.svg", bbox_inches='tight')
plt.show()
