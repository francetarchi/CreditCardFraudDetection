import os
import datetime
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
import joblib
import json

import paths
import constants as const


### CLASSIFICATORI ###
# Decision Tree
from sklearn.tree import DecisionTreeClassifier
# Naïve Bayes
from sklearn.naive_bayes import GaussianNB
# K-NN
from sklearn.neighbors import KNeighborsClassifier
# Random Forest
from sklearn.ensemble import RandomForestClassifier
# AdaBoost
from sklearn.ensemble import AdaBoostClassifier
# XGBoost
from xgboost import XGBClassifier


### TIMING ###
now = datetime.datetime.now()
print("\n--- Start of execution:", now.strftime("%Y-%m-%d %H:%M:%S"))


### INIZIALIZATION OPERATIONS ###
print("\nINIZIALIZING OPERATIONS:")
# Instanziazione dei modelli con relativi parametri
print("Instantiating models...")
param_grids = {
    "DecisionTree": {
        "model": DecisionTreeClassifier(random_state=const.RANDOM_STATE),
        "params": {
            # "max_depth": [3, 5, 10, None],
            # "min_samples_split": [2, 5, 10],
            # "criterion": ["gini", "entropy", "log_loss"],
            # "splitter": ["best", "random"],
            # "max_leaf_nodes": [None, 10, 20, 30],
            # "min_samples_leaf": [1, 2, 5]

            "max_depth": [None],
            "min_samples_split": [2],
            "criterion": ["entropy"],
            "splitter": ["random"],
            "max_leaf_nodes": [None],
            "min_samples_leaf": [1]
        }
    },
    "NaiveBayes": {
        "model": GaussianNB(),
        "params": {
            # "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

            "var_smoothing": [1e-5]
        }
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            # "n_neighbors": [3, 5, 7, 11],

            "n_neighbors": [3],
            "weights": ["distance"],
            "p": [1],
            "algorithm": ["auto"]
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=const.RANDOM_STATE),
        "params": {
            # "max_depth": [5, 10, None],
            # "n_estimators": [50, 100, 200, 500, 1000, 2000, 5000],
            # "criterion": ['entropy', 'gini'],
            # 'max_leaf_nodes': [None, 10, 20, 30],
            # 'max_samples': [None, 0.5, 0.9],
            # 'min_samples_split': [2, 5, 10],
            # 'min_impurity_decrease': [0.0, 0.1, 0.2],
            # "bootstrap": [False, True],

            "n_estimators": [1000],
            "max_depth": [None],
            "criterion": ['gini'],
            'max_leaf_nodes':  [None],
            'max_samples': [None],
            'min_samples_split': [5],
            'min_impurity_decrease': [0.0],
            "bootstrap": [False]
        }
    },
    "AdaBoost": {
        "model": AdaBoostClassifier(random_state=const.RANDOM_STATE),
        "params": {
            "n_estimators": [50, 100, 200, 500, 1000, 1500, 2000, 5000],
            "learning_rate": [0.01, 0.1, 0.5, 0.75, 1.0]

            # "n_estimators": [200],
            # "learning_rate": [1.0]
        }
    },
    "XGBoost": {
        "model": XGBClassifier(random_state=const.RANDOM_STATE),
        "params": {
            # "n_estimators": [50, 100, 200, 500, 1000, 1500, 2000, 5000],
            # "n_estimators": [300, 400, 500, 600, 700, 800, 900],
            # "max_depth": [3, 5, 7, 10, 12, 14, 16, 18, 20, 22, 25, 30],
            # "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 0.75, 1.0],
            # "min_child_weight": [1, 3, 5],
            # "gamma": [0, 0.1, 0.2],
            # "scale_pos_weight": [1, 2, 3, 4, 5, 6, 8, 10],

            "n_estimators": [500],
            "max_depth": [0],
            "learning_rate": [0.05],
            "min_child_weight": [1],
            "gamma": [0],
            "scale_pos_weight": [5]
        }
    }
}


### DATASET ###
print("\nDATASETS LOADING:")
SMOTE_PREP_TRAIN_PATH = paths.SMOTE20_PREP_TRAIN_PATH
SMOTE_DIRECTORY_PATH = paths.SMOTE20_DIRECTORY_PATH

# Caricamento del training set preprocessato (bilanciato al 20% con SMOTE)
print("Loading balanced preprocessed training set...")
train_resampled = pd.read_csv(SMOTE_PREP_TRAIN_PATH)
y_train_res = train_resampled["isFraud"]
X_train_res = train_resampled.drop(columns=["isFraud"])

# Caricamento del test set preprocessato (sbilanciato)
print("Loading preprocessed testing set...")
test_set = pd.read_csv(paths.PREP_TEST_PATH)
y_test = test_set["isFraud"]
X_test = test_set.drop(columns=["isFraud"])


### TRAINING AND TESTING ###
print("\nTRAINING AND TESTING:")
# Addestramento e valutazione dei modelli
print("Training and evaluating models...")
results = []
for name, cfg in param_grids.items():
    print(f"\nModel: {name}")
    
    print("Instantiating grid for GridSearch...")
    grid = GridSearchCV(cfg["model"], cfg["params"], cv=5, scoring=const.SCORING, refit="f1", n_jobs=-1, verbose=2, return_train_score=True)
    
    print(f"Finding best hyper-parameters with GridSearch...")
    grid.fit(X_train_res, y_train_res)
    best_model = grid.best_estimator_

    print("Extracting best model training metrics...")
    cv_results = pd.DataFrame(grid.cv_results_)

    val_acc  = cv_results["mean_test_accuracy"].max()
    val_prec = cv_results["mean_test_precision"].max()
    val_rec  = cv_results["mean_test_recall"].max()
    val_f1   = cv_results["mean_test_f1"].max()
    val_f1_w = cv_results["mean_test_f1_weighted"].max()
    val_pr_w = cv_results["mean_test_precision_weighted"].max()
    val_rc_w = cv_results["mean_test_recall_weighted"].max()
    val_bal  = cv_results["mean_test_balanced_accuracy"].max()
    val_roc  = cv_results["mean_test_roc_auc"].max()
    val_pr   = cv_results["mean_test_average_precision"].max()
    val_spec = max(0.0, min(1.0, 2*val_bal - val_rec))

    tr_acc   = cv_results["mean_train_accuracy"].max()
    tr_prec  = cv_results["mean_train_precision"].max()
    tr_rec   = cv_results["mean_train_recall"].max()
    tr_f1    = cv_results["mean_train_f1"].max()
    tr_f1_w  = cv_results["mean_train_f1_weighted"].max()
    tr_pr_w  = cv_results["mean_train_precision_weighted"].max()
    tr_rc_w  = cv_results["mean_train_recall_weighted"].max()
    tr_bal   = cv_results["mean_train_balanced_accuracy"].max()
    tr_roc   = cv_results["mean_train_roc_auc"].max()
    tr_pr    = cv_results["mean_train_average_precision"].max()
    tr_spec  = max(0.0, min(1.0, 2*tr_bal - tr_rec))

    print(f"Testing best model...")
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    print(f"Evaluating best model...")
    raw_cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion matrix:\n{raw_cm}\n")

    conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    conf_matrix_list = conf_matrix.tolist()

    metrics = {
        "Model": name,
        "Best Parameters": json.dumps(grid.best_params_),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Specificity": specificity,
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "Precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "Balanced_Accuracy": (specificity + sensitivity) / 2,
        "ROC_AUC": roc_auc_score(y_test, y_pred_proba),
        "PR_AUC": average_precision_score(y_test, y_pred_proba),
        "Confusion Matrix": json.dumps(conf_matrix_list)
    }
    results.append(metrics)


    ### RESULTS ###
    print("\nRESULTS:")
    
    # Creo un DataFrame con i risultati
    df_metrics = pd.DataFrame(metrics, index=[0])

    # Creo un DataFrame con i risultati di training e validation
    common_cols = [
        "Set", "Model",
        "Accuracy", "Balanced_Accuracy",
        "Precision", "Recall", "F1", "Specificity",
        "ROC_AUC", "PR_AUC"
    ]
    train_row = {
        "Set": "Training", "Model": name,
        "Accuracy": tr_acc, "Balanced_Accuracy": tr_bal,
        "Precision": tr_prec, "Recall": tr_rec, "F1": tr_f1, "Specificity": tr_spec,
        "ROC_AUC": tr_roc, "PR_AUC": tr_pr
    }
    val_row = {
        "Set": "Validation", "Model": name,
        "Accuracy": val_acc, "Balanced_Accuracy": val_bal,
        "Precision": val_prec, "Recall": val_rec, "F1": val_f1, "Specificity": val_spec,
        "ROC_AUC": val_roc, "PR_AUC": val_pr
    }
    test_row = {
        "Set": "Test", "Model": name,
        "Accuracy": metrics["Accuracy"], "Balanced_Accuracy": metrics["Balanced_Accuracy"],
        "Precision": metrics["Precision"], "Recall": metrics["Recall"], "F1": metrics["F1"], "Specificity": metrics["Specificity"],
        "ROC_AUC": metrics["ROC_AUC"], "PR_AUC": metrics["PR_AUC"]
    }
    df_metrics_aux = pd.DataFrame([train_row, val_row, test_row], columns=common_cols)

    # Salvo i risultati su file CSV
    print("Saving results to CSV...")
    file_path = f"model_results/{name}_{const.TARGET_MINORITY_RATIO_1_5*100}.csv"
    aux_path = f"model_results/{name}_aux.csv"
    if os.path.exists(file_path):
        df_metrics.to_csv(file_path, mode='a', index=False, header=False)
    else:
        df_metrics.to_csv(file_path, index=False)
    if os.path.exists(aux_path):
        # Scrivo in append mode senza header, lasciando una riga vuota prima
        with open(aux_path, mode='a') as f:
            f.write('\n')
        f.close()
        df_metrics_aux.to_csv(aux_path, mode='a', index=False, header=False)
    else:
        df_metrics_aux.to_csv(aux_path, index=False)
    
    # Mostro i risultati a schermo
    print(df_metrics)

    # Creo la cartella se non esiste
    os.makedirs(SMOTE_DIRECTORY_PATH, exist_ok=True)

    # Salvo il modello migliore in un file pickle
    path = os.path.join(SMOTE_DIRECTORY_PATH, f"{name}.pkl")
    joblib.dump(best_model, path)
    print(f"{name} salvato in {path}")


# Creo un DataFrame unico con tutti i risultati
df_results = pd.DataFrame(results)

# Memorizzo tutti i risultati in un unico file CSV
df_results.to_csv("model_results/unique.csv", index=False)

### TIMING ###
now = datetime.datetime.now()
print("\n--- End of execution:", now.strftime("%Y-%m-%d %H:%M:%S"))
