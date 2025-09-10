import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, SelectKBest

import paths
import constants as const


# -------------------- DATASET --------------------
print("Loading imbalanced dataset...")
df = pd.read_csv(paths.RAW_ALL_PATH)


# -------------------- FEATURE ENGINEERING --------------------
print("\nPREPROCESSING:")
# -------------------- Feature temporali -------------------- #
# La colonna TransactionDT è in secondi. Non usiamo il timestamp grezzo perché sono secondi cumulativi
# che non hanno un significato immediato: lo trasformiamo in features che catturino i pattern temporali.
print("Creating temporal features...")
df["TransactionDT_days"] = (df["TransactionDT"] / (24*60*60)).astype(int)
df["hour"] = (df["TransactionDT"] // 3600) % 24
df["dayofweek"] = (df["TransactionDT"] // (24*3600)) % 7

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)


# -------------------- FEATURE / TARGET --------------------
print("Creating feature matrix (X) and target vector (y)...")
X = df.drop(columns=["TransactionID_x", "TransactionID_y", "isFraud"])
y = df["isFraud"]


# -------------------- SPLIT TRAIN / TEST --------------------
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=const.DIM_TEST, random_state=const.RANDOM_STATE, stratify=y
)
num_cols = X.select_dtypes(include=np.number).columns.tolist()


# -------------------- IMPUTATION --------------------
print("Applying imputer...")
imputer = SimpleImputer(strategy="median")
X_train[num_cols] = imputer.fit_transform(X_train[num_cols])
X_test[num_cols] = imputer.transform(X_test[num_cols])


# -------------------- SCALING --------------------
print("Applying scaler...")
scaler = RobustScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])
print("Preprocessing complete.")


# -------------------- FEATURE SELECTION --------------------
# Variance Threshold
print(f"Selecting best features by variance threshold...")
var_thresh = VarianceThreshold(threshold=const.VARIANCE_THRESHOLD)
X_train_var = var_thresh.fit_transform(X_train)
X_test_var = var_thresh.transform(X_test)

# Mask per discrete/continue dopo il variance threshold
remaining_cols = X_train.columns[var_thresh.get_support()]
discrete_mask = [np.issubdtype(X_train[c].dtype, np.integer) for c in remaining_cols]

# Score function mutual info
print("Selecting best features by mutual information...")
def mi_score(X, y):
    return mutual_info_classif(X, y, discrete_features=discrete_mask, random_state=const.RANDOM_STATE)

# Seleziono le migliori k
selector = SelectKBest(score_func=mi_score, k=const.BEST_K_FEATURES)
X_train_sel = selector.fit_transform(X_train_var, y_train)
X_test_sel = selector.transform(X_test_var)

# Mantengo i nomi delle feature selezionate
print("Keeping names of selected features...")
selected_features = remaining_cols[selector.get_support()]
X_train = pd.DataFrame(np.asarray(X_train_sel), columns=selected_features, index=y_train.index)
X_test = pd.DataFrame(np.asarray(X_test_sel), columns=selected_features, index=y_test.index)

# Unisco X e y in un unico DataFrame (separatamente fra training e testing)
prep_train = X_train.copy()
prep_train["isFraud"] = y_train
prep_test = X_test.copy()
prep_test["isFraud"] = y_test

# Plot degli score di Mutual Information (non ci sono p-value con MI)
mi_scores = np.nan_to_num(selector.scores_)  # garantisce niente NaN
order = np.argsort(mi_scores)                # dal più basso al più alto
ordered_scores = mi_scores[order]
indices = np.arange(len(ordered_scores))

plt.bar(indices, ordered_scores, width=0.8)
plt.title("Feature Mutual Information scores (ordered)")
plt.xlabel("Feature rank")
plt.ylabel("MI score")
plt.tight_layout()
plt.show()


# -------------------- SMOTE --------------------
SAMPLING_STRATEGY = const.TARGET_MINORITY_RATIO_1_5
SMOTED_TRAIN_PATH = paths.SMOTE20_PREP_TRAIN_PATH

# SMOTE ci serve per bilanciare i dati del training set: il nostro dataset è fortemente sbilanciato (ci sono pochissime transazioni fraudolente).
print("Applying SMOTE to balance the training set...")
smote = SMOTE(random_state=const.RANDOM_STATE, sampling_strategy=SAMPLING_STRATEGY) # type: ignore
X_train_res, y_train_res = smote.fit_resample(X_train, y_train) # type: ignore[arg-type]

# Unisco X e y in un unico DataFrame
smote_prep_train = X_train_res.copy()
smote_prep_train["isFraud"] = y_train_res

# Stampo a video le stats prima e dopo SMOTE
print("Training set size before SMOTE: ", X_train.shape)
print("Training set size after SMOTE: ", X_train_res.shape)
print("Label array size before SMOTE:", y_train.value_counts())
print("Label array size after SMOTE:", y_train_res.value_counts())


# --------------------- SAVING TO CSV ---------------------
# Salvo il dataset di training preprocessato su un file CSV
print("Saving preprocessed training dataset to CSV...")
prep_train.to_csv(paths.PREP_TRAIN_PATH, index=False)

# Salvo il dataset di testing preprocessato su un file CSV
print("Saving preprocessed testing dataset to CSV...")
prep_test.to_csv(paths.PREP_TEST_PATH, index=False)

# Salvo il dataset di training preprocessato bilanciato su un file CSV
print("Saving balanced preprocessed training dataset to CSV...")
smote_prep_train.to_csv(SMOTED_TRAIN_PATH, index=False)

# Salvo le feature selezionate
print("Saving selected features to CSV...")
selected_features.to_series().to_csv(paths.SELECTED_FEATURES_PATH, index=False)

# Salvo oggetti di preprocessing già fittati
print("Saving preprocessing objects...")
joblib.dump(imputer, paths.IMPUTER_PATH)
joblib.dump(scaler, paths.SCALER_PATH)
joblib.dump(var_thresh, paths.VAR_THRESH_PATH)
joblib.dump(selector, paths.SELECTOR_PATH)
joblib.dump(selected_features, paths.SELECTED_FEATURES_PATH)
