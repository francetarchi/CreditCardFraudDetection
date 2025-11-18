import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import joblib
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE

import paths
import constants as const


# ------------------- AUXILIARY ARRAYS -------------------
print("Defining auxiliary arrays of features...")
all_categorical_features = [
    "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2",
    "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "DeviceType", "DeviceInfo",
    "id_12", "id_13", "id_14", "id_15", "id_16", "id_17", "id_18", "id_19", "id_20", "id_21", "id_22", "id_23", "id_24", "id_25", "id_26", "id_27", "id_28", "id_29","id_30", "id_31", "id_32", "id_33", "id_34", "id_35", "id_36", "id_37", "id_38"
]

all_numerical_features = [
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14",
    "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14", "D15",
    "TransactionAmt",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
    "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "V29", "V30", "V31", "V32", "V33", "V34", "V35", "V36", "V37", "V38", "V39",
    "V40", "V41", "V42", "V43", "V44", "V45", "V46", "V47", "V48", "V49", "V50", "V51", "V52", "V53", "V54", "V55", "V56", "V57", "V58", "V59",
    "V60", "V61", "V62", "V63", "V64", "V65", "V66", "V67", "V68", "V69", "V70", "V71", "V72", "V73", "V74", "V75", "V76", "V77", "V78", "V79",
    "V80", "V81", "V82", "V83", "V84", "V85", "V86", "V87", "V88", "V89", "V90", "V91", "V92", "V93", "V94", "V95", "V96", "V97", "V98", "V99",
    "V100", "V101", "V102", "V103", "V104", "V105", "V106", "V107", "V108", "V109", "V110", "V111", "V112", "V113", "V114", "V115", "V116", "V117", "V118", "V119",
    "V120", "V121", "V122", "V123", "V124", "V125", "V126", "V127", "V128", "V129", "V130", "V131", "V132", "V133", "V134", "V135", "V136", "V137", "V138", "V139",
    "V140", "V141", "V142", "V143", "V144", "V145", "V146", "V147", "V148", "V149", "V150", "V151", "V152", "V153", "V154", "V155", "V156", "V157", "V158", "V159",
    "V160", "V161", "V162", "V163", "V164", "V165", "V166", "V167", "V168", "V169", "V170", "V171", "V172", "V173", "V174", "V175", "V176", "V177", "V178", "V179",
    "V180", "V181", "V182", "V183", "V184", "V185", "V186", "V187", "V188", "V189", "V190", "V191", "V192", "V193", "V194", "V195", "V196", "V197", "V198", "V199",
    "V200", "V201", "V202", "V203", "V204", "V205", "V206", "V207", "V208", "V209", "V210", "V211", "V212", "V213", "V214", "V215", "V216", "V217", "V218", "V219",
    "V220", "V221", "V222", "V223", "V224", "V225", "V226", "V227", "V228", "V229", "V230", "V231", "V232", "V233", "V234", "V235", "V236", "V237", "V238", "V239",
    "V240", "V241", "V242", "V243", "V244", "V245", "V246", "V247", "V248", "V249", "V250", "V251", "V252", "V253", "V254", "V255", "V256", "V257", "V258", "V259",
    "V260", "V261", "V262", "V263", "V264", "V265", "V266", "V267", "V268", "V269", "V270", "V271", "V272", "V273", "V274", "V275", "V276", "V277", "V278", "V279",
    "V280", "V281", "V282", "V283", "V284", "V285", "V286", "V287", "V288", "V289", "V290", "V291", "V292", "V293", "V294", "V295", "V296", "V297", "V298", "V299",
    "V300", "V301", "V302", "V303", "V304", "V305", "V306", "V307", "V308", "V309", "V310", "V311", "V312", "V313", "V314", "V315", "V316", "V317", "V318", "V319",
    "V320", "V321", "V322", "V323", "V324", "V325", "V326", "V327", "V328", "V329", "V330", "V331", "V332", "V333", "V334", "V335", "V336", "V337", "V338", "V339",
    "dist1", "dist2",
    "id_01", "id_02", "id_03", "id_04", "id_05", "id_06", "id_07", "id_08", "id_09", "id_10", "id_11",
    "TransactionDT_days", "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos"  # nuove features temporali
]

print("\n")


# ------------------ FUNZIONI AUSILIARIE ----------------------
print("Defining auxiliary functions...")

# La funzione prende in input un DataFrame e due array di features: aggiorna gli array rimuovendo le features che non sono presenti nel DataFrame.
# Ritorna i due array aggiornati e un terzo array che è la concatenazione dei primi due.
def update_features_arrays(df, categorical_features, numerical_features):
    print("Updating features arrays...")
    cat_ftr = [col for col in categorical_features if col in df.columns]
    num_ftr = [col for col in numerical_features if col in df.columns]

    return cat_ftr, num_ftr, cat_ftr + num_ftr

# La funzione prende in input una matrice di features X e un vettore target y e ritorna gli score di mutual information fra ogni feature e il target.
def mi_score(X, y):
    if hasattr(X, "toarray"):  # se è sparse
        X = X.toarray()
    return mutual_info_classif(X, y, discrete_features=discrete_mask, random_state=const.RANDOM_STATE)

print("\n")


# -------------------- DATASET --------------------
print("Loading imbalanced dataset...")
df = pd.read_csv(paths.RAW_ALL_PATH)

print("\n")


# -------------------- CREATING RAW DATASETS --------------------
print("Splitting raw data into train and test sets...")
raw_train, raw_test = train_test_split(
    df, test_size=const.DIM_TEST, random_state=const.RANDOM_STATE, stratify=df["isFraud"]
)

print("\n")


# -------------------- FEATURE ENGINEERING --------------------
print("\nPREPROCESSING:")

# -------------------- Feature temporali --------------------
# La colonna TransactionDT è in secondi.
# Non usiamo il valore grezzo perché sono secondi cumulativi a partire da una data e ora di riferimento sconosciute, quindi NON hanno un significato immediato.
# Trasformiamo questi valori in features in cui sia possbile catturare pattern temporali (grazie all'applicazione delle funzioni seno e coseno).
print("Creating temporal features...")
df["TransactionDT_days"] = df["TransactionDT"] / (24*60*60)

hour = (df["TransactionDT"] // 3600) % 24
dayofweek = (df["TransactionDT"] // (24*3600)) % 7

df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
df["dayofweek_sin"] = np.sin(2 * np.pi * dayofweek / 7)
df["dayofweek_cos"] = np.cos(2 * np.pi * dayofweek / 7)

categorical_features, numerical_features, feature_names = update_features_arrays(df, all_categorical_features, all_numerical_features)
print("\n")


# -------------------- Features / Target --------------------
print("Creating feature matrix (X) and target vector (y)...")
X = df.drop(columns=["isFraud"])
y = df["isFraud"]

categorical_features, numerical_features, feature_names = update_features_arrays(X, all_categorical_features, all_numerical_features)
print("\n")


# -------------------- 1st feature selection --------------------
print("Applying first feature selection...")
X = X.drop(columns=["TransactionID", "TransactionDT"])  # useless for prediction
X = X.loc[:, X.isnull().mean() < const.MISSING_VALUES_THRESHOLD]    # drop features with more than 90% missing values

categorical_features, numerical_features, feature_names = update_features_arrays(X, all_categorical_features, all_numerical_features)
print("\n")


# -------------------- Train / Test --------------------
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=const.DIM_TEST, random_state=const.RANDOM_STATE, stratify=y
)

print("\n")


# -------------------- Imputation --------------------
print("Applying imputer...")
imputer = ColumnTransformer(
    transformers=[
        ("cat", SimpleImputer(strategy="most_frequent"), categorical_features),
        ("num", SimpleImputer(strategy="median"), numerical_features)
    ],
    verbose=True
)
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_names, index=y_train.index) # type: ignore
X_test = pd.DataFrame(imputer.transform(X_test), columns=feature_names, index=y_test.index) # type: ignore

categorical_features, numerical_features, feature_names = update_features_arrays(X_train, all_categorical_features, all_numerical_features)
print("\n")


# -------------------- Scaling --------------------
print("Applying scaler...")
scaler = ColumnTransformer(
    transformers=[
        ("cat", "passthrough", categorical_features),  # lascia intatte le categoriche
        ("num", RobustScaler(), numerical_features)
    ],
    verbose=True
)
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names, index=y_train.index) # type: ignore
X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=y_test.index) # type: ignore

categorical_features, numerical_features, feature_names = update_features_arrays(X_train, all_categorical_features, all_numerical_features)
print("\n")


# -------------------- Encoding --------------------
print("Applying one-hot encoding...")
encoder = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numerical_features)  # lascia intatte le numeriche
    ],
    verbose=True
)
X_train_sparse = encoder.fit_transform(X_train)
X_test_sparse = encoder.transform(X_test)

feature_names = encoder.get_feature_names_out()


# -------------------- 2nd feature selection --------------------
# Seleziono le feature con varianza sopra una certa soglia (VarianceThreshold)
print("Selecting best features by variance threshold...")
var_thresh = VarianceThreshold(threshold=const.VARIANCE_THRESHOLD)
X_train_sparse = var_thresh.fit_transform(X_train_sparse)
X_test_sparse = var_thresh.transform(X_test_sparse)

# ⬇⬇⬇ NEW: salviamo i nomi delle feature dopo variance threshold ⬇⬇⬇
# feature_names è la lista delle colonne dopo ENCODOER
feature_names_after_var = feature_names[var_thresh.get_support()]

X_train = pd.DataFrame(X_train_sparse.toarray(), columns=feature_names_after_var, index=y_train.index) # type: ignore
X_test = pd.DataFrame(X_test_sparse.toarray(), columns=feature_names_after_var, index=y_test.index) # type: ignore

# Definisco una mask per features discrete/continue
print("Creating dynamic mask for discrete features...")
remaining_cols = X_train.columns
discrete_mask = [np.issubdtype(X_train[c].to_numpy().dtype, np.integer) for c in remaining_cols]

# Seleziono le migliori k feature
print("Selecting best features by mutual information...")
selector = SelectKBest(score_func=mi_score, k=const.BEST_K_FEATURES)
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

# Plot degli score di Mutual Information
print("Plotting Mutual Information scores...")
mi_scores = np.nan_to_num(selector.scores_)  # garantisce niente NaN
order = np.argsort(mi_scores)                # dal più basso al più alto
ordered_scores = mi_scores[order]
indices = np.arange(len(ordered_scores))
plt.bar(indices, ordered_scores, width=0.8)
plt.title("Feature Mutual Information scores (ordered)")
plt.xlabel("Feature rank")
plt.ylabel("MI score")
plt.tight_layout()
plt.savefig("mutual_info/mi_scores.png")
plt.savefig("mutual_info/mi_scores.svg")

# Mantengo i nomi delle features finali
print("Keeping names of selected features...")
# selected_features = remaining_cols[selector.get_support()]

# ⬇⬇⬇ FIX: questa lista è quella da salvare e ricaricare nel preprocessing runtime ⬇⬇⬇
selected_features = feature_names_after_var[selector.get_support()]

# Ricostruisco DataFrame denso
print("Rebuilding dense DataFrame...")
X_train = pd.DataFrame(X_train, columns=selected_features, index=y_train.index)
X_test = pd.DataFrame(X_test, columns=selected_features, index=y_test.index) # type: ignore

# Unisco X e y in un unico DataFrame (separatamente fra training e testing)
print("Combining preprocessed X and y into single DataFrame...")
prep_train = X_train.copy()
prep_train["isFraud"] = y_train
prep_test = X_test.copy()
prep_test["isFraud"] = y_test

print("\n")
print("---------- Preprocessing complete ----------\n")


# -------------------- SMOTE --------------------
print("\nSMOTE:")
SAMPLING_STRATEGY = const.TARGET_MINORITY_RATIO_1_5
SMOTED_TRAIN_PATH = paths.SMOTE20_PREP_TRAIN_PATH

# SMOTE ci serve per bilanciare i dati del training set: il nostro dataset è fortemente sbilanciato (ci sono pochissime transazioni fraudolente).
print("Applying SMOTE to balance the training set...")
smote = SMOTE(random_state=const.RANDOM_STATE, sampling_strategy=SAMPLING_STRATEGY) # type: ignore
X_train, y_train = smote.fit_resample(X_train, y_train) # type: ignore[arg-type]

# Unisco X e y in un unico DataFrame
smote_prep_train = X_train.copy()
smote_prep_train["isFraud"] = y_train

print("\n")
print("---------- SMOTE complete ----------\n")


# --------------------- SAVING TO CSV ---------------------
print("\nSAVING RESULTS:")
# Salvo il dataset di training grezzo su un file CSV
print("Saving raw training dataset to CSV...")
raw_train.to_csv(paths.RAW_TRAIN_PATH, index=False)

# Salvo il dataset di testing grezzo su un file CSV
print("Saving raw testing dataset to CSV...")
raw_test.to_csv(paths.RAW_TEST_PATH, index=False)

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
# pd.DataFrame(selected_features).to_csv(paths.SELECTED_FEATURES_PATH, index=False)

# ⬇⬇⬇ NEW: salviamo selected_features in un file .npy o pickle ⬇⬇⬇
np.save(paths.SELECTED_FEATURES_PATH, selected_features)

# Salvo oggetti di preprocessing già fittati
print("Saving preprocessing objects...")
joblib.dump(imputer, paths.IMPUTER_PATH)
joblib.dump(scaler, paths.SCALER_PATH)
joblib.dump(encoder, paths.ENCODER_PATH)
joblib.dump(var_thresh, paths.VAR_THRESH_PATH)
joblib.dump(selector, paths.SELECTOR_PATH)

print("\n")
print("---------- All files saved ----------\n")
