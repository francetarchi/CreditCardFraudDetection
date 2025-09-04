import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE              # ci serve a bilanciare i dati del training set, il nostro dataset è fortemente sbilanciato (ci sono pochissime transazioni fraudolente)
from sklearn.model_selection import train_test_split


### COSTANTI ###
DIM_TEST = 0.25
DIM_SMOTE = 1.0   # bilanciamento perfetto


### DATASET ###
print("Loading imbalanced dataset...")
# df = pd.read_csv("C:\\Users\\berte\\OneDrive - University of Pisa\\File di Francesco Tarchi - DMML\\Dataset\\dataset.csv")
df = pd.read_csv("C:\\Users\\franc\\OneDrive - University of Pisa\\Documenti\\_Progetti magistrale\\DMML\\Dataset\\dataset.csv")


### PREPROCESSING ###
print("\nPREPROCESSING:")
# La colonna TransactionDT è in secondi: non usiamo il timestamp grezzo perché sono secondi cumulativi che non hanno
# un significato immediato. Lo trasformiamo in features che catturino i pattern temporali.
# -------------------- Feature temporali -------------------- #
print("Creating temporal features...")
df["TransactionDT_days"] = (df["TransactionDT"] / (24*60*60)).astype(int)
df["hour"] = (df["TransactionDT"] // 3600) % 24
df["dayofweek"] = (df["TransactionDT"] // (24*3600)) % 7

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

# -------------------- Colonne numeriche --------------------
print("Processing numerical features...")
num_cols = df.select_dtypes(include=np.number).columns.tolist()
num_cols.remove("isFraud")  # escludiamo target

imputer = SimpleImputer(strategy="mean")
df[num_cols] = imputer.fit_transform(df[num_cols])
df[num_cols] = df[num_cols].fillna(-1)

scaler = RobustScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Salvo il dataset preprocessato su un file CSV
print("Saving preprocessed dataset to CSV...")
# file_path = "C:\\Users\\berte\\OneDrive - University of Pisa\\File di Francesco Tarchi - DMML\\Dataset\\dataset_preprocessed.csv"
file_path = "C:\\Users\\franc\\OneDrive - University of Pisa\\Documenti\\_Progetti magistrale\\DMML\\Dataset\\dataset_preprocessed.csv"
df.to_csv(file_path, index=False)

# -------------------- SMOTE -------------------- #
# SMOTE ci serve per bilanciare i dati del training set: il nostro dataset è fortemente sbilanciato (ci sono pochissime transazioni fraudolente).

# -------------------- X e y --------------------
print("Creating feature matrix (X) and target vector (y)...")
X = df.drop(columns=["TransactionID_x", "TransactionID_y", "isFraud"])
y = df["isFraud"]


# -------------------- Train/Test split --------------------
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=DIM_TEST, random_state=42, stratify=y
)

print("Applying SMOTE to balance the training set...")
smote = SMOTE(random_state=42, sampling_strategy=DIM_SMOTE) # type: ignore
X_train_res, y_train_res = smote.fit_resample(X_train, y_train) # type: ignore

# Unisco X e y in un unico DataFrame
train_resampled = X_train_res.copy()
train_resampled["isFraud"] = y_train_res
train_resampled.to_csv("train_smote_.csv", index=False)

print("Prima di SMOTE:", y_train.value_counts())
print("Dopo SMOTE:", y_train_res.value_counts())

print("Training set size before SMOTE: ", X_train.shape)
print("Training set size after SMOTE: ", X_train_res.shape)

# Salvo il dataset di training preprocessato bilanciato su un file CSV
print("Saving preprocessed balanced training dataset to CSV...")
# file_path = f"C:\\Users\\berte\\OneDrive - University of Pisa\\File di Francesco Tarchi - DMML\\Dataset\\train_smote_{DIM_SMOTE*10}.csv"
file_path = f"C:\\Users\\franc\\OneDrive - University of Pisa\\Documenti\\_Progetti magistrale\\DMML\\Dataset\\train_smote_{DIM_SMOTE*10}.csv"
train_resampled.to_csv(file_path, index=False)
