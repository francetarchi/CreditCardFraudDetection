### COSTANTI ###
RANDOM_STATE = 42

DIM_TRAIN = 0.75
DIM_TEST = 0.25

MISSING_VALUES_THRESHOLD = 0.90  # soglia per la rimozione delle feature con troppi valori mancanti
VARIANCE_THRESHOLD = 0.05  # soglia per la rimozione delle feature a bassa varianza
BEST_K_FEATURES = 60    # numero di feature migliori da selezionare con SelectKBest

TARGET_MINORITY_RATIO_1_5 = 0.20  # rapporto 1:5 (1 fraud ogni 5 non fraud)
TARGET_MINORITY_RATIO_1_4 = 0.25  # rapporto 1:4 (1 fraud ogni 4 non fraud)
TARGET_MINORITY_RATIO_1_3 = 0.33  # rapporto 1:3 (1 fraud ogni 3 non fraud)
TARGET_MINORITY_RATIO_1_2 = 0.50  # rapporto 1:2 (1 fraud ogni 2 non fraud)

MAX_SAMPLES_GLOBAL = 100    # massimo per SHAP / permutation
MIN_MINORITY_KEEP = 17      # tengo almeno questo numero di frodi

TOP_N = 20  # numero di feature da mostrare nei grafici

# Scoring metrics for model evaluation
SCORING = {
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "f1_weighted": "f1_weighted",
        "precision_weighted": "precision_weighted",
        "recall_weighted": "recall_weighted",
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
    }
