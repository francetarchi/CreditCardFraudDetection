### COSTANTI ###
RANDOM_STATE = 42

DIM_TRAIN = 0.75
DIM_TEST = 0.25

DIM_TRAIN_SMALL = 0.3
DIM_TEST_SMALL = 0.7

SELECT_PERCENTILE = 20  # percentuale di feature da mantenere in feature selection
VARIANCE_THRESHOLD = 0.05  # soglia per la rimozione delle feature a bassa varianza
BEST_K_FEATURES = 120  # numero di feature migliori da selezionare con SelectKBest

TARGET_MINORITY_RATIO_1_5 = 0.20  # rapporto 1:5 (1 fraud ogni 5 non fraud)
TARGET_MINORITY_RATIO_1_4 = 0.25  # rapporto 1:4 (1 fraud ogni 4 non fraud)
TARGET_MINORITY_RATIO_1_3 = 0.33  # rapporto 1:3 (1 fraud ogni 3 non fraud)
TARGET_MINORITY_RATIO_1_2 = 0.50  # rapporto 1:2 (1 fraud ogni 2 non fraud)
