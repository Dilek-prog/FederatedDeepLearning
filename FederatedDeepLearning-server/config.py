import os
SHARED_VOLUME_PATH = os.path.abspath("shared-data")
HISTORY_PATH = os.path.abspath("history")
METRICS_TO_PLOT = ["accuracy", "loss", "precision", "recall", "auc"]
RESULTS_DIR = os.path.abspath("shared-data")
RESULT_DATA_DIR = os.path.abspath("results")
PLOTS_DIR = os.path.abspath("plots")
PLOTS_DIR_ALL = f"{PLOTS_DIR}/all_variants"
PLOTS_DIR_BEST = f"{PLOTS_DIR}/best_variants"
PLOTS_DIR_VAL = f"{PLOTS_DIR}/validation"
PLOTS_DIR_POISON_VAL = f"{PLOTS_DIR}/poisoning"
TRAINING_DATA = f"{SHARED_VOLUME_PATH}/training_data"
VALIDATION_FILE = f"{TRAINING_DATA}/validation_data.csv"