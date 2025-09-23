import os

METRICS_TO_PLOT = ["accuracy", "loss", "precision", "recall", "auc"]
RESULTS_DIR = os.path.abspath("shared-data")
PLOTS_DIR = os.path.abspath("plots")
PLOTS_DIR_ALL = f"{PLOTS_DIR}/all_variants"
PLOTS_DIR_BEST = f"{PLOTS_DIR}/best_variants"
PLOTS_DIR_VAL = f"{PLOTS_DIR}/validation"
PLOTS_DIR_TRAP_VAL = f"{PLOTS_DIR}/trap_val"