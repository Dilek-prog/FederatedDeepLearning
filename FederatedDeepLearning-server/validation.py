from collections import defaultdict
import json
import os
import re
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import log_loss, precision_score, recall_score, roc_auc_score, accuracy_score

from util import find_best_variant, get_all_data, get_model_with_weights, get_params_by_variant, get_readable_tag_from_batch
from config import METRICS_TO_PLOT, PLOTS_DIR_VAL, SHARED_VOLUME_PATH

VALIDATION_FILE = f"{SHARED_VOLUME_PATH}/training_data/validation_data.csv"
VALIDATION_DIR = f"{SHARED_VOLUME_PATH}/validation_results"

def _prepare_validation_df(validation_file: str, scaler=StandardScaler()):
    """
    Lädt validation_data.csv, wendet dieselbe Vorverarbeitung wie im Training an
    und gibt X_val, y_val zurück.
    Erwartet: gleiche Spalten wie Trainingsdaten (inkl. ID und Status)
    """
    if not os.path.exists(validation_file):
        print(f"Validation file {validation_file} not found.")
        return None, None

    df_val = pd.read_csv(validation_file, delimiter=";")

    numeric_features = [
        'Children',
        'Income',
        'Age',
        'Years_Experience',
        'Total_Family',
        'Good rate'
    ]

    X_val = df_val.drop(["ID", "Status"], axis=1)
    y_val = df_val["Status"].values

    X_val[numeric_features] = scaler.fit_transform(
        X_val[numeric_features]
    )
    X_val[numeric_features] = scaler.transform(X_val[numeric_features])

    return X_val, y_val

def validate(batch):
    os.makedirs(PLOTS_DIR_VAL, exist_ok=True)
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    X_val_file, y_val_file = _prepare_validation_df(VALIDATION_FILE)
    if X_val_file is None:
        print("No validation evaluation performed (validation file missing or invalid).")
        return

    # Falls y_val aus Training übergeben wurde, wir verwenden hier die explizit gelesene Validation-Datei
    X_val = X_val_file
    y_true = np.asarray(y_val_file).astype(int)

    # Vorhersage-Wahrscheinlichkeiten

    y_pred_prob = get_model_with_weights(batch).predict(X_val, batch_size=64).ravel()
    # harte Vorhersagen
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Metriken berechnen
    try:
        loss_val = float(log_loss(y_true, y_pred_prob))
    except Exception:
        loss_val = None

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_pred_prob))
    except Exception:
        auc = None

    validation_metrics = {
        "loss": loss_val,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "auc": auc
    }
    return validation_metrics


def create_validation_plots(validation_results):
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    # DataFrame für die Plots
    df = pd.DataFrame(validation_results)

    # Pro Metrik ein Diagramm
    for metric in METRICS_TO_PLOT:
        df_metric = df[df["metric"] == metric]

        plt.figure(figsize=(10,6))

        plt.bar(get_readable_tag_from_batch(df_metric["batch"]), df_metric["value"])

        # Y-Skala "eingezoomt"
        y_min = max(0, df_metric["value"].min() - 0.01)  # kleiner Puffer
        y_max = df_metric["value"].max() + 0.01
        if metric != "loss":
            y_max = min(1.1, df_metric["value"].max() + 0.01)

        plt.ylim(y_min, y_max)

        plt.ylabel(metric.capitalize())
        plt.title(f"Validation – Beste Variante pro Algorithmus ({metric})")
        plt.xticks(rotation=0, ha="center") 
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR_VAL, f"validation_{metric}.png"))
        plt.close()


    print(f"✅ Validierungsplots gespeichert in: {PLOTS_DIR_VAL}")

def get_validation_data():
    all_data = get_all_data()

    validation_results = []

    for metric in METRICS_TO_PLOT:
        for algo, variants in all_data.items():
            best_variant = find_best_variant(variants, metric)
            if best_variant is None:
                continue
            params = get_params_by_variant(best_variant)
            batch = f"{algo}-{params['opt']}-{params['lr']}-{params['dropout']}-{params['split']}-{int(params['split'])-1}"

            # Validierung durchführen
            metrics = validate(batch)

            validation_results.append({
                "algo": algo,
                "metric": metric,
                "value": metrics.get(metric),
                "batch": batch
            })
    return validation_results

def main():
    create_validation_plots(get_validation_data())


if __name__ == "__main__":
    main()