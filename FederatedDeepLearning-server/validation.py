import os
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import log_loss, precision_score, recall_score, roc_auc_score, accuracy_score

from util import get_model_with_weights
from config import SHARED_VOLUME_PATH

VALIDATION_FILE = f"{SHARED_VOLUME_PATH}/training_data/validation_data.csv"

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
        'Income',
        'Age',
        'Years_Experience',
        'Total_Family',
    ]

    X_val = df_val.drop(["ID", "Status"], axis=1)
    y_val = df_val["Status"].values

    X_val[numeric_features] = scaler.fit_transform(
        X_val[numeric_features]
    )
    X_val[numeric_features] = scaler.transform(X_val[numeric_features])

    return X_val, y_val

def validate(batch):
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

