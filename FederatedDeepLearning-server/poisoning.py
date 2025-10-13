from collections import defaultdict
import json
import os
import re
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import log_loss, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf

from util import SHARED_VOLUME_PATH, find_best_variant, get_all_data, get_model_with_weights, get_params_by_variant, get_readable_tag_from_batch, retrieve_global_metrics, retrieve_local_metrics, retrieve_local_weights
from config import METRICS_TO_PLOT, PLOTS_DIR_POISON_VAL, PLOTS_DIR_VAL, VALIDATION_FILE
from merger import astraea_merge, fed_merge, no_merge, fed_prox



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

def poison(batch):
    """
    returns new weights after training model with fake data
    """

    def _set_up_poison(source: str):
        df = pd.read_csv(source, delimiter=";")

        # poison Data
        df["Status"] = 1 - df["Status"]

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            df.drop(['ID', 'Status'], axis=1),
            df['Status'],
            test_size=0.2,
            random_state=50,
            stratify=df['Status']
        )

        numeric_features = [
            'Income',
            'Age',
            'Years_Experience',
            'Total_Family',
        ]

        scaler = StandardScaler()

        X_train_full[numeric_features] = scaler.fit_transform(
            X_train_full[numeric_features]
        )

        X_test[numeric_features] = scaler.transform(X_test[numeric_features])

        return X_train_full, X_test, y_train_full, y_test

    def _train_model(model, x_data, y_data, x_val, y_val, class_weight):

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',   # watch validation accuracy
            patience=10,
            mode='max',
            restore_best_weights=True
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=4,
            mode='max',
            min_lr=1e-6
        )
        history = model.fit(
            x_data,
            y_data,
            epochs=50,
            batch_size=32,
            validation_data=(x_val, y_val),
            class_weight=class_weight,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        return model.get_weights(), history

    X_train, X_test, y_train, y_test = _set_up_poison(VALIDATION_FILE)
    neg, pos = np.bincount(y_train)
    total = neg + pos
    class_weight = {0: (1/neg) * (total / 2.0), 1: (1/pos) * (total / 2.0)}
    weights, history = _train_model(
        model=get_model_with_weights(batch),
        x_data=X_train,
        x_val=X_test,
        y_data=y_train,
        y_val=y_test,
        class_weight=class_weight,
    )
    return weights, {k: v[-1] for k, v in history.history.items()}



def validate(batch, weights):
    os.makedirs(PLOTS_DIR_VAL, exist_ok=True)
    os.makedirs(PLOTS_DIR_POISON_VAL, exist_ok=True)
    X_val_file, y_val_file = _prepare_validation_df(VALIDATION_FILE)
    if X_val_file is None:
        print("No posining evaluation performed (validation file missing or invalid).")
        return

    # Falls y_val aus Training übergeben wurde, wir verwenden hier die explizit gelesene Validation-Datei
    X_val = X_val_file
    y_true = np.asarray(y_val_file).astype(int)

    # Vorhersage-Wahrscheinlichkeiten
    model = get_model_with_weights(batch)
    if weights:
        model.set_weights(weights)

    y_pred_prob = model.predict(X_val, batch_size=64).ravel()
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


def create_poisoned_validation_plots(validation_results):
    os.makedirs(PLOTS_DIR_POISON_VAL, exist_ok=True)
    # DataFrame für die Plots
    df = pd.DataFrame(validation_results)

    # Pro Metrik ein Diagramm
    for metric in METRICS_TO_PLOT:
        df_metric = df[df["metric"] == metric]

        plt.figure(figsize=(10,6))

        # Balkenfarbe
        plt.bar(get_readable_tag_from_batch(df_metric["batch"]), df_metric["value"])

        # Y-Skala "eingezoomt"
        y_min = max(0, df_metric["value"].min() - 0.01)  # kleiner Puffer
        y_max = df_metric["value"].max() + 0.01
        if metric != "loss":
            y_max = min(1.1, df_metric["value"].max() + 0.01)

        plt.ylim(y_min, y_max)

        plt.ylabel(metric.capitalize())
        plt.title(f"Data Poisoning - Beste Variante pro Algorithmus ({metric})")
        plt.xticks(rotation=0)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR_POISON_VAL, f"poisoning_{metric}.png"))
        plt.close()


    print(f"✅ Poisoningplots gespeichert in: {PLOTS_DIR_POISON_VAL}")

def poisoning(batch):

    # test penetration
    weights, history = poison(batch)

    merge_algorithms = {
        "fed_merge": fed_merge, 
        "fed_prox": fed_prox,
        "astraea_merge": astraea_merge, 
        "no_merge": no_merge
    }

    # merge weights
    result_weights = merge_algorithms[batch.split("-")[0]](
        global_weights=retrieve_local_weights(batch),
        local_weights=[weights],
        sample_counts=[8, 2],
        trust_scores=[
            1 / retrieve_local_metrics(batch)["val_loss"],
            1 / history["val_loss"]
        ]
    )

    return validate(batch, result_weights)
