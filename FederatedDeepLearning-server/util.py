from collections import defaultdict
import math
import os
import json
import re
import pandas as pd
import tensorflow as tf

from config import METRICS_TO_PLOT, RESULTS_DIR, SHARED_VOLUME_PATH


def get_model_with_weights(step_name):
    model = get_model(step_name)
    model.load_weights(f"{SHARED_VOLUME_PATH}/{step_name}/result.weights.h5")
    return model

def get_model(batch_name):
    with open(f"{SHARED_VOLUME_PATH}/{batch_name}/model_architecture.json") as f:
        return tf.keras.models.model_from_json(f.read())


def retrieve_local_weights(step_name):
    model = get_model(step_name)
    model.load_weights(f"{SHARED_VOLUME_PATH}/{step_name}/result.weights.h5")
    return model.get_weights()


def retrieve_global_weights(batch_name, step_name):
    model = get_model(step_name)
    model.load_weights(f"{SHARED_VOLUME_PATH}/weight_exchange/{batch_name}.weights.h5")
    return model.get_weights()


def retrieve_local_metrics(batch_name):
    with open(f"{SHARED_VOLUME_PATH}/{batch_name}/metrics.json") as fp:
        return json.load(fp)


def retrieve_global_metrics(batch_name):
    with open(f"{SHARED_VOLUME_PATH}/weight_exchange/{batch_name}.json") as fp:
        return json.load(fp)

def check_if_step_exists(batchname: str):
    return os.path.exists(f"{SHARED_VOLUME_PATH}/{batchname}")


def prep_data(file: str, splits: list[int], random_state: int, validation_size: float = 0.2):
    import pandas as pd
    import math
    import os

    # Daten einlesen
    df = pd.read_csv(file, delimiter=";")

    # Shuffle für Reproduzierbarkeit
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # 80/20 Split in Training/Validation
    val_count = int(len(df_shuffled) * validation_size)
    val_df = df_shuffled.iloc[:val_count]
    train_df = df_shuffled.iloc[val_count:]

    # Validation speichern (einmalig)
    val_path = f"{SHARED_VOLUME_PATH}/training_data/validation_data.csv"
    if not os.path.exists(val_path):
        val_df.to_csv(val_path, index=False, sep=";")

    # Trainingsdaten in Splits aufteilen
    for split in splits:
        split_size = math.ceil(len(train_df) / split)
        for i in range(split):
            start = i * split_size
            end = start + split_size
            sample_df = train_df.iloc[start:end]

            name = f"{SHARED_VOLUME_PATH}/training_data/training_data_{split}_{i}.csv"
            if os.path.exists(name):
                continue
            sample_df.to_csv(name, index=False, sep=";")

def find_best_variant(variants, metric):
    best_variant = None
    best_final_value = None
    for variant, step_data in variants.items():
        steps_sorted = sorted(step_data.keys())
        if not steps_sorted:
            continue

        # Für Bestenauswahl nur den letzten Wert berücksichtigen
        final_step = steps_sorted[-1]
        final_value = step_data[final_step].get(metric)

        if final_value is None or final_value == 1:
            continue

        if best_final_value is None or (
            metric == "loss" and final_value < best_final_value
        ) or (
            metric != "loss" and final_value > best_final_value
        ):
            best_final_value = final_value
            best_variant = variant
    return best_variant


def get_all_data():
    # Datensammlung: {algo: {variant: {step: {metric: value}}}}
    all_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    # Neuer Regex: algo-optimizer-lr-dropout-split-step
    folder_pattern = re.compile(
        r"(?P<algo>[^-]+)-(?P<optimizer>[^-]+)-(?P<lr>[\d.]+)-(?P<dropout>[\d.]+)-(?P<split>\d+)-(?P<step>\d+)"
    )

    for folder in os.listdir(RESULTS_DIR):
        match = folder_pattern.match(folder)
        if not match:
            continue

        algo = match.group("algo")
        optimizer = match.group("optimizer")
        lr = match.group("lr")
        dropout = match.group("dropout")
        split = match.group("split")
        step = int(match.group("step"))

        # Variant string enthält jetzt optimizer & split auch
        variant = f"opt={optimizer}|lr={lr}|dropout={dropout}|split={split}"

        metrics_path = os.path.join(RESULTS_DIR, folder, "metrics.json")
        if not os.path.exists(metrics_path):
            continue

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        for metric in METRICS_TO_PLOT:
            value = metrics.get(metric)
            if value is not None:
                all_data[algo][variant][step][metric] = value
    return all_data


def get_params_by_variant(variant:str) -> dict:
    """
        Extrahiert optimizer, lr, dropout und split aus einem String
        wie 'opt=adam|lr=0.001|dropout=0.25|split=3'.
        Gibt ein Dict mit den Werten zurück.
    """
    pattern = re.compile(
        r"opt=(?P<opt>[^|]+)\|lr=(?P<lr>[^|]+)\|dropout=(?P<dropout>[^|]+)\|split=(?P<split>\d+)"
    )
    match = pattern.match(variant)
    if not match:
        return {}
    return match.groupdict()


def get_readable_tag_from_batch(batch):
    pattern = re.compile(
        r"(?P<algo>[^-]+)-(?P<optimizer>[^-]+)-(?P<lr>[\d.]+)-(?P<dropout>[\d.]+)-(?P<split>\d+)-(?P<step>\d+)"
    )
    return [
        f"{(m := pattern.match(x)).group('algo')}"
        f"\nopt={m.group('optimizer')}"
        f"\nlr={m.group('lr')}"
        f"\ndropout={m.group('dropout')}"
        f"\nsplit={m.group('split')}"
        for x in batch if (m := pattern.match(x))
    ]

