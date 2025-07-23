import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import re

RESULTS_DIR = os.path.abspath("shared-data") # Hauptordner mit *_step Ordnern
METRICS_TO_PLOT = ["accuracy", "loss", "precision", "recall", "auc"]
GLOBAL_MARKERS_FILE = os.path.join(RESULTS_DIR, "global_markers.json")
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Datensammlung: {verfahren: {step: {metric: value}}}
all_metrics = defaultdict(lambda: defaultdict(dict))

# Ordner durchgehen
for folder in os.listdir(RESULTS_DIR):
    match = re.match(r"(?P<method>.+)_(?P<step>\d+)", folder)
    if not match:
        continue

    method = match.group("method")
    step = int(match.group("step"))
    metrics_path = os.path.join(RESULTS_DIR, folder, "metrics.json")

    if not os.path.exists(metrics_path):
        continue

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    for metric in METRICS_TO_PLOT:
        value = metrics.get(metric)
        if value is not None:
            all_metrics[method][step][metric] = value

# Globale Marker laden (falls vorhanden)
global_markers = {}
if os.path.exists(GLOBAL_MARKERS_FILE):
    with open(GLOBAL_MARKERS_FILE, "r") as f:
        global_markers = json.load(f)

# Plots erstellen
for metric in METRICS_TO_PLOT:
    plt.figure(figsize=(10, 6))
    for method, step_data in all_metrics.items():
        steps = sorted(step_data.keys())
        values = [step_data[step].get(metric, None) for step in steps]
        if any(v is not None for v in values):
            plt.plot(steps, values, marker="o", label=method)

    plt.title(f"Verlauf von {metric.capitalize()} über Iterationen")
    plt.xlabel("Iteration")
    plt.ylabel(metric.capitalize())
    plt.axhline(y=global_markers[metric], color='gray', linestyle='--', label='Non-Federated')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{metric}_timeline.png"))
    plt.close()

print(f"✅ Alle Plots gespeichert in '{PLOTS_DIR}' mit globalen Markierungen.")
