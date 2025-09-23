import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import re

RESULTS_DIR = os.path.abspath("shared-data")
PLOTS_DIR_ALL = "plots/all_variants"
PLOTS_DIR_BEST = "plots/best_variants"
os.makedirs(PLOTS_DIR_ALL, exist_ok=True)
os.makedirs(PLOTS_DIR_BEST, exist_ok=True)

METRICS_TO_PLOT = ["accuracy", "loss", "precision", "recall", "auc"]
GLOBAL_MARKERS_FILE = os.path.join(RESULTS_DIR, "global_markers.json")

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

# Globale Marker laden (optional)
global_markers = {}
if os.path.exists(GLOBAL_MARKERS_FILE):
    with open(GLOBAL_MARKERS_FILE, "r") as f:
        global_markers = json.load(f)

# Plot 1: Alle Varianten eines Algorithmus
for metric in METRICS_TO_PLOT:
    for algo, variants in all_data.items():
        plt.figure(figsize=(10, 6))
        for variant, step_data in variants.items():
            steps_sorted = sorted(step_data.keys())
            if not steps_sorted:
                continue
            max_step = max(steps_sorted) + 1
            x = [(s / max_step) * 100 for s in steps_sorted]
            y = [step_data[s].get(metric, 0) for s in steps_sorted]
            plt.plot(x, y, marker='o', label=f"{variant}")

        if metric in global_markers:
            plt.axhline(y=global_markers[metric], color='gray', linestyle='--', label='Global Marker')

        plt.title(f"{metric.capitalize()} – alle Varianten von {algo}")
        plt.xlabel("Genutzte Daten (%)")
        plt.ylabel(metric.capitalize())
        plt.xticks(range(0, 101, 10))
        plt.grid(True)
        plt.legend(fontsize="small")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR_ALL, f"{algo}_{metric}_all_variants.png"))
        plt.close()

# Plot 2: Beste Variante je Algorithmus
for metric in METRICS_TO_PLOT:
    plt.figure(figsize=(10, 6))
    for algo, variants in all_data.items():
        best_variant = None
        best_final_value = None
        for variant, step_data in variants.items():
            steps_sorted = sorted(step_data.keys())
            if not steps_sorted:
                continue
            final_value = step_data[steps_sorted[-1]].get(metric)
            if final_value is not None:
                if best_final_value is None or (
                    metric == "loss" and final_value < best_final_value
                ) or (
                    metric != "loss" and final_value > best_final_value
                ):
                    best_final_value = final_value
                    best_variant = variant

        if best_variant:
            step_data = all_data[algo][best_variant]
            steps_sorted = sorted(step_data.keys())
            max_step = max(steps_sorted) + 1
            x = [(s / max_step) * 100 for s in steps_sorted]
            y = [step_data[s].get(metric, 0) for s in steps_sorted]
            plt.plot(x, y, marker='o', label=f"{algo} (best: {best_variant})")

    if metric in global_markers:
        plt.axhline(y=global_markers[metric], color='gray', linestyle='--', label='Global Marker')

    plt.title(f"{metric.capitalize()} – beste Variante pro Algorithmus")
    plt.xlabel("Genutzte Daten (%)")
    plt.ylabel(metric.capitalize())
    plt.xticks(range(0, 101, 10))
    plt.grid(True)
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR_BEST, f"{metric}_best_variants.png"))
    plt.close()

print(f"✅ Plots gespeichert in:\n📂 {PLOTS_DIR_ALL} (alle Varianten)\n📂 {PLOTS_DIR_BEST} (beste Varianten)")
