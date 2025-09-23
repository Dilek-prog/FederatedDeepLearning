import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import re

from util import find_best_variant, get_all_data
from config import METRICS_TO_PLOT, PLOTS_DIR_ALL, PLOTS_DIR_BEST, RESULTS_DIR


os.makedirs(PLOTS_DIR_ALL, exist_ok=True)
os.makedirs(PLOTS_DIR_BEST, exist_ok=True)

GLOBAL_MARKERS_FILE = os.path.join(RESULTS_DIR, "global_markers.json")

all_data = get_all_data()

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

        best_variant = find_best_variant(variants, metric)

        # Beste Variante plotten
        if best_variant:
            step_data = all_data[algo][best_variant]
            steps_sorted = sorted(step_data.keys())
            max_step = len(steps_sorted)   # letzter Step für diesen Algo

            # Prozentuale Darstellung bezogen auf den max Step
            x = [((s + 1) / max_step) * 100 for s in steps_sorted]
            y = [step_data[s].get(metric, 0) for s in steps_sorted]
            plt.plot(x, y, marker='o', label=f"{algo} (best: {best_variant})")

    # Global Marker falls vorhanden
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
