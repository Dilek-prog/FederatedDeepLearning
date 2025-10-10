import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Aggregierte CSV laden
data = pd.read_csv("results/aggregated_validation_poison.csv")

# --------------------
# 🔧 Parameter zum Anpassen
# --------------------
metric = "accuracy_mean"   # z. B. "accuracy_mean", "loss_mean", "auc_mean", ...
x_axis = "algo"        # X-Achse: z. B. "optimizer"
y_axis = "split"            # Y-Achse: z. B. "split"
hue_type = "step_type"      # "validation" oder "poison" oder None für beide zusammen
# --------------------

# Nur gewünschte Step-Typen filtern
if hue_type in ["validation", "poison"]:
    plot_data = data[data["step_type"] == hue_type]
else:
    plot_data = data.copy()

# Pivot-Tabelle für Heatmap (eine Zeile pro Kombination)
pivot = plot_data.pivot_table(
    index=y_axis, columns=x_axis, values=metric, aggfunc="mean"
)

plt.figure(figsize=(8, 6))
sns.heatmap(
    pivot,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    cbar_kws={'label': metric},
)

title = f"Heatmap – {metric.replace('_', ' ').title()} ({hue_type if hue_type else 'all'})"
plt.title(title)
plt.ylabel(y_axis.title())
plt.xlabel(x_axis.title())

plt.tight_layout()

# Heatmap speichern
output_path = os.path.join("results", f"heatmap_{metric}_{hue_type or 'all'}.png")
plt.savefig(output_path, dpi=300)
plt.show()

print(f"✅ Heatmap gespeichert unter: {output_path}")


# === Parameter ===
BASE_DIR = "results"
AGG_FILE = os.path.join(BASE_DIR, "aggregated_validation_poison.csv")

metric = "accuracy_mean"   # Metrik, die du vergleichen willst
x_axis = "algo"        # Spalten in der Heatmap
y_axis = "split"            # Zeilen in der Heatmap
# ==================

# Daten laden
df = pd.read_csv(AGG_FILE)

# Sicherstellen, dass die Werte numerisch sind
df[metric] = pd.to_numeric(df[metric], errors="coerce")

# Pivot-Tabelle: eine für Validation, eine für Poison
val = df[df["step_type"] == "validation"]
poison = df[df["step_type"] == "poison"]

# Gleiche Gruppierung für beide
group_cols = ["algo", "optimizer", "learning_rate", "dropout", "split"]

val = val.set_index(group_cols)[metric]
poison = poison.set_index(group_cols)[metric]

# Differenz berechnen: Validation minus Poison
diff = (val - poison).reset_index(name="delta")

# Für Heatmap gewünschte Variablen auswählen
pivot = diff.pivot_table(index=y_axis, columns=x_axis, values="delta", aggfunc="mean")

# --- Plot ---
plt.figure(figsize=(8, 6))
sns.heatmap(
    pivot,
    annot=True,
    fmt=".3f",
    cmap="RdYlBu_r",
    center=0,
    cbar_kws={'label': f"Δ {metric.replace('_', ' ').title()} (Validation − Poison)"}
)

plt.title(f"Poisoning Impact Heatmap – {metric.replace('_', ' ').title()}")
plt.xlabel(x_axis.title())
plt.ylabel(y_axis.title())
plt.tight_layout()

# Datei speichern
out_path = os.path.join(BASE_DIR, f"heatmap_poisoning_delta_{metric}.png")
plt.savefig(out_path, dpi=300)

print(f"✅ Poisoning Impact Heatmap gespeichert unter: {out_path}")