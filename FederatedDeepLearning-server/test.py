import os
import pandas as pd
from glob import glob

# Basisordner (alles klein)
BASE_DIR = "results"

# Alle all-results.csv-Dateien in Unterordnern finden
csv_files = glob(os.path.join(BASE_DIR, "*", "all_results.csv"))

if not csv_files:
    raise ValueError(f"❌ Keine 'all-results.csv'-Dateien im Ordner '{BASE_DIR}' gefunden!")

# Alle CSV-Dateien einlesen und kombinieren
df_list = []
for file in csv_files:
    try:
        temp = pd.read_csv(file)
        temp["source_folder"] = os.path.basename(os.path.dirname(file))
        df_list.append(temp)
    except Exception as e:
        print(f"⚠️ Fehler beim Lesen von {file}: {e}")

data = pd.concat(df_list, ignore_index=True)

# --- Konfiguration ---
metrics = ["accuracy", "loss", "precision", "recall", "auc"]
group_cols = ["algo", "optimizer", "learning_rate", "dropout", "split"]
# ----------------------

# Datentypen sicherstellen
for col in metrics + ["step", "learning_rate", "dropout", "split"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Nur Validation und Poison-Daten behalten
data = data[data["step"].isin([444, 666])].copy()

# Step-Typ zuordnen
data["step_type"] = data["step"].map({444: "validation", 666: "poison"})

# Gruppieren nach Setup + Step-Typ
grouped = data.groupby(group_cols + ["step_type"])

# Statistische Auswertung
stats = grouped[metrics].agg(["mean", "var", "std", "min", "max", "count"])
stats.columns = [f"{metric}_{stat}" for metric, stat in stats.columns]
stats = stats.reset_index()

# Sortieren für Übersicht
stats = stats.sort_values(by=["algo", "optimizer", "learning_rate", "dropout", "split", "step_type"])

# Ergebnisse speichern
output_file = os.path.join(BASE_DIR, "aggregated_validation_poison.csv")
stats.to_csv(output_file, index=False)

print(f"✅ Aggregation abgeschlossen. Datei gespeichert unter: {output_file}")
print(f"Gefundene Kombinationen (Setup + Step-Typ): {len(stats)}\n")

print(stats.head(10))

