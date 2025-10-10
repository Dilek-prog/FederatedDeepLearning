import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===================================
# CONFIG
# ===================================
BASE_DIR = "results"
OUTPUT_DIR = os.path.join(BASE_DIR, "plots_best_hyperparams_fancy")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================================
# LOAD ALL CSVs
# ===================================
all_files = glob.glob(os.path.join(BASE_DIR, "*", "all_results.csv"))

if not all_files:
    raise FileNotFoundError("❌ Keine 'all_results.csv'-Dateien gefunden!")

df_list = []
for file in all_files:
    tmp = pd.read_csv(file)
    tmp["run_id"] = os.path.basename(os.path.dirname(file))  # Durchlauf-ID
    df_list.append(tmp)

df = pd.concat(df_list, ignore_index=True)

# Lowercase columns
df.columns = [c.strip().lower() for c in df.columns]

# Konvertiere numerische Spalten
for col in ["step", "accuracy", "loss", "split", "learning_rate", "dropout"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Nur Trainingsschritte (step != 444, 666)
df_train = df[~df["step"].isin([444, 666])].copy()

# ===================================
# FIND BEST HYPERPARAMS
# ===================================
best_rows = []

for algo, group in df.groupby("algo"):
    # Best Accuracy
    acc_means = group.groupby(["optimizer", "learning_rate", "dropout", "split"])["accuracy"].mean()
    best_acc_combo = acc_means.idxmax()

    # Best Loss
    loss_means = group.groupby(["optimizer", "learning_rate", "dropout", "split"])["loss"].mean()
    best_loss_combo = loss_means.idxmin()

    best_rows.append({"algo": algo, "metric": "accuracy", "optimizer": best_acc_combo[0],
                      "learning_rate": best_acc_combo[1], "dropout": best_acc_combo[2], "split": best_acc_combo[3]})
    best_rows.append({"algo": algo, "metric": "loss", "optimizer": best_loss_combo[0],
                      "learning_rate": best_loss_combo[1], "dropout": best_loss_combo[2], "split": best_loss_combo[3]})

best_df = pd.DataFrame(best_rows)

# ===================================
# FANCY SEABORN TRAINING PLOTS
# ===================================
sns.set(style="whitegrid", palette="Set2", font_scale=1.1)

for _, row in best_df.iterrows():
    algo = row["algo"]
    metric = row["metric"]
    opt = row["optimizer"]
    lr = row["learning_rate"]
    drop = row["dropout"]
    split = row["split"]

    subset = df_train[
        (df_train["algo"] == algo) &
        (df_train["optimizer"] == opt) &
        (df_train["learning_rate"] == lr) &
        (df_train["dropout"] == drop) &
        (df_train["split"] == split)
    ].copy()

    if subset.empty:
        print(f"⚠️ Keine Trainingsdaten für {algo} - {metric} gefunden. Überspringe.")
        continue

    # Prozentualer Fortschritt
    subset["progress_pct"] = (subset["step"] + 1) / split * 100

    plt.figure(figsize=(9, 5))
    ax = sns.lineplot(
        data=subset,
        x="progress_pct",
        y=metric,
        hue="run_id",
        marker="o",
        errorbar="sd",   # ±1 Std als shaded area (statt ci="sd")
        palette="tab10",
        linewidth=2
    )


    ax.set_title(f"{algo} – {metric.title()} Progression\n"
                 f"({opt}, LR={lr}, Dropout={drop}, Split={split})", fontsize=14)
    ax.set_xlabel("Training Progress (%)", fontsize=12)
    ax.set_ylabel(metric.title(), fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 10))
    ax.legend(title="Run ID", bbox_to_anchor=(1.05, 1), loc='upper left')
    sns.despine(trim=True)
    plt.tight_layout()

    # Speichern
    filename = f"{algo}_{metric}_progress_fancy.png".replace(" ", "_").lower()
    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"💾 Fancy Plot gespeichert unter: {out_path}")

print("\n🎯 Alle fancy Trainingsplots erfolgreich erstellt!")
