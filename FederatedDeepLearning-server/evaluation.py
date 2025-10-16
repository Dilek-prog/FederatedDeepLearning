import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# =============================================================================
# 1. Setup
# =============================================================================

INPUT_PATH = "results/aggregated_validation_poison.csv"
OUTPUT_ROOT = "results/full_analysis"

# =============================================================================
# 2. Hilfsfunktionen
# =============================================================================

def make_dirs(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(f"{base_dir}/plots", exist_ok=True)
    os.makedirs(f"{base_dir}/tables", exist_ok=True)

def correlation_analysis(data_encoded, target_name, output_dir, invert=False):
    corr = data_encoded.corr(numeric_only=True)
    corr_target = corr[[target_name]].sort_values(by=target_name, ascending=False)

    if invert:
        corr_target[f"{target_name}_interpreted"] = -corr_target[target_name]
        corr_target = corr_target.sort_values(by=f"{target_name}_interpreted", ascending=False)

    csv_path = f"{output_dir}/tables/correlation_{target_name}.csv"
    corr_target.to_csv(csv_path)

    plt.figure(figsize=(7, len(corr_target) * 0.4 + 1))
    sns.heatmap(corr_target[[target_name]], annot=True, cmap="coolwarm", fmt=".3f", center=0)
    plt.title(f"Korrelation der Hyperparameter mit {target_name}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/correlation_heatmap_{target_name}.png", dpi=300)
    plt.close()

    print(f"✅ Korrelationen für {target_name} gespeichert → {csv_path}")

# =============================================================================
# 3. Analysepipeline pro StepType
# =============================================================================

def run_analysis(df, step_type):
    print(f"\n🚀 Starte Analyse für StepType = {step_type}\n")

    # Verzeichnis vorbereiten
    output_dir = f"{OUTPUT_ROOT}/{step_type.lower()}"
    make_dirs(output_dir)

    # Flag für föderiert
    df["is_federated"] = df["split"] > 1

    # -------------------------------------------------------------------------
    # Vergleich Föderiert vs. Zentral
    # -------------------------------------------------------------------------
    grouped = df.groupby("is_federated")[["accuracy_mean", "loss_mean"]].mean().reset_index()
    grouped["type"] = grouped["is_federated"].map({True: "Federated", False: "Centralized"})

    plt.figure(figsize=(6, 4))
    sns.barplot(data=grouped, x="type", y="accuracy_mean", palette="Blues")
    plt.title(f"Accuracy: Föderiert vs. Zentral ({step_type})")
    plt.xlabel("")
    plt.ylabel("Accuracy (mean)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/accuracy_federated_vs_centralized.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.barplot(data=grouped, x="type", y="loss_mean", palette="Reds")
    plt.title(f"Loss: Föderiert vs. Zentral ({step_type})")
    plt.xlabel("")
    plt.ylabel("Loss (mean)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/loss_federated_vs_centralized.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Accuracy/Loss über Splits
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="split", y="accuracy_mean", hue="algo", marker="o")
    plt.title(f"Accuracy über Splits ({step_type})")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/accuracy_over_splits.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="split", y="loss_mean", hue="algo", marker="o", palette="Reds")
    plt.title(f"Loss über Splits ({step_type})")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/loss_over_splits.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Accuracy/Loss per Algorithm
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="algo", y="accuracy_mean")
    plt.title(f"Accuracy pro Algorithmus ({step_type})")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/accuracy_per_algorithm.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="algo", y="loss_mean", palette="Reds")
    plt.title(f"Loss pro Algorithmus ({step_type})")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/loss_per_algorithm.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Korrelationen (jetzt inkl. Algorithmus)
    # -------------------------------------------------------------------------
    encode_cols = [
        "optimizer", "algo", "learning_rate", "dropout", "split",
        "step_type", "accuracy_mean", "loss_mean"
    ]
    cols_existing = [c for c in encode_cols if c in df.columns]
    data_encoded = pd.get_dummies(df[cols_existing], columns=["optimizer", "algo", "step_type"], drop_first=False)

    correlation_analysis(data_encoded, "accuracy_mean", output_dir, invert=False)
    correlation_analysis(data_encoded, "loss_mean", output_dir, invert=True)

    # -------------------------------------------------------------------------
    # Hyperparameter-Wirkung (Delta Federated vs. Central)
    # -------------------------------------------------------------------------
    corr_fed = df[df["is_federated"]].corr(numeric_only=True)["accuracy_mean"]
    corr_central = df[~df["is_federated"]].corr(numeric_only=True)["accuracy_mean"]

    comparison = pd.DataFrame({
        "federated_corr": corr_fed,
        "central_corr": corr_central,
        "delta": corr_fed - corr_central
    }).dropna().reset_index().rename(columns={"index": "metric"})

    exclude_keywords = ["accuracy", "auc", "recall", "precision"]
    comparison = comparison[~comparison["metric"].str.contains("|".join(exclude_keywords), case=False, na=False)]
    comparison = comparison.sort_values("delta", ascending=False)
    comparison.to_csv(f"{output_dir}/tables/hyperparam_correlation_comparison_filtered.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=comparison, x="metric", y="delta")
    plt.title(f"Delta-Korrelation (Föderiert vs. Zentral, {step_type})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/delta_hyperparam_correlation_filtered.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Statistischer Vergleich
    # -------------------------------------------------------------------------
    if df["split"].nunique() > 1:
        t_stat, p_val = ttest_ind(
            df[df["split"] == 1]["accuracy_mean"],
            df[df["split"] > 1]["accuracy_mean"],
            equal_var=False
        )
        stats_path = f"{output_dir}/tables/statistical_test_results.txt"
        with open(stats_path, "w") as f:
            f.write(f"=== T-Test ({step_type}) ===\n")
            f.write(f"t-Statistik: {t_stat:.4f}\n")
            f.write(f"p-Wert: {p_val:.6f}\n")
            if p_val < 0.05:
                f.write("→ Unterschied ist signifikant (p < 0.05)\n")
            else:
                f.write("→ Unterschied ist NICHT signifikant (p >= 0.05)\n")
        print(f"✅ Statistische Analyse gespeichert → {stats_path}")

    print(f"🎯 Analyse für {step_type} abgeschlossen.")
    print(f"   Ergebnisse unter: {os.path.abspath(output_dir)}\n")

# =============================================================================
# 4. Hauptablauf
# =============================================================================

if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)
    print(f"✅ CSV geladen ({len(df)} Zeilen)")

    step_types = df["step_type"].unique()
    print(f"Gefundene StepTypes: {step_types}")

    for st in step_types:
        sub_df = df[df["step_type"] == st].copy()
        if len(sub_df) < 5:
            print(f"⚠️ Zu wenige Daten für {st}, übersprungen.")
            continue
        run_analysis(sub_df, st)

    # -------------------------------------------------------------------------
    # Vergleich Validation vs. Poison
    # -------------------------------------------------------------------------
    if set(["validation", "poison"]).issubset(set(df["step_type"].unique())):
        print("\n📊 Erstelle Vergleich: Validation vs. Poison pro Algorithmus")

        grouped = (
            df.groupby(["algo", "step_type"])[["accuracy_mean", "loss_mean"]]
            .mean()
            .reset_index()
            .pivot(index="algo", columns="step_type")
        )

        grouped.columns = ["accuracy_validation", "accuracy_poison", "loss_validation", "loss_poison"]
        grouped["accuracy_drop"] = grouped["accuracy_poison"] - grouped["accuracy_validation"]
        grouped["loss_increase"] = grouped["loss_poison"] - grouped["loss_validation"]
        grouped.to_csv(f"{OUTPUT_ROOT}/validation_vs_poison_comparison.csv")

        # Plot: Accuracy-Drop pro Algorithmus
        plt.figure(figsize=(8, 5))
        sns.barplot(data=grouped.reset_index(), x="algo", y="accuracy_drop", palette="coolwarm")
        plt.title("Performance Drop (Poison – Validation) pro Algorithmus (Accuracy)")
        plt.ylabel("Δ Accuracy (negativ = schlechter)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_ROOT}/validation_vs_poison_accuracy_drop.png", dpi=300)
        plt.close()

        # Plot: Loss-Increase pro Algorithmus
        plt.figure(figsize=(8, 5))
        sns.barplot(data=grouped.reset_index(), x="algo", y="loss_increase", palette="Reds")
        plt.title("Loss-Anstieg (Poison – Validation) pro Algorithmus")
        plt.ylabel("Δ Loss (positiv = schlechter)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_ROOT}/validation_vs_poison_loss_increase.png", dpi=300)
        plt.close()

        print(f"✅ Validation-vs-Poison Vergleich gespeichert unter {OUTPUT_ROOT}/")
    else:
        print("⚠️ Kein vollständiger Validation/Poison-Datensatz – Vergleich übersprungen.")

    print("\n✅ Gesamte Analyse abgeschlossen!")
