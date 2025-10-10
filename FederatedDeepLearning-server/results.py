import csv
import os
from config import METRICS_TO_PLOT, RESULT_DATA_DIR
from util import get_all_data


def save_results(iteration: int):
    OUTPUT_CSV = os.path.join(RESULT_DATA_DIR, str(iteration), "all_results.csv")
    os.makedirs(os.path.join(RESULT_DATA_DIR, str(iteration)), exist_ok=True)
    # CSV-Header: Hyperparameter + Step + alle Metriken
    header = ["algo", "optimizer", "learning_rate", "dropout", "split", "step"] + METRICS_TO_PLOT

    with open(OUTPUT_CSV, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for algo, variants in get_all_data().items():
            for variant, step_data in variants.items():
                # Variant zerlegen: opt=adam|lr=0.001|dropout=0.3|split=3
                parts = dict(part.split("=") for part in variant.split("|"))

                optimizer = parts.get("opt")
                lr = parts.get("lr")
                dropout = parts.get("dropout")
                split = parts.get("split")

                for step, metrics in step_data.items():
                    row = [
                        algo,
                        optimizer,
                        float(lr),
                        float(dropout),
                        int(split),
                        int(step)
                    ]
                    for metric in METRICS_TO_PLOT:
                        row.append(metrics.get(metric, None))
                    writer.writerow(row)

    print(f"✅ CSV gespeichert: {OUTPUT_CSV}")
