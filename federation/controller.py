import math
import shutil
import os
import argparse
import time

import docker
from sklearn.model_selection import ParameterGrid
import pandas as pd

from merger import fed_merge, qfedavg, fed_prox, astraea_merge
from util import SHARED_VOLUME_PATH, get_model, retrieve_global_metrics, \
    retrieve_global_weights, retrieve_local_metrics, retrieve_local_weights


client = docker.from_env()


os.makedirs(SHARED_VOLUME_PATH + "/training_data", exist_ok=True)

current_dir = os.path.dirname(os.path.abspath(__file__))


def run_container(name, image, volumes, command, **kwargs):
    print(f"🟢 Starte {name} ...")
    container = client.containers.run(
        image=image,
        name=name,
        runtime="nvidia",
        command=command,
        detach=True,
        remove=True,
        volumes=volumes,
        stdout=True,
        stderr=True,
        tty=False,
        user=1000,
        device_requests=[
            docker.types.DeviceRequest(
                count=-1,
                capabilities=[['gpu']]
            )
        ],
        **kwargs
    )
    for line in container.logs(stream=True):
        print(line.decode("utf-8").strip())
    container.wait()
    print(f"✅ {name} beendet")
    return container


def prep_data(file: str, splits: float, random_state: int):
    df = pd.read_csv(file, delimiter=";")
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Calculate size per split
    split_size = math.ceil(len(df) / splits)

    # Save each sample to a separate CSV file
    for i in range(splits):
        start = i * split_size
        end = start + split_size
        sample_df = df_shuffled.iloc[start:end]
        sample_df.to_csv(f"{SHARED_VOLUME_PATH}/training_data/training_data_{i}.csv", index=False, sep=";")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, default=None, help='source for your training data')
    parser.add_argument('--split', type=int, default=None, help='choose in how many chunks you want to split the dataset')
    args = parser.parse_args()

    print("🔨 Building image ...")
    client.images.build(path="./FederatedDeepLearning", tag="federated-deep-learning:0.0.0", rm=True)
    print("finished building image")

    volumes = {
        SHARED_VOLUME_PATH: {'bind': '/shared-data', 'mode': 'rw'}
    }

    # preparing training data
    prep_data(
        file=args.filepath,
        splits=args.split,
        random_state=1
    )

    options = ParameterGrid({
        'merge_algortihm': [fed_merge, fed_prox, astraea_merge],
        'learning_rate': [0.001, 0.0001],
        'dropout': [0.2, 0.3, 0.4],
    })

    start = time.time()

    for option in options:
        weight_location = ""
        for federation_step in range(args.split):
            batch_name = f"{option["merge_algortihm"].__name__}-{option["learning_rate"]}-{option["dropout"]}-{federation_step}"
            run_container(
                name=f"federated-deep-learning_{batch_name}",
                image="federated-deep-learning:0.0.0",
                volumes=volumes,
                command=[
                    "python", "script.py",
                    "--weights", weight_location,
                    "--learning-rate", str(option["learning_rate"]),
                    "--dropout", str(option["dropout"]),
                    "--training-data", f"/shared-data/training_data/training_data_{federation_step}.csv",
                    "--batch-name", batch_name,
                ]
            )
        
            if not weight_location:
                shutil.copy(
                    f"{SHARED_VOLUME_PATH}/{batch_name}/result.weights.h5",
                    f"{SHARED_VOLUME_PATH}/global.weights.h5"
                )
                weight_location = "/shared-data/global.weights.h5"
                continue

            shutil.copy(
                    f"{SHARED_VOLUME_PATH}/{batch_name}/metrics.json",
                    f"{SHARED_VOLUME_PATH}/metrics.json"
                )
            
            result_weights = option["merge_algortihm"](
                global_weights=retrieve_global_weights(batch_name),
                local_weights=[retrieve_local_weights(batch_name)],
                sample_counts=[federation_step, 1],
                trust_scores=[
                    1 / retrieve_global_metrics()["val_loss"],
                    1 / retrieve_local_metrics(batch_name)["val_loss"]
                ]
            )
            model = get_model(batch_name)
            model.set_weights(result_weights)
            model.save_weights(f"{SHARED_VOLUME_PATH}/global.weights.h5")
    end = time.time()
    print(f"Laufzeit: {end - start:.2f} Sekunden")


if __name__ == "__main__":
    main()
