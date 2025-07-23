import math
import shutil
import time
import os
import argparse

import docker
from sklearn.model_selection import ParameterGrid
import pandas as pd
from tensorflow.keras.models import model_from_json

from merger import *

client = docker.from_env()
shared_volume_path = os.path.abspath("shared-data")

# Stelle sicher, dass das shared volume Verzeichnis existiert
os.makedirs(shared_volume_path + "/training_data", exist_ok=True)

current_dir = os.path.dirname(os.path.abspath(__file__))

def run_container(name, image, volumes, command, **kwargs):
    print(f"🟢 Starte {name} ...")
    container = client.containers.run(
        image=image,
        name=name,
        command=command,
        detach=True,
        remove=True,
        volumes=volumes,
        stdout=True,
        stderr=True,
        tty=False, 
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
        sample_df.to_csv(f"{shared_volume_path}/training_data/training_data_{i}.csv", index=False, sep=";")

def get_model(batch_name):
    with open(f"{shared_volume_path}/{batch_name}/model_architecture.json") as f:
        return model_from_json(f.read())

def retrieve_new_weights(batch_name):
    model = get_model(batch_name)
    model.load_weights(f"{shared_volume_path}/{batch_name}/result.weights.h5")
    return model.get_weights()

def retrieve_global_weights(batch_name):
    model = get_model(batch_name)
    model.load_weights(f"{shared_volume_path}/global.weights.h5")
    return model.get_weights()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, default=None, help='source for your training data')
    parser.add_argument('--split', type=int, default=None, help='choose in how many chunks you want to split the dataset')
    args = parser.parse_args()

    print("🔨 Building image ...")
    client.images.build(path="./FederatedDeepLearning", tag="federated-deep-learning:0.0.0", rm=True)
    print("finished building image")

    volumes = {
        shared_volume_path: {'bind': '/shared-data', 'mode': 'rw'}
    }

    # preparing training data
    prep_data(
        file=args.filepath,
        splits=args.split,
        random_state=1
    )

    options = ParameterGrid({
        'merge_algortihm': [fed_merge],
        'learning_rate': [0.001, 0.0001],
        'dropout': [0.2, 0.3, 0.4],
    })

    for option in options:
        weight_location = ""
        for federation_step in range(args.split):
            batch_name = f"{option["merge_algortihm"].__name__}-{option["learning_rate"]}-{option["dropout"]}_{federation_step}"
            run_container(
                name=f"federated-deep-learning_{federation_step}",
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
                shutil.copy(f"{shared_volume_path}/{batch_name}/result.weights.h5", f"{shared_volume_path}/global.weights.h5")
                weight_location = f"/shared-data/global.weights.h5"
                continue
            
            result_weights = fed_merge(weights=retrieve_global_weights(batch_name), new_weights=retrieve_new_weights(batch_name=batch_name), weight=(1 / federation_step))
            model = get_model(batch_name)
            model.set_weights(result_weights)
            model.save_weights(f"{shared_volume_path}/global.weights.h5")

if __name__ == "__main__":
    main()
