from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import math
import shutil
import os
import argparse
import signal
import sys
import time

from tqdm import tqdm
import docker
from sklearn.model_selection import ParameterGrid

from merger import fed_merge, no_merge, fed_prox, astraea_merge
from util import check_if_step_exists, get_model, prep_data, retrieve_global_metrics, \
    retrieve_global_weights, retrieve_local_metrics, retrieve_local_weights
from config import SHARED_VOLUME_PATH

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


client = docker.from_env()

os.makedirs(SHARED_VOLUME_PATH + "/training_data", exist_ok=True)
os.makedirs(SHARED_VOLUME_PATH + "/weight_exchange", exist_ok=True)

executor = None

def shutdown(signum, frame):
    global executor
    tqdm.write("⚠️ Abbruch erkannt, stoppe Executor...")
    if executor:
        executor.shutdown(wait=False, cancel_futures=True)
    sys.exit(1)

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

parser = argparse.ArgumentParser()
parser.add_argument('--filepath', type=str, default=None, help='source for your training data')
parser.add_argument('--splits', type=int, nargs='+', default=None, help='list of splits you want to train on')
parser.add_argument('--verbose', type=int, default=1, help='Set level of verbosity')
args = parser.parse_args()


def run_container(name, image, volumes, command, **kwargs):
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
        user=1000,
        # runtime="nvidia",
        # device_requests=[
        #     docker.types.DeviceRequest(
        #         count=-1,
        #         capabilities=[['gpu']]
        #     )
        # ],
        environment={
            "TF_CPP_MIN_LOG_LEVEL": "2"
        }, # reduces the huge amounts of log messages by tf and cuda
        **kwargs
    )
    if args.verbose > 1:
        for line in container.logs(stream=True):
            tqdm.write(line.decode("utf-8").strip())
    container.wait()
    return container


def run_option(option, volumes):
    weight_location = ""
    for federation_step in range(option["split"]):
        batch_name = f"{option["merge_algortihm"].__name__}-{option["optimizer"]}-{option["learning_rate"]}-{option["dropout"]}-{option["split"]}"
        step_name = f"{batch_name}-{federation_step}"

        if check_if_step_exists(step_name):
            if args.verbose > 0:
                tqdm.write(f"Batch {step_name} already done")
            continue

        if args.verbose > 0:
            tqdm.write(f"Running {step_name} ...")

        run_container(
            name=f"federated-deep-learning_{step_name}",
            image="federated-deep-learning:0.0.0",
            volumes=volumes,
            command=[
                "python", "script.py",
                "--weights", weight_location,
                "--learning-rate", str(option["learning_rate"]),
                "--dropout", str(option["dropout"]),
                "--training-data", f"/shared-data/training_data/training_data_{option["split"]}_{federation_step}.csv",
                "--batch-name", step_name,
                "--optimizer", option["optimizer"]
            ]
        )
        if not weight_location:
            # first step of learning 
            shutil.copy(
                f"{SHARED_VOLUME_PATH}/{step_name}/result.weights.h5",
                f"{SHARED_VOLUME_PATH}/weight_exchange/{batch_name}.weights.h5"
            )
            shutil.copy(
                f"{SHARED_VOLUME_PATH}/{step_name}/metrics.json",
                f"{SHARED_VOLUME_PATH}/weight_exchange/{batch_name}.json"
            )
            weight_location = f"/shared-data/weight_exchange/{batch_name}.weights.h5"
            continue

        result_weights = option["merge_algortihm"](
            global_weights=retrieve_global_weights(batch_name, step_name),
            local_weights=[retrieve_local_weights(step_name)],
            sample_counts=[federation_step, 1],
            trust_scores=[
                1 / retrieve_global_metrics(batch_name).get("val_loss", 1),
                1 / retrieve_local_metrics(step_name)["val_loss"]
            ]
        )

        shutil.copy(
                f"{SHARED_VOLUME_PATH}/{step_name}/metrics.json",
                f"{SHARED_VOLUME_PATH}/weight_exchange/{batch_name}.json"
            )

        model = get_model(step_name)
        model.set_weights(result_weights)
        model.save_weights(f"{SHARED_VOLUME_PATH}/weight_exchange/{batch_name}.weights.h5")

    return f"Batch {option} abgeschlossen"


def main():

    # tqdm.write("🔨 Building image ...")
    # client.images.build(path="./FederatedDeepLearning-client", tag="federated-deep-learning:0.0.0", rm=True)
    # tqdm.write("finished building image")

    VOLUMES = {
        SHARED_VOLUME_PATH: {'bind': '/shared-data', 'mode': 'rw'}
    }

    # preparing training data
    prep_data(
        file=args.filepath,
        splits=args.splits,
        random_state=1
    )

    options = ParameterGrid({
        'merge_algortihm': [fed_merge, fed_prox, astraea_merge, no_merge],
        'learning_rate': [0.1, 0.001, 0.0001],
        'dropout': [0.2, 0.3, 0.4],
        'split': args.splits,
        'optimizer': ["adam", "sdg", "adadelta"]
    })

    start = time.time()
    with tqdm(total=len(options), desc="Training Process") as pbar:
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(run_option, option, VOLUMES) for option in options]

            for future in as_completed(futures):
                result = future.result()
                pbar.update(1)
                tqdm.write(result)

    end = time.time()
    tqdm.write(f"Laufzeit: {end - start:.2f} Sekunden")


if __name__ == "__main__":
    main()
