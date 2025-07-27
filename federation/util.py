import os
import json
import tensorflow as tf


SHARED_VOLUME_PATH = os.path.abspath("shared-data")


def get_model(batch_name):
    with open(f"{SHARED_VOLUME_PATH}/{batch_name}/model_architecture.json") as f:
        return tf.keras.models.model_from_json(f.read())


def retrieve_local_weights(batch_name):
    model = get_model(batch_name)
    model.load_weights(f"{SHARED_VOLUME_PATH}/{batch_name}/result.weights.h5")
    return model.get_weights()


def retrieve_global_weights(batch_name):
    model = get_model(batch_name)
    model.load_weights(f"{SHARED_VOLUME_PATH}/global.weights.h5")
    return model.get_weights()


def retrieve_local_metrics(batch_name):
    with open(f"{SHARED_VOLUME_PATH}/{batch_name}/metrics.json") as fp:
        return json.load(fp)


def retrieve_global_metrics():
    with open(f"{SHARED_VOLUME_PATH}/metrics.json") as fp:
        return json.load(fp)
