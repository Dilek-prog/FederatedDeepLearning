import tensorflow as tf
from tensorflow import keras


def merge_gradients(learning_rate: int, gradients, model):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def merge_weights():
    pass

def fed_avg(weights, new_weights):
    result_weights = []
    for params in zip(weights, new_weights):
        result_weights.append((sum(params) / len(params)))
    return result_weights

def fed_merge(weights, new_weights, weight):
    result_weights = []
    for w_old, w_new in zip(weights, new_weights):
        merged = (1 - weight) * w_old + weight * w_new
        result_weights.append(merged)
    return result_weights
