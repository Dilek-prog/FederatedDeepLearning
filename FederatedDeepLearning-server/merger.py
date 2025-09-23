import tensorflow as tf
import numpy as np

from util import get_model


def weighted_average(weights_list, sample_counts, **kwargs):
    total_samples = np.sum(sample_counts)
    averaged_weights = []

    for weights in zip(*weights_list):
        weighted = np.sum(
            [
                w * n
                for w, n
                in zip(weights, sample_counts)
            ],
            axis=0
        ) / total_samples
        averaged_weights.append(weighted)

    return averaged_weights


def fed_avg(global_weights, local_weights, sample_counts, **kwargs):
    return weighted_average([global_weights, local_weights], sample_counts)


def fed_prox(global_weights, local_weights, sample_counts, mu=0.01, **kwargs):
    prox_weights_list = []

    for local_weights in local_weights:
        prox_weights = []
        for gw, lw in zip(global_weights, local_weights):
            prox_layer = lw - mu * (lw - gw)
            prox_weights.append(prox_layer)
        prox_weights_list.append(prox_weights)

    return weighted_average(prox_weights_list, sample_counts)


def fed_merge(global_weights, local_weights, sample_counts, **kwargs):
    total_samples = sum(sample_counts)
    merged_weights = []
    for layers in zip(global_weights, *local_weights):

        global_layer = layers[0]
        local_layers = layers[1:]

        merged_layer = global_layer.copy()
        for lw, n in zip(local_layers, sample_counts):
            merged_layer += (n / total_samples) * (lw - global_layer)
        merged_weights.append(merged_layer)
    return merged_weights


def normalize_scores(scores, eps=1e-8):
    scores = np.maximum(scores, eps)  # Stabilität
    return scores / np.sum(scores)


def astraea_merge(local_weights, trust_scores, **kwargs):
    """Astraea-Merge: Vertrauen (Trust) basiert auf lokaler Performance."""
    trust_weights = normalize_scores(np.array(trust_scores))  # z. B. 1/val_loss

    aggregated_weights = []
    for layers in zip(*local_weights):
        weighted_sum = sum(w * t for w, t in zip(layers, trust_weights))
        aggregated_weights.append(weighted_sum)

    return aggregated_weights


def no_merge(local_weights, **kwargs):
    return local_weights[0]
