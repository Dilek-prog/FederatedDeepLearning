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


def qfedavg(local_grads_list, losses, sample_counts, q=0.5, **kwargs):
    total_weight = 0
    merged_grads = [np.zeros_like(g) for g in local_grads_list[0]]

    for grads, loss, count in zip(local_grads_list, losses, sample_counts):
        weight = (loss + 1e-10) ** q * count
        total_weight += weight
        for i, g in enumerate(grads):
            merged_grads[i] += weight * g

    return [g / total_weight for g in merged_grads]


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


def merge_gradients(learning_rate: int, gradients, model, batch_name, **kwargs):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Beispielmodell
    model = get_model(batch_name)

    # Optimizer definieren (kann auch aus dem Model geladen werden)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Beispiel: Forward + Backward Pass
    with tf.GradientTape() as tape:
        preds = model(tf.random.normal((8, 5)))  # Dummy Input
        loss = tf.keras.losses.mean_squared_error(tf.ones_like(preds), preds)
        loss = tf.reduce_mean(loss)

    # Gradienten berechnen
    grads = tape.gradient(loss, model.trainable_variables)

    # Gradienten anwenden
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

def fedavgm(global_weights, local_weights_list, momentum_buffer, beta=0.9):
    """
    global_weights: list of np arrays (model.get_weights())
    local_weights_list: list of list of weights from each client
    momentum_buffer: list of tf.Tensor (same shape as global_weights)
    beta: momentum coefficient
    """

    # Average local weights
    new_avg = []
    for weights in zip(*local_weights_list):
        new_avg.append(tf.reduce_mean(tf.stack(weights), axis=0))

    # Apply momentum
    new_momentum = []
    new_weights = []
    for g, a, m in zip(global_weights, new_avg, momentum_buffer):
        update = a - g
        new_m = beta * m + update
        new_w = g + new_m
        new_momentum.append(new_m)
        new_weights.append(new_w)

    return new_weights, new_momentum

# momentum_buffer = [tf.zeros_like(w) for w in global_model.get_weights()]


def fedadam(global_weights, local_weights_list, m, v, beta1=0.9, beta2=0.999, lr=0.001, epsilon=1e-8, t=1):
    """
    global_weights: list of np arrays
    local_weights_list: list of list of weights from each client
    m, v: lists of tensors (momentum and variance)
    t: round number
    """

    # Step 1: average update
    new_avg = []
    for weights in zip(*local_weights_list):
        new_avg.append(tf.reduce_mean(tf.stack(weights), axis=0))

    # Step 2: compute delta
    updates = [a - g for a, g in zip(new_avg, global_weights)]

    # Step 3: FedAdam update
    new_weights = []
    new_m = []
    new_v = []
    for i in range(len(global_weights)):
        m[i] = beta1 * m[i] + (1 - beta1) * updates[i]
        v[i] = beta2 * v[i] + (1 - beta2) * tf.square(updates[i])

        m_hat = m[i] / (1 - beta1 ** t)
        v_hat = v[i] / (1 - beta2 ** t)

        w_new = global_weights[i] + lr * m_hat / (tf.sqrt(v_hat) + epsilon)

        new_weights.append(w_new)
        new_m.append(m[i])
        new_v.append(v[i])

    return new_weights, new_m, new_v

