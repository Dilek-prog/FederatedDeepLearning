import sys
import json
import logging
import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("FederatedDeepLearning")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='weights file in .keras or .weights.h5 format'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='learning rate used by the optimizer'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.25,
        help='dropout used in deep learning'
    )
    parser.add_argument(
        '--training-data',
        type=str,
        default=None,
        help='source for training data'
    )
    parser.add_argument(
        '--batch-name',
        type=str,
        default=None,
        help='batch name for saving results'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default=None,
        help='optimizer function used for deep learning'
    )
    args = parser.parse_args()
    logger.info(args)

    logger.info("Started Script")
    x_train, x_val, y_train, y_val = create_test_data(args.training_data)

    neg, pos = np.bincount(y_train)
    total = neg + pos
    class_weight = {0: (1/neg) * (total / 2.0), 1: (1/pos) * (total / 2.0)}
    logger.info(f"Berechnete Klassengewichte: {class_weight}")


    optimizer_dict = {
        "adam": tf.keras.optimizers.Adam,
        "sdg": tf.keras.optimizers.SGD,
        "adadelta": tf.keras.optimizers.Adadelta
    }
    logger.info("creating model...")
    model = create_model(
        feature_count=x_train.shape[1],
        weights=args.weights,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        optimizer=optimizer_dict[args.optimizer]
    )
    logger.info("finished creating model")

    logger.info("training model...")
    history = train_model(
        model,
        x_data=x_train,
        y_data=y_train,
        x_val=x_val,
        y_val=y_val,
        class_weight=class_weight,
    )
    logger.info("finished training")
    logger.info("saving results...")
    save_results(
        model=model,
        batch_name=args.batch_name,
        x_val=x_val,
        y_val=y_val,
        history=history
    )
    logger.info("finished saving")
    logger.info("federated learning is done")


def create_test_data(source: str):
    df = pd.read_csv(source, delimiter=";")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        df.drop(['ID', 'Status'], axis=1),
        df['Status'],
        test_size=0.2,
        random_state=50,
        stratify=df['Status']
    )

    numeric_features = [
        'Children',
        'Income',
        'Age',
        'Years_Experience',
        'Total_Family',
    ]

    scaler = StandardScaler()

    X_train_full[numeric_features] = scaler.fit_transform(
        X_train_full[numeric_features]
    )

    X_test[numeric_features] = scaler.transform(X_test[numeric_features])

    return X_train_full, X_test, y_train_full, y_test


def create_model(
    feature_count: int,
    weights: list[int] = None,
    dropout: int = 0.25,
    learning_rate: int = 0.001,
    optimizer = tf.keras.optimizers.Adam
):
    model = tf.keras.Sequential(name="CreditModel")
    model.add(tf.keras.Input(shape=(feature_count,)))

    # Schicht 1
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout))

    # Schicht 2

    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout))

    # Schicht 3

    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout))

    # Ausgabeschicht
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=optimizer(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC(),
            'binary_crossentropy'
        ]
    )
    if weights:
        model.load_weights(weights)
    return model


def train_model(model, x_data, y_data, x_val, y_val, class_weight):

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',   # watch validation accuracy
        patience=10,
        mode='max',
        restore_best_weights=True
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=4,
        mode='max',
        min_lr=1e-6
    )

    return model.fit(
        x_data,
        y_data,
        epochs=50,
        batch_size=32,
        validation_data=(x_val, y_val),
        class_weight=class_weight,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )


def save_results(model, batch_name: str, x_val, y_val, history):
    os.makedirs(f"/shared-data/{batch_name}", exist_ok=True)
    model.save_weights(f"/shared-data/{batch_name}/result.weights.h5")

    # x_val_numeric = x_val.copy()
    # for col in x_val_numeric.select_dtypes(include=['bool']).columns:
    #     x_val_numeric[col] = x_val_numeric[col].astype(np.float32)
    # x_val_numeric = x_val_numeric.astype(np.float32)

    # loss_fn = tf.keras.losses.BinaryCrossentropy()
    # with tf.GradientTape() as tape:
    #     predictions = model(x_val_numeric)
    #     loss = loss_fn(y_val, predictions)
    # gradients = tape.gradient(loss, model.trainable_variables)
    # for idx, grad in enumerate(gradients):
    #     if grad is not None:
    #         np.save(
    #             f"/shared-data/{batch_name}/gradient_{idx}.npy", grad.numpy()
    #         )

    last_metrics = {k: v[-1] for k, v in history.history.items()}
    with open(f"/shared-data/{batch_name}/metrics.json", "w") as f:
        json.dump(last_metrics, f)

    model_json = model.to_json()
    with open(f"/shared-data/{batch_name}/model_architecture.json", "w") as f:
        f.write(model_json)


if __name__ == "__main__":
    main()
