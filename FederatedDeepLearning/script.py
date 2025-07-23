import sys
import json
import logging
import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logger = logging.getLogger("FederatedDeepLearning")
logger.setLevel(logging.INFO)

# Handler auf stdout setzen
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=None, help='weights file in .keras or .weights.h5 format')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='learning rate used by the optimizer')
    parser.add_argument('--dropout', type=float, default=0.25, help='dropout used in deep learning')
    parser.add_argument('--training-data', type=str, default=None, help='source for training data')
    parser.add_argument('--batch-name', type=str, default=None, help='batch name for saving results')
    args = parser.parse_args()
    logger.info(args)

    logger.info("Started Script")
    x_train, x_val, y_train, y_val = create_test_data(args.training_data)

    neg, pos = np.bincount(y_train) #Zählt, wie oft jeder Wert in einem Array vorkommt
    total = neg + pos #Gesamtzahl der Trainingsdatenpunkte
    class_weight = {0: (1/neg) * (total / 2.0), 1: (1/pos) * (total / 2.0)} #Berechnung Klassengewichte
    logger.info(f"Berechnete Klassengewichte: {class_weight}")

    logger.info("creating model...")
    model = create_model(feature_count=x_train.shape[1], weights=args.weights, dropout=args.dropout, learning_rate=args.learning_rate)
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
    X_train_full, X_test, y_train_full, y_test = train_test_split(df.drop(['ID', 'Status'], axis=1), df['Status'], test_size=0.2, random_state=50, stratify=df['Status']) #stratify=y Versuche die prozentuale Verteilung der Klassen in y_train und y_test so ähnlich wie möglich zu halten wie in y

    #Skalierung der numerischen Spalten (Ursprüngliche numerische Spalten in df, die nicht entfernt wurden)

    numeric_features = ['Children', 'Income', 'Age', 'Years_Experience', 'Total_Family', 'Good rate']

    scaler = StandardScaler()

    #Numerische Spalten im Trainingsset fitten und transformieren
    X_train_full[numeric_features] = scaler.fit_transform(X_train_full[numeric_features])

    # Die gleichen Statistiken des Trainingsset verwenden um das Testset zu transformieren
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])

    return X_train_full, X_test, y_train_full, y_test

def create_model(feature_count: int, weights: list[int] = None, dropout: int = 0.25, learning_rate: int = 0.001):
    model = keras.Sequential(name="CreditModel") #Aufeinanderfolgen von verschiedenen Schichten "Dense", Daten fließen von erster bis zur letzten Ausgabeschicht
    model.add(keras.Input(shape=(feature_count,)))

    # Schicht 1
    model.add(keras.layers.Dense(128, activation='relu')) #gängige Praxis Kombination von RelU und Sigmoid: in den "hidden"-layers: 
    #es löst kritische Probleme der Gradienten Verteilung und Recheneffizienz, Sigmoid interpretiert die Ergebnisse für binäre Klassifikationsprobleme als Wahrscheinlichkeiten
    model.add(keras.layers.BatchNormalization()) #BatchNormalization um Konvergenz zu verbessern --> Eingaben jeder Schicht wird auf einen Mittelwert von 0 und Std-Abweichung von 1 normalisiert.
    model.add(keras.layers.Dropout(dropout))

    # Schicht 2

    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(dropout))

    # Schicht 3

    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(dropout))

    # Ausgabeschicht
    model.add(keras.layers.Dense(1, activation='sigmoid')) #Sollte nicht lieber sigmoid als Aktivierungsfunktion benutzt werden und nicht softmax?
    # wir wollen ja nur 2 mögliche Klassen ausgeben, bei SOftmax wird eine Liste von Wahrscheinlichkeiten produziert, die sich zu 1 summieren

    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate), #AdamOptimizer
        loss = 'binary_crossentropy', #Loss-Funktion für binäre Klassifikation bei Sigmoid
        metrics=[
            'accuracy', #Grundlegende Genauigkeit
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC()
        ]
    )
    if weights:
        model.load_weights(weights)
    return model

def train_model(model, x_data, y_data, x_val, y_val, class_weight):

    early_stopping = tf.keras.callbacks.EarlyStopping(#Beendet das Training, wenn der Validierungs-Loss sich nicht mehr verbessert
        monitor = 'val_loss', #Überwacht den Validierungs-loss
        patience = 10, #ANzahl der Epochen ohne Verbesserung, bevor gestoppt wird
        restore_best_weights=True, #Wiederherstellung der besten Modellgewichte der besten Epoche
        verbose=1 #Anzeigen, wann Early Stopping greift
    )
    return model.fit(x_data, y_data,
                        epochs=50, #hohe Anzahl von Epochen, aber das Early Stopping soll das tatsächliche Ende regeln
                        batch_size=32, #Anzahl der Trainingsbeispiele pro Update des Modells
                        validation_data = (x_val, y_val), #Validierungsdaten für Training
                        class_weight=class_weight,
                        callbacks=[early_stopping], #einsetzten EarlyStopping
                        verbose=1 #Anzeigen Trainingsfortschritt
                        )

def save_results(model, batch_name: str, x_val, y_val, history):
    os.makedirs(f"/shared-data/{batch_name}", exist_ok=True)
    model.save_weights(f"/shared-data/{batch_name}/result.weights.h5")

    x_val_numeric = x_val.copy()
    for col in x_val_numeric.select_dtypes(include=['bool']).columns:
        x_val_numeric[col] = x_val_numeric[col].astype(np.float32)
    x_val_numeric = x_val_numeric.astype(np.float32)

    loss_fn = tf.keras.losses.BinaryCrossentropy()
    with tf.GradientTape() as tape:
        predictions = model(x_val_numeric)
        loss = loss_fn(y_val, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    for idx, grad in enumerate(gradients):
        if grad is not None:
            np.save(f"/shared-data/{batch_name}/gradient_{idx}.npy", grad.numpy())

    last_metrics = {k: v[-1] for k, v in history.history.items()}
    with open(f"/shared-data/{batch_name}/metrics.json", "w") as f:
        json.dump(last_metrics, f)

    model_json = model.to_json()
    with open(f"/shared-data/{batch_name}/model_architecture.json", "w") as f:
        f.write(model_json)

if __name__ == "__main__":
    main()