import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def main():
    x_train, x_val, y_train, y_val = create_test_data()

    #Berechnung der Klassengewichte
    neg, pos = np.bincount(y_train) #Zählt, wie oft jeder Wert in einem Array vorkommt
    total = neg + pos #Gesamtzahl der Trainingsdatenpunkte
    class_weight = {0: (1/neg) * (total / 2.0), 1: (1/pos) * (total / 2.0)} #Berechnung Klassengewichte
    print(f"Berechnete Klassengewichte: {class_weight}")

    model = create_model(feature_count=x_train.shape[1])
    history = train_model(
        model, 
        x_data=x_train, 
        y_data=y_train,
        x_val=x_val,
        y_val=y_val,
        class_weight=class_weight,
    )

def create_test_data():
    df = pd.read_csv('./data.csv')
    X_train_full, X_test, y_train_full, y_test = train_test_split(df.drop(['ID', 'Status']), df['Status'], test_size=0.2, random_state=50, stratify=y) #stratify=y Versuche die prozentuale Verteilung der Klassen in y_train und y_test so ähnlich wie möglich zu halten wie in y

    #Skalierung der numerischen Spalten (Ursprüngliche numerische Spalten in df, die nicht entfernt wurden)

    numeric_features = ['Children', 'Income', 'Age', 'Years_Experience', 'Total_Family', 'Good rate']

    scaler = StandardScaler()

    #Numerische Spalten im Trainingsset fitten und transformieren
    X_train_full[numeric_features] = scaler.fit_transform(X_train_full[numeric_features])

    # Die gleichen Statistiken des Trainingsset verwenden um das Testset zu transformieren
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])


    #Aufteilung des X_train_full in echtes Training und Validierung
    #test_size = 25% von X_train_full

    return train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=50, stratify=y_train_full
    )

def create_model(feature_count: int):
    model = keras.Sequential(name="CreditModel") #Aufeinanderfolgen von verschiedenen Schichten "Dense", Daten fließen von erster bis zur letzten Ausgabeschicht
    model.add(keras.Input(shape=(feature_count,)))

    # Schicht 1
    model.add(keras.layers.Dense(128, activation='relu')) #gängige Praxis Kombination von RelU und Sigmoid: in den "hidden"-layers: 
    #es löst kritische Probleme der Gradienten Verteilung und Recheneffizienz, Sigmoid interpretiert die Ergebnisse für binäre Klassifikationsprobleme als Wahrscheinlichkeiten
    model.add(keras.layers.BatchNormalization()) #BatchNormalization um Konvergenz zu verbessern --> Eingaben jeder Schicht wird auf einen Mittelwert von 0 und Std-Abweichung von 1 normalisiert.
    model.add(keras.layers.Dropout(0.25))

    # Schicht 2

    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.25))

    # Schicht 3

    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.25))

    # Ausgabeschicht
    model.add(keras.layers.Dense(1, activation='sigmoid')) #Sollte nicht lieber sigmoid als Aktivierungsfunktion benutzt werden und nicht softmax?
    # wir wollen ja nur 2 mögliche Klassen ausgeben, bei SOftmax wird eine Liste von Wahrscheinlichkeiten produziert, die sich zu 1 summieren

    initial_learning_rate = 0.001
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=initial_learning_rate), #AdamOptimizer
        loss = 'binary_crossentropy', #Loss-Funktion für binäre Klassifikation bei Sigmoid
        metrics=[
            'accuracy', #Grundlegende Genauigkeit
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC()
        ]
    )
    try:
        model.load_weights(weights_filename)
        model.summary()
    except tf.errors.NotFoundError:
        print(f"Fehler: Die Gewichtsdatei '{weights_filename}' wurde nicht gefunden.")
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
    return model

def train_model(model, x_data, y_data, x_val, y_val, class_weight):

    early_stopping = tf.keras.callbacks.EarlyStopping(#Beendet das Training, wenn der Validierungs-Loss sich nicht mehr verbessert
        monitor = 'val_loss', #Überwacht den Validierungs-loss
        patience = 10, #ANzahl der Epochen ohne Verbesserung, bevor gestoppt wird
        restore_best_weights=True, #Wiederherstellung der besten Modellgewichte der besten Epoche
        verbose=1 #Anzeigen, wann Early Stopping greift
    )
    print("n\Starte Modelltraining...")
    history = model.fit(x_data, y_data,
                        epochs=50, #hohe Anzahl von Epochen, aber das Early Stopping soll das tatsächliche Ende regeln
                        batch_size=32, #Anzahl der Trainingsbeispiele pro Update des Modells
                        validation_data = (x_val, y_val), #Validierungsdaten für Training
                        class_weight=class_weight,
                        callbacks=[early_stopping], #einsetzten EarlyStopping
                        verbose=1 #Anzeigen Trainingsfortschritt
                        )
    print("Modelltraining abgeschlossen")
    # save histroy.history 
    return history

if __name__ == "__script__":
    main()