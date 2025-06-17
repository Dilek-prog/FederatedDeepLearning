# Federated Machine Learning

Dataset: https://www.kaggle.com/code/heshamhammam/credit-approval-detection#Data-Merging <br>
How to install Tensorflow on Jupyter Notebook: https://www.youtube.com/watch?v=rbsWdaZYahE

## Projektübersicht:

Dieses Projekt konzentriert sich auf die Entwicklung und Implementierung eines Neuronalen Netzes zur Bewertung des Kreditrisikos. Als fortführendes Ziel wird die Überführung des Modells in eine Federated Learning Architektur angestrebt, um datenschutzkonformes Training über verteilte Datenquellen hinweg zu ermöglichen.

Das aktuelle Repository enthält das Kern-Modell und die notwendigen Skripte zur Datenvorbereitung sowie zur Demonstration der Modellpersistenz.

## Inhalte des Repositories:

- `FederatedMachineLearning.ipynb`: Das Jupyter Notebook, das die Hauptentwicklung des Modells enthält.

Das Notebook deckt folgende Bereiche ab:

1.  **Robuste Datenvorverarbeitung:**

    - Import und Bereinigung von Rohdaten.
    - Feature Engineering (z.B. Umwandlung von `Age`, `Years_Experience`).
    - Ableitung der Zielvariable (`Credit Status`) und Berechnung der "Good Rate".
    - One-Hot-Encoding für kategorische Features.
    - Standardisierung/Normalisierung von numerischen Features.
    - Aufteilung der Daten in Trainings-, Validierungs- und Testsets mit Stratifizierung, um die Klassenverteilung zu erhalten.

2.  **Künstliches Neuronales Netz (KNN) für die binäre Klassifikation:**

    - Definition einer Multi-Layer Perceptron (MLP) Architektur mit `Dense`, `Dropout` und `BatchNormalization` Schichten.
    - Verwendung von `Adam`-Optimierer und `Binary Crossentropy` als Verlustfunktion.
    - Training des Modells unter Berücksichtigung von **Klassengewichten** (für unausgewogene Daten) und **Early Stopping** (zur Vermeidung von Overfitting).

3.  **Umfassende Modell-Evaluierung:**

    - Berechnung und Anzeige von Verlust (Loss), Genauigkeit (Accuracy), Präzision (Precision), Recall und AUC (Area Under the Curve) auf den Testdaten.
    - Visuelle Darstellung der **Konfusionsmatrix** und des **Klassifikationsberichts** zur detaillierten Analyse der Modellleistung.
    - Plotten der Lernkurven (Loss und Accuracy über Epochen).

4.  **Vorbereitung für Federated Learning (Modellpersistenz):**
    - Demonstration, wie die **gelernten Modellgewichte** (`.weights.h5`-Datei) gespeichert werden können. Dies simuliert den Schritt, den ein FL-Client nach seinem lokalen Training zum Senden der Updates ausführt.
    - Anleitung zum **Laden dieser Gewichte** in eine neu initialisierte Modellarchitektur (`client_model`). Dies bildet ab, wie ein FL-Client die globalen Gewichte vom Server empfängt.
    - Beispiel für das **"Nachtrainieren" (Fine-Tuning)** des geladenen Modells auf einem Datensatz, was den lokalen Trainingsschritt eines Clients im FL-Prozess simuliert.

## Technologien

- **Python**
- **TensorFlow / Keras:** Für den Aufbau und das Training des Neuronalen Netzes.
- **scikit-learn:** Für Datenvorverarbeitung (Splitting, Skalierung) und Metriken.
- **Pandas:** Für Datenmanipulation und -analyse.
- **NumPy:** Für numerische Operationen.
- **Matplotlib / Seaborn:** Für Datenvisualisierung.
- **Keras Tuner (optional):** Für zukünftiges Hyperparameter-Tuning.

## Erste Schritte mit dem Notebook

1. **Repository Klonen**

   ```bash
   git clone https://github.com/Dilek-prog/FederatedDeepLearning
   cd <YOUR_REPO_NAME>
   ```

2. **Virtuelle Umgebung einrichten (empfohlen):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Unter Windows: venv\Scripts\activate
   ```
3. **Abhängigkeiten installieren:**

   ```bash
   pip install -r requirements.txt
   # Falls keine requirements.txt vorhanden, installiere manuell:
   # pip install pandas numpy matplotlib tensorflow scikit-learn seaborn keras-tuner
   ```

4. **Jupyter Notebook starten:**
   ```bash
   jupyter notebook
   ```
5. Öffne `FederatedMachineLearning.ipynb` und führe die Zellen der Reihe nach aus.
