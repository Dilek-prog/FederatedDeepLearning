# Federated Machine Learning

Dataset: https://www.kaggle.com/code/heshamhammam/credit-approval-detection#Data-Merging
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

## Federation Test Setup

Das Federation Setup dient der automatisierten durchführung von Modelltrainings mit einem 
Satz von Hyperparametern auf einem Prozentsatz der Trainingsdaten und mehreren durchläufen.
Das Validieren und das Durchführen eines Poisoning Attacks sind ebenfalls Teil des Scripts.
Es gibt außerdem eine automatische Erstellung von Evaluationsmetriken.

### Voraussetzungen

Es wird eine lokale Docker-Instanz und uv als Packetmanager benötigt.

### Installation

Zum Installieren des Python environments wird UV benutzt. Nutze

    uv sync

zum installieren aller dependencies.

### Starten des Trainingsprozesses

    uv run FederatedDeepLearning-server/controller.py [args]

Mögliche Argumente sind:
- filepath: Gibt den Pfad zum Trainingsdatensatz an.
- splits: Gibt an in wie viele Teile der Trainingsdatensatz unterteilt werden soll für das Training. Nimmt eine Liste entgegen.
- iteration: Gibt an wie oft das Training wiederholt werden soll um vergleichbare Ergebnisse zu erzeugen.

Beispielsweise könnte der Startbefehl so aussehen:

    uv run FederatedDeepLearning-server/controller.py --filepath training_data_new.csv --splits 1 5 10 --iterations 5

### Funktionsweise

Das Setup besteht aus einem Controller und mehreren Docker-Containern, die jeweils ein Deep-Learning-Skript ausführen.

#### Controller-Logik:

- Der Controller startet für jede Konfiguration mehrere Container.

- Jeder Container erhält einen Teil des Datensatzes (Split) zum Trainieren.

- Die Anzahl der Splits (z. B. 1er, 5er oder 10er) bestimmt, in wie viele Teile der Datensatz aufgeteilt wird – und somit, wie viele Föderationsschritte stattfinden.

#### Training & Föderation

- Für jede Hyperparameter-Kombination wird jeder Split separat trainiert.

- Bei mehrstufigen Splits (z. B. 5er) werden die Modelle nacheinander trainiert; jedes erhält die Gewichte des vorherigen.

- Nach Abschluss aller Föderationen erfolgt eine Validierung auf einem separaten Datensatz sowie ein Poisoning-Test.

#### Mehrfache Ausführung & Aggregation

- Um die Ergebnisse vergleichbar zu machen, kann das komplette Setup mehrfach ausgeführt werden.

- Alle Modelle und Ergebnisse werden versioniert im history/-Ordner abgelegt.

- Der aggregate-results.py-Prozess fasst alle Resultate zusammen, mittelt sie und erzeugt eine aggregated-validation-poison.csv.

#### Evaluation

- Das Skript evaluation.py erstellt automatisch Tabellen und Visualisierungen, um die aggregierten Ergebnisse auszuwerten.

