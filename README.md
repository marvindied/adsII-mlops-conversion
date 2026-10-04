# MLOps E-Commerce Conversion Predictor

Ein lokal lauffähiger MLOps-Prototyp zur Vorhersage von Kaufwahrscheinlichkeiten in einem Onlineshop, um gezielt 10%- und 20%-Gutscheine auszuspielen. 

Dieses Projekt wurde im Rahmen des Moduls "MLOps" (M.Sc. Data Science & Management) entwickelt. Es demonstriert zentrale MLOps-Prinzipien wie reproduzierbare Datenverarbeitung, Experiment Tracking, CI/CD und API-Deployment.

## 1. Voraussetzungen und Abhängigkeiten
- **Python:** Version 3.10 oder höher.
- **Paketmanager:** `pip` und optional `make` für Automatisierungen.
- **Abhängigkeiten:** Alle benötigten Bibliotheken (wie `xgboost`, `mlflow`, `fastapi`, `pytest`) sind in der `requirements.txt` definiert.

*Hinweis für Umgebungen mit strikten Firewalls (SSL-Zertifikatsfehler):*
Nutze bei der Installation lokaler Pakete das `--trusted-host` Flag:
`pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org`

## 2. Quickstart (mit Makefile)
Für die einfachste Ausführung der gesamten Pipeline steht ein Makefile zur Verfügung. 
Führe im Terminal im Hauptverzeichnis nacheinander folgende Befehle aus:
1. `make install` (Installiert alle benötigten Abhängigkeiten)
2. `make pipeline` (Führt Preprocessing, Modelltraining und Tests vollautomatisiert aus)
3. `make run-api` (Startet den Webserver und das Frontend)

## 3. Manuelle Einrichtung & Pipeline-Schritte

### Schritt 1: Reproduzierbare Datenverarbeitung (Preprocessing)
Die Rohdaten (`online_shoppers_intention.csv`) liegen im Ordner `data/`.
Um das Feature Engineering (z.B. `Is_Holiday_Season` und `Total_Clicks`) automatisiert durchzuführen, starte das Skript:
`python src/preprocess.py`
Dies erstellt den bereinigten Datensatz `processed_online_shoppers.csv`. (Die ursprüngliche explorative Datenanalyse ist zur Dokumentation weiterhin unter `notebooks/01_EDA_and_Preprocessing.ipynb` zu finden).

### Schritt 2: Modelltraining & Experiment Tracking (MLflow)
Wir trainieren ein Baseline-Modell (Logistische Regression) und ein Champion-Modell (XGBoost), um die Klassenimbalance (nur 15% Käufer) optimal zu bewältigen.
Führe das Training aus:
`python src/train.py`
Das Skript loggt automatisch Hyperparameter, Metriken (F1-Score, ROC-AUC) und das finale Modellartefakt lokal in eine SQLite-Datenbank (`mlflow.db`) sowie in den Ordner `mlruns/`.

### Schritt 3: Lokale Bereitstellung (FastAPI & Frontend)
Das Champion-Modell wird über FastAPI als Echtzeit-Schnittstelle bereitgestellt. 
1. Starte den Uvicorn-Server aus dem Hauptverzeichnis (falls nicht schon per `make run-api` geschehen):
   `python -m uvicorn api.main:app --reload`
2. **Web-Frontend:** Öffne den interaktiven Gutschein-Simulator im Browser unter [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
3. **API-Doku:** Die Swagger-UI für Entwickler ist unter [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) erreichbar.

## 4. Testing & CI/CD (GitHub Actions)
- **Tests:** Grundlegende Unit- und Integrationstests (für API und Vorhersagelogik) befinden sich im Ordner `tests/`. Sie können lokal mit `pytest tests/` ausgeführt werden.
- **CI/CD Pipeline:** Im Ordner `.github/workflows/ci-pipeline.yml` ist eine automatisierte Continuous-Integration-Pipeline definiert. Bei jedem Push oder Pull Request auf den `main`-Branch baut GitHub Actions die Umgebung auf und führt die Tests automatisiert aus. Dies stellt sicher, dass fehlerhafter Code das Projekt nicht unbemerkt beschädigt.