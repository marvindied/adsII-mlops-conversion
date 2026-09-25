# MLOps Project: E-Commerce Conversion Prediction

Dieses Repository enthält den praktischen Prototypen für das Modul MLOps. 
Ziel des Projekts ist es, die Kaufwahrscheinlichkeit von Onlineshop-Besuchern in Echtzeit vorherzusagen, um Rabattgutscheine gezielt und profitmaximierend an "unentschlossene" Kunden auszuspielen.

## Projektstruktur
- `data/`: Beinhaltet den Kaggle-Datensatz (lokal, nicht versioniert).
- `src/`: Skripte für Datenvorbereitung und Modelltraining (inkl. MLflow).
- `api/`: FastAPI-Anwendung zur Bereitstellung des Modells.
- `tests/`: Automatisierte Unit-Tests.
- `.github/workflows/`: CI/CD Pipeline für automatisiertes Testing.


-------------------------------------------     NEW      ----------------------------------------------------

# MLOps E-Commerce Conversion Predictor

Ein lokal lauffähiger MLOps-Prototyp zur Vorhersage von Kaufwahrscheinlichkeiten in einem Onlineshop, um gezielt 10%-Gutscheine auszuspielen. 

Dieses Projekt wurde im Rahmen des Moduls "MLOps" (M.Sc. Data Science & Management) entwickelt. Es demonstriert zentrale MLOps-Prinzipien wie reproduzierbare Datenverarbeitung, Experiment Tracking, CI/CD und API-Deployment.

## 1. Voraussetzungen und Abhängigkeiten
* **Python:** Version 3.10 oder höher.
* **Paketmanager:** `pip`
* **Abhängigkeiten:** Alle benötigten Bibliotheken (wie `xgboost`, `mlflow`, `fastapi`, `pytest`) sind in der `requirements.txt` definiert.

*Hinweis für Umgebungen mit strikten Firewalls (SSL-Zertifikatsfehler):*
Nutze bei der Installation lokaler Pakete das `--trusted-host` Flag:
`pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org`

## 2. Lokale Einrichtung & Datenverarbeitung
1. Repository klonen und in das Verzeichnis wechseln.
2. Virtuelle Umgebung erstellen und aktivieren (optional aber empfohlen).
3. Abhängigkeiten installieren: `pip install -r requirements.txt`
4. Die Rohdaten (`online_shoppers_intention.csv`) liegen im Ordner `data/`.
5. Das Jupyter Notebook `notebooks/01_EDA_and_Preprocessing.ipynb` ausführen. Dies bereinigt die Daten, führt das Feature Engineering (z.B. `Is_Holiday_Season`) durch und speichert den bereiten Datensatz als `processed_online_shoppers.csv`.

## 3. Modelltraining & Experiment Tracking (MLflow)
* Das Training findet im Notebook `notebooks/02_Baseline.ipynb` statt.
* Wir starten mit einer Logistischen Regression (Baseline) und steigern die Komplexität zu einem XGBoost-Modell, um die Klassenimbalance (nur 15% Käufer) besser zu handhaben.
* **MLflow Tracking:** Das Skript loggt automatisch Hyperparameter, Metriken (F1-Score, ROC-AUC) und das finale Modellartefakt im lokalen Verzeichnis `mlruns/`.

## 4. Lokale Bereitstellung (FastAPI)
Das Champion-Modell (XGBoost) wird als Echtzeit-Schnittstelle bereitgestellt. 
1. Starte den Uvicorn-Server aus dem Hauptverzeichnis:
   `python -m uvicorn api.main:app --reload`
2. Öffne die Swagger-UI im Browser unter `http://127.0.0.1:8000/docs`
3. Über den Endpunkt `/predict` können nun Live-Wahrscheinlichkeiten für die Gutscheinausspielung berechnet werden.

## 5. Testing & CI/CD (GitHub Actions)
* **Tests:** Grundlegende API- und Integrationstests befinden sich im Ordner `tests/`. Sie können lokal mit `pytest tests/` ausgeführt werden.
* **CI/CD Pipeline:** Im Ordner `.github/workflows/ci-pipeline.yml` ist eine automatisierte Pipeline definiert. Bei jedem Push auf den `main`-Branch baut GitHub Actions die Umgebung auf und führt die Tests automatisiert aus, um sicherzustellen, dass fehlerhafter Code das Deployment nicht bricht.