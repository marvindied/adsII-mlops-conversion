from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.xgboost
import pandas as pd
import os

app = FastAPI(
    title="E-Commerce Conversion API",
    description="API zur Vorhersage der Kaufwahrscheinlichkeit für gezielte Gutschein-Ausspielung.",
    version="1.0.0"
)

# Pfad zum MLflow-Modell (wird beim Start der API einmalig geladen)
# Passe den RUN_ID Ordnernamen an den an, den MLflow bei dir lokal generiert hat!
MODEL_URI = "notebooks/mlruns/1//artifacts/m-2ce91cd0d5cf4a4baddb7da67a2e7c79"

try:
    # Modell in den Speicher laden
    model = mlflow.xgboost.load_model(MODEL_URI)
    print("✅ Modell erfolgreich aus MLflow geladen!")
except Exception as e:
    print(f"❌ Fehler beim Laden des Modells: {e}")

# Pydantic-Schema für die eingehenden API-Daten
class ShopSession(BaseModel):
    Administrative: int
    Administrative_Duration: float
    Informational: int
    Informational_Duration: float
    ProductRelated: int
    ProductRelated_Duration: float
    BounceRates: float
    ExitRates: float
    PageValues: float
    SpecialDay: float
    OperatingSystems: int
    Browser: int
    Region: int
    TrafficType: int
    Weekend: int
    Is_Holiday_Season: int
    Total_Clicks: int
    # Hinweis: One-Hot-Encoded Spalten (wie Month_Feb) müssen hier ebenfalls definiert werden, 
    # passend zu deinem X_train Datensatz.

@app.post("/predict")
def predict_conversion(session: ShopSession):
    try:
        # Daten in ein Pandas DataFrame umwandeln
        input_data = pd.DataFrame([session.dict()])
        
        # Wahrscheinlichkeit vorhersagen (Klasse 1 = Kauf)
        prediction_proba = model.predict_proba(input_data)[0][1]
        
        # Business-Logik: Gutschein ausspielen, wenn Wahrscheinlichkeit zwischen 20% und 80% liegt
        send_voucher = bool(0.20 <= prediction_proba <= 0.80)
        
        return {
            "purchase_probability": float(prediction_proba),
            "send_voucher": send_voucher
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))