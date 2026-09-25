import base64
import glob
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import mlflow.xgboost # Nativer XGBoost-Wrapper
import xgboost as xgb # Für die DMatrix
import pandas as pd

app = FastAPI(
    title="E-Commerce Conversion API",
    description="API zur Vorhersage der Kaufwahrscheinlichkeit für gezielte Gutschein-Ausspielung.",
    version="1.0.0",
)

def find_model_path():
    search_pattern = os.path.join("**", "MLmodel")
    matches = glob.glob(search_pattern, recursive=True)
    if matches:
        return os.path.dirname(matches[0])
    raise FileNotFoundError("Keine 'MLmodel'-Datei im Repository gefunden.")

try:
    model_path = find_model_path()
    # Lade das native XGBoost-Modell statt des generischen pyfunc-Wrappers!
    model = mlflow.xgboost.load_model(model_path)
    print(f"✅ XGBoost-Modell nativ geladen aus: {model_path}")
except Exception as e:
    print(f"⚠️ Warnung beim Laden des Modells: {e}")
    model = None

class ShopSession(BaseModel):
    Administrative: int = 0
    Administrative_Duration: float = 0.0
    Informational: int = 0
    Informational_Duration: float = 0.0
    ProductRelated: int = 10
    ProductRelated_Duration: float = 200.0
    BounceRates: float = 0.0
    ExitRates: float = 0.05
    PageValues: float = 0.0
    SpecialDay: float = 0.0
    OperatingSystems: int = 1
    Browser: int = 1
    Region: int = 1
    TrafficType: int = 1
    Weekend: int = 0
    Is_Holiday_Season: int = 1
    Total_Clicks: int = 25
    Month_Dec: int = 0
    Month_Feb: int = 0
    Month_Jul: int = 0
    Month_June: int = 0
    Month_Mar: int = 0
    Month_May: int = 0
    Month_Nov: int = 0
    Month_Oct: int = 0
    Month_Sep: int = 0
    VisitorType_Other: int = 0
    VisitorType_Returning_Visitor: int = 1

@app.get("/", response_class=HTMLResponse)
def index():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>File not found</h1>"

@app.post("/predict")
def predict(session: ShopSession):
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Modell konnte beim Serverstart nicht geladen werden."
        )
    try:
        # 1. Normales DataFrame erstellen (keine DMatrix!)
        input_df = pd.DataFrame([session.dict()])
        
        # 2. Wahrscheinlichkeiten mit predict_proba abrufen
        # Das liefert ein Array wie z.B. [[0.35, 0.65]] (Wahrsch. für Klasse 0 und 1)
        prob_array = model.predict_proba(input_df)
        
        # 3. Wir greifen auf die Wahrscheinlichkeit für Klasse 1 (Kauf) zu
        prob = float(prob_array[0][1])
        
        # 4. Logik auswerten
        send_voucher = bool(0.20 <= prob <= 0.80)
        return {"purchase_probability": prob, "send_voucher": send_voucher}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))