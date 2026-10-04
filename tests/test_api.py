# tests/test_api.py
from fastapi.testclient import TestClient
from api.main import app  # Achte darauf, dass der Import-Pfad zu deiner Struktur passt

client = TestClient(app)

def test_predict_success():
    # Gültigen Payload exakt nach dem ShopSession-Schema definieren
    valid_payload = {
        "PageValues": 45.5,
        "ExitRates": 0.02,
        "Total_Clicks": 12,
        "Browser": 1,  # Muss als Integer übergeben werden
        "Region": 1
    }
    
    response = client.post("/predict", json=valid_payload)
    
    assert response.status_code == 200, f"Unerwarteter Statuscode: {response.text}"
    response_data = response.json()
    assert "send_voucher" in response_data
    assert "purchase_probability" in response_data