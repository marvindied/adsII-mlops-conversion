from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_read_index():
    response = client.get("/")
    assert response.status_code == 200
    assert "Gutschein-Ausspielung" in response.text

def test_predict_valid_session():
    valid_payload = {
        "Administrative": 0,
        "Administrative_Duration": 0.0,
        "Informational": 0,
        "Informational_Duration": 0.0,
        "ProductRelated": 10,
        "ProductRelated_Duration": 200.0,
        "BounceRates": 0.0,
        "ExitRates": 0.05,
        "PageValues": 15.0,
        "SpecialDay": 0.0,
        "OperatingSystems": 1,
        "Browser": 1,
        "Region": 1,
        "TrafficType": 1,
        "Weekend": 0,
        "Is_Holiday_Season": 1,
        "Total_Clicks": 25
    }
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert "purchase_probability" in data
    assert "send_voucher" in data
    assert "voucher_value" in data

def test_predict_invalid_data():
    invalid_payload = {"Total_Clicks": "text_statt_zahl"}
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # HTTP 422 Validation Error