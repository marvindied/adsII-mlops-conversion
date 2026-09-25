from fastapi.testclient import TestClient
from api.main import app

# Einen Test-Client für unsere FastAPI-Anwendung erstellen
client = TestClient(app)

def test_api_starts_successfully():
    """
    Ein grundlegender Test, der prüft, ob die API hochfährt
    und die Metadaten korrekt gesetzt sind.
    """
    assert app.title == "E-Commerce Conversion API"
    assert app.version == "1.0.0"

def test_prediction_endpoint_exists():
    """
    Prüft, ob der Endpunkt für die Vorhersage existiert 
    (erwartet Fehler 422, weil wir keine Daten mitschicken, 
    aber das beweist, dass der Endpunkt da ist!).
    """
    response = client.post("/predict")
    assert response.status_code == 422