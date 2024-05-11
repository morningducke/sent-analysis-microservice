from fastapi.testclient import TestClient

from .main import app

client = TestClient(app)
positive_text = "i like movies"
negative_text = "i hate rain"
neutral_text = "i am not sure how i feel"

def test_predict():
    postive_response = client.post("/predict", params={"to_analyse" : positive_text})
    assert postive_response.status_code == 200
    assert postive_response.json()["label"] == "positive"
    
    negative_response = client.post("/predict", params={"to_analyse" : negative_text})
    assert negative_response.status_code == 200
    assert negative_response.json()["label"] == "negative"
    
    neutral_response = client.post("/predict", params={"to_analyse" : neutral_text})
    assert neutral_response.status_code == 200
    assert neutral_response.json()["label"] == "neutral"
    