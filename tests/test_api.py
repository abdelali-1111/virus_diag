from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)


def test_root():
    assert client.get("/").status_code == 200


def test_diagnose():
    r = client.post("/diagnose", json={
        "age": 70, "temperature": 39.2, "cough": 1,
        "oxygen_saturation": 89, "crp": 85})
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["diagnosis"] in ("Infecté", "Sain")