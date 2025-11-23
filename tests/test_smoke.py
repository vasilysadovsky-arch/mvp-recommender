from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json().get("ok") is True

def test_topn():
    r = client.get("/topN?mode=content&fair=0&k=5&user_id=u_001")
    assert r.status_code == 200
    js = r.json()
    assert "items" in js and len(js["items"]) == 5
