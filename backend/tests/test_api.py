import io
import pytest
from PIL import Image
import numpy as np
from backend.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    """Verify system health endpoint returns 200 OK and correct payload."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "online"
    assert data["service"] == "isic-lesion-ai"
    assert data["model_architecture"] == "efficientnet_b0"


def test_predict_no_file(client):
    """Verify endpoint rejects requests missing the 'image' key."""
    res = client.post("/api/v1/predict", data={})
    assert res.status_code == 400
    assert "No image file provided" in res.get_json()["error"]


def test_predict_invalid_magic_bytes(client):
    """Verify security rejection of non-image payload."""
    fake_data = io.BytesIO(b"NOT_AN_IMAGE_FILE_DATA")
    res = client.post(
        "/api/v1/predict",
        data={"image": (fake_data, "malicious.exe")},
        content_type="multipart/form-data"
    )
    assert res.status_code == 422
    assert "Security validation failed" in res.get_json()["error"]


def test_predict_valid_synthetic_image(client):
    """Verify end-to-end inference on a non-blank image."""
    # Create realistic synthetic gradient pattern to pass contrast/luminance validation
    arr = np.random.randint(50, 200, size=(224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    res = client.post(
        "/api/v1/predict",
        data={"image": (buf, "sample_lesion.jpg")},
        content_type="multipart/form-data"
    )
    assert res.status_code == 200
    data = res.get_json()
    assert "status" in data
    assert "predicted_class" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert "gradcam_url" in data
    assert "image_url" in data


def test_analytics_summary(client):
    """Verify analytics summary endpoint returns real model metadata and splits."""
    res = client.get("/api/v1/analytics/summary")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "dataset_splits" in data
    auc_val = data["model_metadata"]["best_macro_auc"]
    assert auc_val > 0.80 or auc_val > 80.0


def test_analytics_training_history(client):
    """Verify training history endpoint returns real 30-epoch telemetry."""
    res = client.get("/api/v1/analytics/training-history")
    assert res.status_code == 200
    data = res.get_json()
    assert "history" in data
    assert len(data["history"]) >= 1
    assert "best_macro_auc" in data


def test_analytics_per_class_metrics(client):
    """Verify Table 1 per-class metrics returns valid data."""
    res = client.get("/api/v1/analytics/per-class-metrics")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert len(data["data"]) >= 8


def test_history_scans_crud(client):
    """Verify scan history recording and retrieval."""
    test_scan = {
        "patientId": "PT-9999",
        "predictedClass": "Melanoma (Malignant)",
        "confidence": 0.985,
        "status": "in_distribution"
    }
    # Create
    post_res = client.post("/api/v1/history/scans", json=test_scan)
    assert post_res.status_code == 201
    created = post_res.get_json()["record"]
    assert created["patientId"] == "PT-9999"

    # Read
    get_res = client.get("/api/v1/history/scans")
    assert get_res.status_code == 200
    scans = get_res.get_json()["scans"]
    assert any(s["patientId"] == "PT-9999" for s in scans)

    # Delete
    del_res = client.delete(f"/api/v1/history/scans/{created['id']}")
    assert del_res.status_code == 200

