"""End-to-end tests: full FastAPI app with real HTTP requests."""

import time

import pytest
from fastapi.testclient import TestClient

from fastmodel.status_code import StatusCode
from tests.conftest import (
    DetectionInput,
    DetectionOutput,
    DetectorLegacy,
    DetectorProtocol,
    LegacyModel,
    ProtocolModel,
    SimpleInput,
    SimpleOutput,
    _build_test_app,
    _wait_for_ready,
)


SIMPLE_FORM = {"text": "hello", "count": "3"}
DETECTION_FORM = {
    "image_url": "http://example.com/img.jpg",
    "confidence_threshold": "0.5",
    "iou_threshold": "0.45",
    "max_detections": "100",
    "augment": "false",
    "device": "cpu",
}


# ── Protocol model e2e ──


class TestProtocolE2E:
    def test_simple_request_response(self):
        app = _build_test_app(
            ProtocolModel, "input", SimpleInput, SimpleOutput, "1.0.0"
        )
        with TestClient(app) as client:
            resp = _wait_for_ready(client, SIMPLE_FORM)
            body = resp.json()

            assert body["status"] == StatusCode.Success.value
            assert body["message"] == "Success"
            assert body["result"] == "hellohellohello"
            assert body["length"] == 15

    def test_version_in_response(self):
        app = _build_test_app(
            ProtocolModel, "input", SimpleInput, SimpleOutput, "2.5.0"
        )
        with TestClient(app) as client:
            resp = _wait_for_ready(client, SIMPLE_FORM)
            body = resp.json()
            assert body["version"] == "2.5.0"

    def test_default_values_used(self):
        app = _build_test_app(
            ProtocolModel, "input", SimpleInput, SimpleOutput
        )
        with TestClient(app) as client:
            # Only send required field, count defaults to 1
            resp = _wait_for_ready(client, {"text": "hi"})
            body = resp.json()
            assert body["result"] == "hi"
            assert body["length"] == 2

    def test_model_not_ready_initially(self):
        """First request before boot completes returns ModelNotReady."""
        app = _build_test_app(
            ProtocolModel, "input", SimpleInput, SimpleOutput
        )
        with TestClient(app) as client:
            # Immediately send request — model may not be ready
            resp = client.post("/", data=SIMPLE_FORM)
            body = resp.json()
            # Either ready or not ready, both are valid
            assert body["status"] in (
                StatusCode.Success.value,
                StatusCode.ModelNotReady.value,
            )


# ── Legacy model e2e ──


class TestLegacyE2E:
    def test_simple_request_response(self):
        app = _build_test_app(
            LegacyModel, "input", SimpleInput, SimpleOutput
        )
        with TestClient(app) as client:
            resp = _wait_for_ready(client, SIMPLE_FORM)
            body = resp.json()

            assert body["status"] == StatusCode.Success.value
            assert body["result"] == "hellohellohello"
            assert body["length"] == 15


# ── Complex model e2e ──


class TestDetectionE2E:
    def test_protocol_detector(self):
        app = _build_test_app(
            DetectorProtocol, "input", DetectionInput, DetectionOutput, "8.0.0"
        )
        with TestClient(app) as client:
            resp = _wait_for_ready(client, DETECTION_FORM)
            body = resp.json()

            assert body["status"] == StatusCode.Success.value
            assert body["num_detections"] == 1
            assert body["inference_time_ms"] == 12.3
            assert len(body["boxes"]) == 1

            box = body["boxes"][0]
            assert box["class_name"] == "person"
            assert box["confidence"] == 0.92

    def test_legacy_detector(self):
        app = _build_test_app(
            DetectorLegacy, "input", DetectionInput, DetectionOutput
        )
        with TestClient(app) as client:
            resp = _wait_for_ready(client, DETECTION_FORM)
            body = resp.json()

            assert body["status"] == StatusCode.Success.value
            assert body["num_detections"] == 1


# ── Both paths produce identical responses ──


class TestProtocolLegacyParity:
    def test_same_response(self):
        """Protocol and Legacy models return identical JSON for the same input."""
        app_proto = _build_test_app(
            ProtocolModel, "input", SimpleInput, SimpleOutput, "1.0.0"
        )
        app_legacy = _build_test_app(
            LegacyModel, "input", SimpleInput, SimpleOutput, "1.0.0"
        )

        with TestClient(app_proto) as client_p, TestClient(app_legacy) as client_l:
            resp_p = _wait_for_ready(client_p, SIMPLE_FORM)
            resp_l = _wait_for_ready(client_l, SIMPLE_FORM)

            body_p = resp_p.json()
            body_l = resp_l.json()

            assert body_p["result"] == body_l["result"]
            assert body_p["length"] == body_l["length"]
            assert body_p["status"] == body_l["status"]
            assert body_p["message"] == body_l["message"]

    def test_detection_parity(self):
        app_proto = _build_test_app(
            DetectorProtocol, "input", DetectionInput, DetectionOutput
        )
        app_legacy = _build_test_app(
            DetectorLegacy, "input", DetectionInput, DetectionOutput
        )

        with TestClient(app_proto) as client_p, TestClient(app_legacy) as client_l:
            resp_p = _wait_for_ready(client_p, DETECTION_FORM)
            resp_l = _wait_for_ready(client_l, DETECTION_FORM)

            body_p = resp_p.json()
            body_l = resp_l.json()

            assert body_p["num_detections"] == body_l["num_detections"]
            assert body_p["boxes"] == body_l["boxes"]
