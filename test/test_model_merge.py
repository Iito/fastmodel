"""Tests for create_request_model and create_response_model."""

import json
from typing import Optional, get_type_hints

import pytest
from pydantic import BaseModel, Field

from fastmodel.fastapi.io import (
    ApiBaseRequest,
    ApiBaseResponse,
    create_request_model,
    create_response_model,
)
from test.conftest import (
    DetectionBox,
    DetectionInput,
    DetectionOutput,
    SimpleInput,
    SimpleOutput,
)


# ── Response model creation ──


class TestResponseModel:
    def test_creates_with_correct_fields(self):
        resp = create_response_model(SimpleOutput)
        field_names = set(resp.model_fields.keys())
        assert "result" in field_names
        assert "length" in field_names
        # Inherits base response fields
        assert "status" in field_names
        assert "message" in field_names
        assert "version" in field_names

    def test_response_fields_are_optional_with_none_default(self):
        resp = create_response_model(SimpleOutput)
        for name in ("result", "length"):
            field = resp.model_fields[name]
            assert field.default is None, f"{name} should default to None"

    def test_response_instantiation(self):
        resp = create_response_model(SimpleOutput)
        instance = resp(status=100, message="ok")
        assert instance.status == 100
        assert instance.message == "ok"
        assert instance.result is None
        assert instance.length is None

    def test_response_with_values(self):
        resp = create_response_model(SimpleOutput)
        instance = resp(status=100, result="hello", length=5)
        assert instance.result == "hello"
        assert instance.length == 5

    def test_response_json_serializable(self):
        resp = create_response_model(SimpleOutput)
        instance = resp(status=100, message="ok", result="test", length=4)
        data = instance.model_dump()
        # Should be JSON serializable
        json.dumps(data)

    def test_complex_nested_response(self):
        resp = create_response_model(DetectionOutput)
        field_names = set(resp.model_fields.keys())
        assert "boxes" in field_names
        assert "num_detections" in field_names
        assert "inference_time_ms" in field_names

    def test_response_inherits_from_base(self):
        resp = create_response_model(SimpleOutput)
        assert issubclass(resp, ApiBaseResponse)
        assert issubclass(resp, SimpleOutput)

    def test_response_name_strips_output(self):
        resp = create_response_model(SimpleOutput)
        assert "Response" in resp.__name__
        assert "Output" not in resp.__name__


# ── Request model creation ──


class TestRequestModel:
    def test_creates_with_correct_fields(self):
        req = create_request_model(SimpleInput)
        field_names = set(req.model_fields.keys())
        assert "text" in field_names
        assert "count" in field_names

    def test_request_preserves_defaults(self):
        req = create_request_model(SimpleInput)
        assert req.model_fields["count"].default == 1

    def test_request_has_validate_method(self):
        req = create_request_model(SimpleInput)
        assert hasattr(req, "_validate")
        assert callable(req._validate)

    def test_request_inherits_from_base(self):
        req = create_request_model(SimpleInput)
        assert issubclass(req, ApiBaseRequest)
        assert issubclass(req, SimpleInput)

    def test_request_name_strips_input(self):
        req = create_request_model(SimpleInput)
        assert "Request" in req.__name__
        assert "Input" not in req.__name__

    def test_complex_request_fields(self):
        req = create_request_model(DetectionInput)
        field_names = set(req.model_fields.keys())
        assert "image_url" in field_names
        assert "confidence_threshold" in field_names
        assert "iou_threshold" in field_names
        assert "max_detections" in field_names
        assert "augment" in field_names
        assert "device" in field_names

    def test_complex_request_defaults(self):
        req = create_request_model(DetectionInput)
        assert req.model_fields["confidence_threshold"].default == 0.5
        assert req.model_fields["iou_threshold"].default == 0.45
        assert req.model_fields["max_detections"].default == 100
        assert req.model_fields["augment"].default is False
        assert req.model_fields["device"].default == "cpu"


# ── Edge cases ──


class TestEdgeCases:
    def test_model_with_no_fields(self):
        class EmptyOutput(BaseModel):
            pass

        resp = create_response_model(EmptyOutput)
        # Should still have base fields
        assert "status" in resp.model_fields
        assert "message" in resp.model_fields

    def test_model_with_field_metadata(self):
        class Annotated(BaseModel):
            score: float = Field(default=0.0, description="A score", ge=0, le=1)

        req = create_request_model(Annotated)
        assert "score" in req.model_fields
        assert req.model_fields["score"].default == 0.0

    def test_model_with_optional_field(self):
        class WithOptional(BaseModel):
            name: str
            tag: Optional[str] = None

        req = create_request_model(WithOptional)
        assert req.model_fields["tag"].default is None
