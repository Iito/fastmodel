"""Shared fixtures for fastmodel tests."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import ClassVar, Optional

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from fastmodel.fastapi.io import create_request_model, create_response_model
from fastmodel.protocol import ServingModule, is_serving_module
from fastmodel.status_code import StatusCode
from fastmodel.utils.async_model import create_async_boot


# ── Pydantic IO models ──


class SimpleInput(BaseModel):
    text: str
    count: int = 1


class SimpleOutput(BaseModel):
    result: str
    length: int


class ImageInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: SkipValidation[bytes]
    width: int = Field(default=224, ge=1)
    height: int = Field(default=224, ge=1)
    grayscale: bool = False


class DetectionBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str


class DetectionInput(BaseModel):
    image_url: str
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    max_detections: int = Field(default=100, ge=1, le=1000)
    augment: bool = False
    device: str = "cpu"


class DetectionOutput(BaseModel):
    boxes: list[DetectionBox]
    num_detections: int
    inference_time_ms: float


# ── Protocol model ──


class ProtocolModel:
    MODULE_NAME: ClassVar[str] = "test.model"
    MODULE_VERSION: ClassVar[str] = "1.0.0"
    INPUT_TYPE: ClassVar[type] = SimpleInput
    OUTPUT_TYPE: ClassVar[type] = SimpleOutput

    def __init__(self):
        pass

    def __call__(self, input: SimpleInput) -> SimpleOutput:
        return SimpleOutput(
            result=input.text * input.count,
            length=len(input.text) * input.count,
        )


# ── Legacy model (no protocol) ──


class LegacyModel:
    def __init__(self):
        pass

    def __call__(self, input: SimpleInput) -> SimpleOutput:
        return SimpleOutput(
            result=input.text * input.count,
            length=len(input.text) * input.count,
        )


# ── Model with alternate arg names ──


class AltArgModel:
    MODULE_NAME: ClassVar[str] = "test.alt"
    MODULE_VERSION: ClassVar[str] = "0.1.0"
    INPUT_TYPE: ClassVar[type] = SimpleInput
    OUTPUT_TYPE: ClassVar[type] = SimpleOutput

    def __init__(self):
        pass

    def __call__(self, input_data: SimpleInput) -> SimpleOutput:
        return SimpleOutput(
            result=input_data.text,
            length=len(input_data.text),
        )


# ── Complex model for detection ──


class DetectorProtocol:
    MODULE_NAME: ClassVar[str] = "yolo.detector"
    MODULE_VERSION: ClassVar[str] = "8.0.0"
    INPUT_TYPE: ClassVar[type] = DetectionInput
    OUTPUT_TYPE: ClassVar[type] = DetectionOutput

    def __init__(self):
        pass

    def __call__(self, input: DetectionInput) -> DetectionOutput:
        box = DetectionBox(
            x1=10, y1=20, x2=100, y2=200,
            confidence=0.92, class_id=0, class_name="person",
        )
        return DetectionOutput(
            boxes=[box], num_detections=1, inference_time_ms=12.3,
        )


class DetectorLegacy:
    def __init__(self):
        pass

    def __call__(self, input: DetectionInput) -> DetectionOutput:
        box = DetectionBox(
            x1=10, y1=20, x2=100, y2=200,
            confidence=0.92, class_id=0, class_name="person",
        )
        return DetectionOutput(
            boxes=[box], num_detections=1, inference_time_ms=12.3,
        )


# ── Fixtures ──


def _build_test_app(model_cls, input_arg, input_type, output_type, version="1.0.0"):
    """Build a FastAPI test app mirroring serve.py logic."""
    logger = logging.getLogger(f"test.{model_cls.__name__}")
    logger.setLevel(logging.WARNING)

    instance = create_async_boot(model_cls, logger, None)

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        asyncio.create_task(instance.async_boot())
        yield

    app = FastAPI(lifespan=_lifespan, title=model_cls.__name__, version=version)

    resp_model = create_response_model(output_type)
    resp_model.force_json = False
    req_model = create_request_model(input_type)
    req_model.returns = input_type
    resp_model._version = version

    def get_logger():
        return logger

    @app.post("/")
    def predict(
        model_input: req_model = Depends(req_model._validate),
        logger=Depends(get_logger),
        model=Depends(instance.get),
    ) -> resp_model:
        response = resp_model(status=0)
        if not model:
            response.status = StatusCode.ModelNotReady
            response.message = StatusCode.ModelNotReady.msg
            return response.generate_streaming_response()

        _response = resp_model(status=0)
        model_args = {input_arg: model_input}
        _response = model(**model_args)

        response.status = StatusCode.Success
        response.message = StatusCode.Success.msg
        for k, v in _response.model_dump().items():
            setattr(response, k, v)
        return response.generate_streaming_response()

    return app


def _wait_for_ready(client, form_data, retries=50):
    """Post requests until the model is booted and returns success."""
    for _ in range(retries):
        resp = client.post("/", data=form_data)
        if resp.status_code == 200:
            body = resp.json()
            if body.get("status") == StatusCode.Success.value:
                return resp
        time.sleep(0.02)
    raise TimeoutError("Model did not become ready")
