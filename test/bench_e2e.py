"""End-to-end benchmark: full server build + first request + response.

Measures exactly what happens in production:
  1. import_from_string (resolve IO types)
  2. create_async_boot (wrap model)
  3. FastAPI app construction (routes, request/response model generation)
  4. Model boot (async_boot)
  5. POST / with real form data → get JSON response back

Runs each phase for both Protocol and Legacy models, then compares.
"""

import asyncio
import importlib.metadata
import logging
import time
from contextlib import asynccontextmanager
from typing import ClassVar, Optional

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from fastmodel.fastapi.io import create_request_model, create_response_model
from fastmodel.status_code import StatusCode
from fastmodel.utils.async_model import create_async_boot
from fastmodel.utils.importer import (
    _resolve_from_annotations,
    _resolve_from_protocol,
    _cached_signature,
)
from fastmodel.protocol import is_serving_module


# ══════════════════════════════════════════════════════════════
# Model definitions
# ══════════════════════════════════════════════════════════════


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
    classes: Optional[list[int]] = None
    augment: bool = False
    device: str = "cpu"


class DetectionOutput(BaseModel):
    boxes: list[DetectionBox]
    num_detections: int
    inference_time_ms: float


class DetectorProtocol:
    MODULE_NAME: ClassVar[str] = "yolo.detector"
    MODULE_VERSION: ClassVar[str] = "8.1.0"
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


# ══════════════════════════════════════════════════════════════
# Build the full FastAPI app exactly as serve.py does
# ══════════════════════════════════════════════════════════════


def build_app(model_class, input_arg, input_type, output_type, model_version):
    """Replicates the app-building logic from serve.py."""
    logger = logging.getLogger(f"bench.{model_class.__name__}")
    logger.setLevel(logging.WARNING)  # quiet during benchmarks

    model_instance = create_async_boot(model_class, logger, None)

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        asyncio.create_task(model_instance.async_boot())
        yield

    app = FastAPI(
        lifespan=_lifespan,
        title=model_class.__name__,
        version=model_version or "0.0.0",
    )

    model_response = create_response_model(output_type)
    model_response.force_json = False
    model_request = create_request_model(input_type)
    model_request.returns = input_type
    model_response._version = model_version

    def get_logger():
        return logger

    @app.post("/")
    def predict(
        model_input: model_request = Depends(model_request._validate),
        logger=Depends(get_logger),
        model=Depends(model_instance.get),
    ) -> model_response:
        response = model_response(status=0)
        if not model:
            response.status = StatusCode.ModelNotReady
            response.message = StatusCode.ModelNotReady.msg
            return response.generate_streaming_response()

        _response = model_response(status=0)
        model_args = {input_arg: model_input}
        _response: output_type = model(**model_args)

        response.status = StatusCode.Success
        response.message = StatusCode.Success.msg
        for key, value in _response.model_dump().items():
            setattr(response, key, value)
        return response.generate_streaming_response()

    return app


# ══════════════════════════════════════════════════════════════
# Timed phases
# ══════════════════════════════════════════════════════════════


def run_full_cycle(label, model_class, use_protocol):
    """Run the full lifecycle and return phase timings."""
    timings = {}

    # Phase 1: IO resolution
    _cached_signature.cache_clear()
    t0 = time.perf_counter()
    if use_protocol:
        _, (input_arg, input_type), (_, output_type) = _resolve_from_protocol(
            model_class, None, None
        )
    else:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            input_arg, input_type = _resolve_from_annotations(None, model_class, "input")
            _, output_type = _resolve_from_annotations(None, model_class, "return")
    timings["1_resolve"] = time.perf_counter() - t0

    # Phase 2: Version resolution
    t0 = time.perf_counter()
    if hasattr(model_class, "MODULE_VERSION"):
        version = model_class.MODULE_VERSION
    elif hasattr(model_class, "version") and callable(model_class.version):
        version = model_class.version()
    else:
        version = "0.0.0"
    timings["2_version"] = time.perf_counter() - t0

    # Phase 3: App construction (FastAPI + request/response model generation)
    t0 = time.perf_counter()
    app = build_app(model_class, input_arg, input_type, output_type, version)
    timings["3_app_build"] = time.perf_counter() - t0

    # Phase 4: First request (includes model boot via lifespan)
    t0 = time.perf_counter()
    with TestClient(app) as client:
        timings["4_boot"] = time.perf_counter() - t0

        # Wait for async boot to complete
        for _ in range(100):
            resp = client.post("/", data={
                "image_url": "http://example.com/img.jpg",
                "confidence_threshold": "0.5",
                "iou_threshold": "0.45",
                "max_detections": "100",
                "augment": "false",
                "device": "cpu",
            })
            if resp.status_code == 200:
                body = resp.json()
                if body.get("status") == StatusCode.Success.value:
                    break
            time.sleep(0.01)

        timings["5_first_req"] = time.perf_counter() - t0 - timings["4_boot"]

    timings["total"] = sum(timings.values())
    return timings, body


# ══════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════

RUNS = 5


def format_us(seconds):
    if seconds < 0.001:
        return f"{seconds*1e6:>9.1f} µs"
    else:
        return f"{seconds*1e3:>9.2f} ms"


if __name__ == "__main__":
    print("=" * 80)
    print("  End-to-End Benchmark: Server Build + First Request")
    print("=" * 80)

    for run in range(RUNS):
        print(f"\n── Run {run + 1}/{RUNS} ──\n")

        proto_timings, proto_body = run_full_cycle(
            "Protocol", DetectorProtocol, use_protocol=True,
        )
        legacy_timings, legacy_body = run_full_cycle(
            "Legacy", DetectorLegacy, use_protocol=False,
        )

        # Verify both produce the same response
        for key in ("num_detections", "inference_time_ms", "status"):
            assert proto_body[key] == legacy_body[key], (
                f"Mismatch on {key}: {proto_body[key]} vs {legacy_body[key]}"
            )

        header = f"  {'Phase':<20s} {'Protocol':>12s} {'Legacy':>12s} {'Speedup':>10s}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for phase in ("1_resolve", "2_version", "3_app_build", "4_boot", "5_first_req", "total"):
            p = proto_timings[phase]
            l = legacy_timings[phase]
            speedup = l / p if p > 0 else float("inf")
            label = phase.replace("_", " ").title()
            print(f"  {label:<20s} {format_us(p)} {format_us(l)} {speedup:>9.1f}x")

    print(f"\nResponse correctness verified across all runs ✓")
    print()
