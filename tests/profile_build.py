"""Profile the app build phase after optimization."""

import time
import warnings

from pydantic import BaseModel, Field
from typing import Optional

warnings.filterwarnings("ignore")


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


def fmt(s):
    if s < 0.001:
        return f"{s*1e6:>8.1f} µs"
    return f"{s*1e3:>8.2f} ms"


N = 200

from fastmodel.fastapi.io import create_request_model, create_response_model
from fastmodel.fastapi.utils import _merge_models

print("=" * 55)
print("  Build Phase Profile (optimized)")
print("=" * 55)

t0 = time.perf_counter()
for _ in range(N):
    resp = create_response_model(DetectionOutput)
t_resp = (time.perf_counter() - t0) / N
print(f"  create_response_model:  {fmt(t_resp)}")

t0 = time.perf_counter()
for _ in range(N):
    req = create_request_model(DetectionInput)
t_req = (time.perf_counter() - t0) / N
print(f"  create_request_model:   {fmt(t_req)}")

t_total = t_resp + t_req
print(f"  Combined:               {fmt(t_total)}")
