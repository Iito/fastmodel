"""Benchmark: Protocol vs Legacy with realistic CV/NLP model complexity."""

import timeit
import warnings
from typing import ClassVar, Optional

from pydantic import BaseModel, Field, ConfigDict, SkipValidation

from fastmodel.protocol import is_serving_module
from fastmodel.utils.importer import (
    _resolve_from_annotations,
    _resolve_from_protocol,
    _cached_signature,
)


# ══════════════════════════════════════════════════════════════
# 1. Object Detection (YOLO-style) — deeply nested output
# ══════════════════════════════════════════════════════════════


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str


class DetectionResult(BaseModel):
    boxes: list[BoundingBox]
    num_detections: int
    inference_time_ms: float
    image_width: int
    image_height: int
    model_name: str = "yolov8"


class DetectionInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: SkipValidation[bytes]
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    max_detections: int = Field(default=100, ge=1, le=1000)
    classes: Optional[list[int]] = None
    augment: bool = False
    half_precision: bool = False
    device: str = "cuda:0"


class YOLOv8Protocol:
    MODULE_NAME: ClassVar[str] = "yolov8.detector"
    MODULE_VERSION: ClassVar[str] = "8.1.0"
    INPUT_TYPE: ClassVar[type] = DetectionInput
    OUTPUT_TYPE: ClassVar[type] = DetectionResult

    def __call__(self, input: DetectionInput) -> DetectionResult:
        return DetectionResult(
            boxes=[], num_detections=0, inference_time_ms=0.0,
            image_width=640, image_height=640,
        )


class YOLOv8Legacy:
    def __call__(self, input: DetectionInput) -> DetectionResult:
        return DetectionResult(
            boxes=[], num_detections=0, inference_time_ms=0.0,
            image_width=640, image_height=640,
        )


# ══════════════════════════════════════════════════════════════
# 2. NLP Text Generation (LLM-style) — many parameters
# ══════════════════════════════════════════════════════════════


class Message(BaseModel):
    role: str
    content: str


class GenerationConfig(BaseModel):
    max_new_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    repetition_penalty: float = Field(default=1.1, ge=1.0)
    do_sample: bool = True
    num_beams: int = Field(default=1, ge=1)
    length_penalty: float = 1.0
    early_stopping: bool = False
    no_repeat_ngram_size: int = Field(default=0, ge=0)


class LLMInput(BaseModel):
    messages: list[Message]
    generation_config: GenerationConfig = GenerationConfig()
    system_prompt: Optional[str] = None
    stop_sequences: list[str] = Field(default_factory=list)
    stream: bool = False
    tools: Optional[list[dict]] = None
    response_format: Optional[str] = None
    seed: Optional[int] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMOutput(BaseModel):
    text: str
    finish_reason: str
    usage: TokenUsage
    model_name: str
    logprobs: Optional[list[float]] = None


class LLMProtocol:
    MODULE_NAME: ClassVar[str] = "llm.generator"
    MODULE_VERSION: ClassVar[str] = "2.0.0"
    INPUT_TYPE: ClassVar[type] = LLMInput
    OUTPUT_TYPE: ClassVar[type] = LLMOutput

    def __call__(self, input_data: LLMInput) -> LLMOutput:
        return LLMOutput(
            text="", finish_reason="stop", model_name="llama",
            usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )


class LLMLegacy:
    def __call__(self, input_data: LLMInput) -> LLMOutput:
        return LLMOutput(
            text="", finish_reason="stop", model_name="llama",
            usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )


# ══════════════════════════════════════════════════════════════
# 3. Image Segmentation (SAM-style) — mixed binary + JSON IO
# ══════════════════════════════════════════════════════════════


class PointPrompt(BaseModel):
    x: float
    y: float
    label: int  # 1=foreground, 0=background


class BoxPrompt(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class SegmentationInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: SkipValidation[bytes]
    point_prompts: Optional[list[PointPrompt]] = None
    box_prompts: Optional[list[BoxPrompt]] = None
    mask_input: Optional[SkipValidation[bytes]] = None
    multimask_output: bool = True
    return_logits: bool = False
    normalize_coords: bool = True
    image_format: str = "RGB"
    original_size: Optional[tuple[int, int]] = None


class SegmentationMask(BaseModel):
    mask_id: int
    iou_prediction: float
    area: int
    bbox: list[float]
    stability_score: float


class SegmentationOutput(BaseModel):
    masks: list[SegmentationMask]
    num_masks: int
    inference_time_ms: float
    image_embedding_time_ms: float
    mask_data: Optional[SkipValidation[bytes]] = None


class SAMProtocol:
    MODULE_NAME: ClassVar[str] = "sam.segmentation"
    MODULE_VERSION: ClassVar[str] = "2.1.0"
    INPUT_TYPE: ClassVar[type] = SegmentationInput
    OUTPUT_TYPE: ClassVar[type] = SegmentationOutput

    def __call__(self, input: SegmentationInput) -> SegmentationOutput:
        return SegmentationOutput(
            masks=[], num_masks=0, inference_time_ms=0.0,
            image_embedding_time_ms=0.0,
        )


class SAMLegacy:
    def __call__(self, input: SegmentationInput) -> SegmentationOutput:
        return SegmentationOutput(
            masks=[], num_masks=0, inference_time_ms=0.0,
            image_embedding_time_ms=0.0,
        )


# ══════════════════════════════════════════════════════════════
# 4. OCR Pipeline (multi-stage) — inheritance chain
# ══════════════════════════════════════════════════════════════


class Region(BaseModel):
    text: str
    confidence: float
    bbox: list[float]
    language: Optional[str] = None


class Line(BaseModel):
    text: str
    regions: list[Region]
    bbox: list[float]
    direction: str = "ltr"


class Page(BaseModel):
    lines: list[Line]
    width: int
    height: int
    rotation: float = 0.0
    language: str = "en"


class OCRInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: SkipValidation[bytes]
    languages: list[str] = Field(default_factory=lambda: ["en"])
    detect_orientation: bool = True
    correct_skew: bool = True
    psm: int = Field(default=3, ge=0, le=13, description="Page segmentation mode")
    oem: int = Field(default=3, ge=0, le=3, description="OCR engine mode")
    dpi: int = Field(default=300, ge=72, le=1200)
    whitelist: Optional[str] = None
    blacklist: Optional[str] = None
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class OCROutput(BaseModel):
    pages: list[Page]
    full_text: str
    num_pages: int
    num_lines: int
    num_regions: int
    avg_confidence: float
    detected_languages: list[str]
    processing_time_ms: float


class OCRProtocol:
    MODULE_NAME: ClassVar[str] = "ocr.pipeline"
    MODULE_VERSION: ClassVar[str] = "3.0.0"
    INPUT_TYPE: ClassVar[type] = OCRInput
    OUTPUT_TYPE: ClassVar[type] = OCROutput

    def __call__(self, input: OCRInput) -> OCROutput:
        return OCROutput(
            pages=[], full_text="", num_pages=0, num_lines=0,
            num_regions=0, avg_confidence=0.0, detected_languages=[],
            processing_time_ms=0.0,
        )


class OCRLegacy:
    def __call__(self, input: OCRInput) -> OCROutput:
        return OCROutput(
            pages=[], full_text="", num_pages=0, num_lines=0,
            num_regions=0, avg_confidence=0.0, detected_languages=[],
            processing_time_ms=0.0,
        )


# ══════════════════════════════════════════════════════════════
# 5. Multi-modal (CLIP-style) — dual input
# ══════════════════════════════════════════════════════════════


class CLIPInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: Optional[SkipValidation[bytes]] = None
    text: Optional[list[str]] = None
    normalize: bool = True
    truncate_text: bool = True
    max_text_length: int = Field(default=77, ge=1, le=512)
    image_size: int = Field(default=224, ge=32, le=1024)
    interpolation: str = "bicubic"
    return_tensors: str = "np"
    batch_size: int = Field(default=32, ge=1, le=512)
    device: str = "cuda:0"
    precision: str = "fp16"


class EmbeddingResult(BaseModel):
    image_embedding: Optional[list[float]] = None
    text_embeddings: Optional[list[list[float]]] = None
    similarity_scores: Optional[list[float]] = None
    embedding_dim: int
    model_name: str
    inference_time_ms: float


class CLIPProtocol:
    MODULE_NAME: ClassVar[str] = "clip.embedder"
    MODULE_VERSION: ClassVar[str] = "1.5.0"
    INPUT_TYPE: ClassVar[type] = CLIPInput
    OUTPUT_TYPE: ClassVar[type] = EmbeddingResult

    def __call__(self, input: CLIPInput) -> EmbeddingResult:
        return EmbeddingResult(
            embedding_dim=512, model_name="clip", inference_time_ms=0.0,
        )


class CLIPLegacy:
    def __call__(self, input: CLIPInput) -> EmbeddingResult:
        return EmbeddingResult(
            embedding_dim=512, model_name="clip", inference_time_ms=0.0,
        )


# ══════════════════════════════════════════════════════════════
# Benchmark runner
# ══════════════════════════════════════════════════════════════

MODELS = [
    ("YOLOv8  (Object Detection)", YOLOv8Protocol, YOLOv8Legacy),
    ("LLM     (Text Generation) ", LLMProtocol, LLMLegacy),
    ("SAM     (Segmentation)    ", SAMProtocol, SAMLegacy),
    ("OCR     (Pipeline)        ", OCRProtocol, OCRLegacy),
    ("CLIP    (Multi-modal)     ", CLIPProtocol, CLIPLegacy),
]

N = 50_000


def count_fields(model_cls):
    """Recursively count total fields across input + output Pydantic models."""
    counted = set()

    def _count(cls):
        if id(cls) in counted or not hasattr(cls, "model_fields"):
            return 0
        counted.add(id(cls))
        total = len(cls.model_fields)
        for field in cls.model_fields.values():
            ann = field.annotation
            if hasattr(ann, "model_fields"):
                total += _count(ann)
            elif hasattr(ann, "__args__"):
                for arg in ann.__args__:
                    if hasattr(arg, "model_fields"):
                        total += _count(arg)
        return total

    if hasattr(model_cls, "INPUT_TYPE"):
        return _count(model_cls.INPUT_TYPE) + _count(model_cls.OUTPUT_TYPE)
    # legacy: get from __call__ annotations
    anns = model_cls.__call__.__annotations__
    total = 0
    for v in anns.values():
        if hasattr(v, "model_fields"):
            total += _count(v)
    return total


def bench_model(name, proto_cls, legacy_cls):
    fields = count_fields(proto_cls)

    # Clear signature cache between models
    _cached_signature.cache_clear()

    t_proto = timeit.timeit(
        lambda: _resolve_from_protocol(proto_cls, None, None), number=N
    )

    _cached_signature.cache_clear()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_legacy = timeit.timeit(
            lambda: (
                _resolve_from_annotations(None, legacy_cls, "input"),
                _resolve_from_annotations(None, legacy_cls, "return"),
            ),
            number=N,
        )

    speedup = t_legacy / t_proto
    print(
        f"  {name}  "
        f"fields={fields:>3d}  "
        f"proto={t_proto/N*1e6:>6.2f} µs  "
        f"legacy={t_legacy/N*1e6:>6.2f} µs  "
        f"speedup={speedup:>5.1f}x"
    )
    return t_proto, t_legacy, fields


def bench_cold_start():
    """Simulate cold starts: clear cache before each call."""
    print("\n── Cold start (no cache, single call) ──\n")
    for name, proto_cls, legacy_cls in MODELS:
        _cached_signature.cache_clear()
        t_proto = timeit.timeit(
            lambda: _resolve_from_protocol(proto_cls, None, None), number=1
        )
        _cached_signature.cache_clear()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t_legacy = timeit.timeit(
                lambda: (
                    _resolve_from_annotations(None, legacy_cls, "input"),
                    _resolve_from_annotations(None, legacy_cls, "return"),
                ),
                number=1,
            )
        speedup = t_legacy / t_proto if t_proto > 0 else float("inf")
        print(
            f"  {name}  "
            f"proto={t_proto*1e6:>8.2f} µs  "
            f"legacy={t_legacy*1e6:>8.2f} µs  "
            f"speedup={speedup:>5.1f}x"
        )


if __name__ == "__main__":
    print("=" * 80)
    print("  Complex Model Benchmark: Protocol vs Legacy Resolution")
    print("=" * 80)

    print(f"\n── Warmed (x{N:,} calls, cache active after first) ──\n")
    for name, proto_cls, legacy_cls in MODELS:
        bench_model(name, proto_cls, legacy_cls)

    bench_cold_start()
    print()
