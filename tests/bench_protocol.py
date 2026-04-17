"""Benchmark: Protocol-based vs legacy annotation-based model resolution."""

import timeit
from typing import ClassVar

from pydantic import BaseModel

from fastmodel.protocol import is_serving_module
from fastmodel.utils.importer import (
    _get_call_input_arg,
    _resolve_from_annotations,
    _resolve_from_protocol,
)


# ── Pydantic IO models ──


class ImageInput(BaseModel):
    image_url: str
    width: int = 224
    height: int = 224


class ImageOutput(BaseModel):
    label: str
    confidence: float


# ── Protocol model ──


class ProtocolClassifier:
    MODULE_NAME: ClassVar[str] = "image.classifier"
    MODULE_VERSION: ClassVar[str] = "1.0.0"
    INPUT_TYPE: ClassVar[type] = ImageInput
    OUTPUT_TYPE: ClassVar[type] = ImageOutput

    def __call__(self, input: ImageInput) -> ImageOutput:
        return ImageOutput(label="cat", confidence=0.95)


# ── Legacy model (no ClassVars, relies on annotation introspection) ──


class LegacyClassifier:
    def __call__(self, input: ImageInput) -> ImageOutput:
        return ImageOutput(label="cat", confidence=0.95)


# ── Benchmarks ──

N = 50_000


def bench_is_serving_module():
    t_proto = timeit.timeit(lambda: is_serving_module(ProtocolClassifier), number=N)
    t_legacy = timeit.timeit(lambda: is_serving_module(LegacyClassifier), number=N)
    print(f"is_serving_module() x{N}")
    print(f"  Protocol model:  {t_proto:.4f}s  ({t_proto/N*1e6:.2f} µs/call)")
    print(f"  Legacy model:    {t_legacy:.4f}s  ({t_legacy/N*1e6:.2f} µs/call)")
    print()


def bench_resolution():
    t_proto = timeit.timeit(
        lambda: _resolve_from_protocol(ProtocolClassifier, None, None), number=N
    )

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_legacy_in = timeit.timeit(
            lambda: _resolve_from_annotations(None, LegacyClassifier, "input"), number=N
        )
        t_legacy_out = timeit.timeit(
            lambda: _resolve_from_annotations(None, LegacyClassifier, "return"), number=N
        )
    t_legacy = t_legacy_in + t_legacy_out

    print(f"Full IO resolution x{N}")
    print(f"  Protocol path:   {t_proto:.4f}s  ({t_proto/N*1e6:.2f} µs/call)")
    print(f"  Legacy path:     {t_legacy:.4f}s  ({t_legacy/N*1e6:.2f} µs/call)")
    print(f"  Speedup:         {t_legacy/t_proto:.1f}x")
    print()


def bench_full_pipeline():
    """Simulate the full import_from_string decision path."""

    def protocol_path():
        if is_serving_module(ProtocolClassifier):
            return _resolve_from_protocol(ProtocolClassifier, None, None)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        def legacy_path():
            if not is_serving_module(LegacyClassifier):
                inp = _resolve_from_annotations(None, LegacyClassifier, "input")
                out = _resolve_from_annotations(None, LegacyClassifier, "return")
                return LegacyClassifier, inp, out

    t_proto = timeit.timeit(protocol_path, number=N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_legacy = timeit.timeit(legacy_path, number=N)

    print(f"Full pipeline (detect + resolve) x{N}")
    print(f"  Protocol model:  {t_proto:.4f}s  ({t_proto/N*1e6:.2f} µs/call)")
    print(f"  Legacy model:    {t_legacy:.4f}s  ({t_legacy/N*1e6:.2f} µs/call)")
    print(f"  Speedup:         {t_legacy/t_proto:.1f}x")
    print()


def bench_correctness():
    """Verify both paths produce equivalent results."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        cls_p, (arg_p, in_p), (ret_p, out_p) = _resolve_from_protocol(
            ProtocolClassifier, None, None
        )
        arg_l, in_l = _resolve_from_annotations(None, LegacyClassifier, "input")
        _, out_l = _resolve_from_annotations(None, LegacyClassifier, "return")

    assert in_p is ImageInput and in_l is ImageInput
    assert out_p is ImageOutput and out_l is ImageOutput
    assert arg_p == arg_l == "input"
    print("Correctness: both paths resolve to the same IO types ✓")
    print()


if __name__ == "__main__":
    print("=" * 55)
    print("  Protocol vs Legacy Resolution Benchmark")
    print("=" * 55)
    print()
    bench_correctness()
    bench_is_serving_module()
    bench_resolution()
    bench_full_pipeline()
