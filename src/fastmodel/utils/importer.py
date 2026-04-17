# This is a modified version of the uvicorn.importer module from the Uvicorn project.

import importlib
import inspect
import warnings
from functools import lru_cache
from typing import Any, Tuple, Type

from ..protocol import is_serving_module


class ImportFromStringError(Exception):
    pass


@lru_cache(maxsize=None)
def _cached_signature(cls: Type) -> inspect.Signature:
    return inspect.signature(cls.__call__)


def import_from_string(import_str: Any) -> Tuple[Type, Tuple[str, Type], Tuple[str, Type]]:
    if not isinstance(import_str, str):
        return import_str

    module_str, input_model_str, output_model_str = parse_import_string(import_str)
    model_class = import_model_class(module_str)

    # Fast path: protocol-compliant class with ClassVars
    if is_serving_module(model_class):
        return _resolve_from_protocol(model_class, input_model_str, output_model_str)

    # Fallback: legacy annotation introspection
    input_model_class = _resolve_from_annotations(input_model_str, model_class, "input")
    output_model_class = _resolve_from_annotations(output_model_str, model_class, "return")
    return model_class, input_model_class, output_model_class


def parse_import_string(import_str: str) -> Tuple[str, str, str]:
    module_str, _, attrs_str = import_str.partition(":")
    if not module_str and not attrs_str:
        raise ImportFromStringError(
            f'Import string "{import_str}" must be in format "<module>:<input>-><output>".'
        )
    input_model_str = output_model_str = None
    if attrs_str:
        input_model_str, _, output_model_str = attrs_str.partition("->")
        if not input_model_str or not output_model_str:
            raise ImportFromStringError(
                f'Import string "{import_str}" must be in format "<module>:<input>-><output>".'
            )

    return module_str, input_model_str, output_model_str


def import_model_class(module_str: str) -> Type:
    try:
        module_class_str = None
        if module_str.split(".")[-1][0].isupper():
            try:
                module_str, module_class_str = module_str.rsplit(".", 1)
            except ValueError:
                raise ImportFromStringError(
                    f'Import string "{module_str}" must be in format "<module>.<class>".'
                )
        model_module = importlib.import_module(module_str)

        if module_class_str:
            return getattr(model_module, module_class_str)
        else:
            raise ImportFromStringError("Model class not found in module.")
    except ModuleNotFoundError as exc:
        if exc.name != module_str:
            raise exc from None
        raise ImportFromStringError(f'Could not import module "{module_str}".')


# ── Protocol-aware resolution ──


def _resolve_from_protocol(
    model_class: Type, input_model_str: str | None, output_model_str: str | None
) -> Tuple[Type, Tuple[str, Type], Tuple[str, Type]]:
    """Resolve IO types directly from ServingModule ClassVars."""
    input_type = model_class.INPUT_TYPE
    output_type = model_class.OUTPUT_TYPE

    if input_model_str:
        override = import_class(input_model_str)
        if override != input_type:
            warnings.warn(
                f"Overriding INPUT_TYPE {input_type.__name__} with {input_model_str}"
            )
        input_type = override

    if output_model_str:
        override = import_class(output_model_str)
        if override != output_type:
            warnings.warn(
                f"Overriding OUTPUT_TYPE {output_type.__name__} with {output_model_str}"
            )
        output_type = override

    input_arg = _get_call_input_arg(model_class)
    return model_class, (input_arg, input_type), ("return", output_type)


def _get_call_input_arg(model_class: Type) -> str:
    """Get the first non-self parameter name from __call__.

    Fast path: read __code__.co_varnames directly (avoids inspect.signature).
    Fallback: use cached inspect.signature for edge cases (e.g. decorators).
    """
    code = getattr(model_class.__call__, "__code__", None)
    if code and code.co_varnames:
        # co_varnames starts with arg names; skip 'self'
        for name in code.co_varnames[: code.co_argcount]:
            if name != "self":
                return name

    # Fallback for wrapped/decorated __call__
    params = _cached_signature(model_class).parameters
    for name in params:
        if name != "self":
            return name

    raise ImportFromStringError(
        f"{model_class.__name__}.__call__ must accept at least one parameter."
    )


# ── Legacy annotation-based resolution ──


def _resolve_from_annotations(
    model_str: str, model_class: Type, annotation_key: str
) -> Tuple[str, Type]:
    """Fall back to __call__ annotation introspection for non-protocol models."""
    if not hasattr(model_class, "__call__"):
        raise ImportFromStringError(f"Model {model_class.__name__} must be callable.")

    expected_model = None
    call_annotations = model_class.__call__.__annotations__

    # Try direct annotation match: e.g. "input", "input_data", "return"
    for annotation in (annotation_key, annotation_key + "_data"):
        expected_model = call_annotations.get(annotation)
        if expected_model:
            annotation_key = annotation
            break

    if not expected_model:
        # No annotation found — check if parameter/return exists without type hint
        sig = _cached_signature(model_class)
        if annotation_key in ("input", "input_data"):
            for key in (annotation_key, annotation_key + "_data"):
                if sig.parameters.get(key):
                    expected_model = True if model_str else False
                    annotation_key = key
                    break
        if annotation_key == "return":
            if sig.return_annotation is not inspect.Parameter.empty:
                expected_model = True if model_str else False

    if not expected_model:
        raise ImportFromStringError(
            f"No {annotation_key} model found in {model_class.__name__}. "
            f"Please provide the IO such as `fastmodel serve {model_class.__name__}:INPUT->OUTPUT`."
        )

    selected_model = expected_model
    if model_str:
        if hasattr(expected_model, "__name__") and model_str != expected_model.__name__:
            if not issubclass(selected_model, expected_model):
                warnings.warn(
                    f'Expected {annotation_key} model "{expected_model.__name__}" '
                    f'but got "{model_str}", proceeding but this may break the web server.'
                )
        else:
            warnings.warn(f"{model_str} will be used as the {annotation_key} model.")
        selected_model = import_class(model_str)
    elif expected_model:
        warnings.warn(
            f"No {annotation_key} model provided, using {selected_model.__name__} "
            f"found in {model_class.__name__}."
        )
    else:
        try:
            for key, value in call_annotations.items():
                if key == "return":
                    continue
                if annotation_key in value.__name__.lower():
                    selected_model = _resolve_from_annotations(model_str, model_class, key)
                    if selected_model:
                        break
        except (AttributeError, KeyError, TypeError):
            raise ImportFromStringError(
                f"No {annotation_key} model found in {model_class.__name__}."
            )

    return (annotation_key, selected_model)


def import_class(class_str: str) -> Type:
    module_str, class_name = class_str.rsplit(".", 1)
    module = importlib.import_module(module_str)
    return getattr(module, class_name)
