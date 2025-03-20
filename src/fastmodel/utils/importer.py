# This is a modified version of the uvicorn.importer module from the Uvicorn project.

import importlib
import warnings
from typing import Any, Tuple, Type


class ImportFromStringError(Exception):
    pass


def import_from_string(import_str: Any) -> Tuple[Type, Type, Type]:
    if not isinstance(import_str, str):
        return import_str

    module_str, input_model_str, output_model_str = parse_import_string(import_str)
    model_class = import_model_class(module_str)
    input_model_class = validate_model_class(input_model_str, model_class, "input")
    output_model_class = validate_model_class(output_model_str, model_class, "return")

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


def validate_model_class(
    model_str: str, model_class: Type, annotation_key: str
) -> tuple[str, Type]:
    if not hasattr(model_class, "__call__"):
        raise ImportFromStringError(f"Model {model_class.__name__} must be callable.")

    call_annotations = model_class.__call__.__annotations__
    expected_model = call_annotations.get(annotation_key)
    if not expected_model:
        annotation_key = annotation_key + "_data"
        expected_model = call_annotations.get(annotation_key)
    selected_model = expected_model
    if model_str:
        if model_str != expected_model.__name__:
            selected_model = import_class(model_str)
            if not issubclass(selected_model, expected_model):
                warnings.warn(
                    f'Expected {annotation_key} model "{expected_model.__name__}" but got "{model_str}", proceeding but this may break the web server.'
                )
    elif expected_model:
        warnings.warn(
            f"No {annotation_key} model provided, using {selected_model.__name__} found in {model_class.__name__}."
        )
    else:
        try:
            for key, value in call_annotations.items():
                if key == "return":
                    continue
                if annotation_key in value.__name__.lower():
                    selected_model = validate_model_class(model_str, model_class, key)
                    if selected_model:
                        break
        except:
            raise ImportFromStringError(
                f"No {annotation_key} model found in {model_class.__name__}."
            )
    return (annotation_key, selected_model)


def import_class(class_str: str) -> Type:
    module_str, class_name = class_str.rsplit(".", 1)
    module = importlib.import_module(module_str)
    return getattr(module, class_name)
