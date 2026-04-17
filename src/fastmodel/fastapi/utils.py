import json
import warnings
from io import BytesIO
from typing import Any, ForwardRef, Literal, Optional, Type, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, create_model

from ..config import IMG_FORMAT

HAS_TORCH = HAS_PIL = HAS_NUMPY = HAS_TRIMESH = False

try:
    from aiohttp.client import ClientResponse
except ImportError:
    ClientResponse = ForwardRef("ClientResponse")
from importlib.metadata import version
PYDANTIC_VERSION = int(version("pydantic").split('.')[1])

def _get_serializable_data(field_name: str, data: any) -> tuple[str, bytes]:
    """
    Get the data in a serializable format for the streaming response.
    :param field_name: The name of the field.
    :param data: The data to be serialized.
    :return: A tuple containing the headers and the serialized data.
    """
    headers = (
        f'Content-Disposition: form-data; name="{field_name}"\r\n'
        f"Content-Type: application/json\r\n\r\n"
    )
    return headers, json.dumps(data).encode("utf-8")


def _get_streaming_data(field_name: str, data: Any, parse_inner=False) -> list[tuple[str, bytes]]:
    """
    Get the data in a streaming format for the streaming response.
    :param field_name: The name of the field.
    :param data: The data to be serialized.
    :param parse_inner: Whether to parse the inner data.
    :return: A list of tuples containing the headers and the serialized data.
    """
    global HAS_TORCH, HAS_PIL, HAS_NUMPY, HAS_TRIMESH

    # This is a workaround to avoid circular imports and improve performance
    try:
        import torch
        from safetensors.torch import save_file
        from torch import Tensor

        HAS_TORCH = True
    except ImportError:
        warnings.warn("torch library not found, tensor validation will not work")
    try:
        import PIL
        from PIL import Image as ImageReader
        from PIL.Image import Image

        HAS_PIL = True
    except ImportError:
        warnings.warn("PIL library not found, image processing will not work")

    try:
        import numpy as np
        from numpy import ndarray

        HAS_NUMPY = True
    except ImportError:
        warnings.warn("numpy library not found, tensor validation will not work")

    try:
        import trimesh

        HAS_TRIMESH = True
    except ImportError:
        warnings.warn("trimesh library not found, trimesh validation will not work")

    try:
        import cloudpickle as sio

    except ImportError:
        warnings.warn("cloudpickle library not found, defaulting to pickle")
        import pickle as sio

    if isinstance(data, (dict, list, tuple, set)):
        if parse_inner:
            return get_inner_data(field_name, data)
        else:
            # Using cloudpickle/pickle to serialize the data
            data = sio.dumps(data)
            filename = f"{field_name}.bin"

    content_type = "application/octet-stream"
    if HAS_PIL and isinstance(data, Image):
        raw_data = BytesIO()
        # RGBA images are not supported by JPEG
        if IMG_FORMAT == "JPEG":
            data = data.convert("RGB")

        data.save(raw_data, format=IMG_FORMAT)
        raw_data.seek(0)
        filename = f"{field_name}.{IMG_FORMAT.lower()}"
        content_type = f"image/{IMG_FORMAT.lower()}"

    elif HAS_TORCH and isinstance(data, Tensor):
        raw_data = BytesIO()
        save_file(data, raw_data)
        raw_data.seek(0)
        filename = f"{field_name}.pt"

    elif HAS_NUMPY and isinstance(data, ndarray):
        raw_data = BytesIO()
        np.save(raw_data, data)
        raw_data.seek(0)
        filename = f"{field_name}.npy"
    elif HAS_TRIMESH and isinstance(data, trimesh.base.Trimesh):
        raw_data = BytesIO()
        data.export(file_obj=raw_data, file_type="stl")
        raw_data.seek(0)
        filename = f"{field_name}.stl"
    else:
        raw_data = BytesIO(data)
        raw_data.seek(0)
        filename = f"{field_name}.bin"

    headers = (
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    )
    return [(headers, raw_data)]


def get_inner_data(field_name: str, data: Any) -> list[tuple[str, bytes]]:
    """
    Get the inner data from a dictionary, list, tuple or set.
    :param field_name: The name of the field.
    :param data: The data to be serialized.
    :return: A list of tuples containing the headers and the serialized data.
    """
    if isinstance(data, (list, tuple)):
        data_list = []
        value = (type(data).__name__, len(data))
        chunk = _get_serializable_data("super_" + field_name, value)
        data_list.append(chunk)
        for i, value in enumerate(data):
            chunk = _get_streaming_data(f"{field_name}_{i}", value)
            data_list.extend(chunk)
    else:
        for field_name, value in data.items():
            if isinstance(value, (dict, list, tuple, set)):
                data_list.extend(get_inner_data(field_name=field_name, data=value))
            else:
                chunk = _get_streaming_data(field_name=field_name, data=value)
                data_list.extend(chunk)
    return data_list


def _merge_models(
    models: Type[BaseModel] | list[Type[BaseModel]],
    request_or_response: Literal["request", "response"],
    base: Type[BaseModel],
) -> Type[BaseModel]:
    """
    Merge multiple Pydantic models into a single model in order to use them in FastAPI.
    Request model will use Form and File from FastAPI.
    Response model will use JSONResponse and StreamingResponse from FastAPI.
    :param model: The orignal Pydantic Models to generate Request/Response from.
    :param request_or_response: The type of model to generate.
    :param base: The base model to inherit from, either ApiBaseRequest or ApiBaseResponse.
    :return: The merged model-> NewModel(ApiBaseRequest, Model1, Model2, ...) or NewModel(ApiBaseResponse, Model1, Model2, ...)

    """
    if not isinstance(models, list):
        models = [models]

    is_response = request_or_response.lower() == "response"

    # Build merged model name
    merged_model_name = _build_merged_name(models, request_or_response)

    # Collect fields — response wraps in Optional[T]=None, request inherits as-is
    fields = {}
    skipping_keys = set(base.model_fields.keys()) | {"model_config", "validate", "_validate", "_args_signature"}

    for m in models:
        for field_name, field_type in get_type_hints(m).items():
            if field_name in skipping_keys or field_name.startswith("_"):
                continue

            if is_response:
                # Response: make all user fields Optional with default=None
                new_type = field_type
                field = m.model_fields[field_name]
                if field.metadata:
                    new_type = field.metadata[0].__class__[field_type]
                fields[field_name] = (Optional[new_type], None)
            else:
                # Request: preserve original type and default
                field = m.model_fields[field_name]
                fields[field_name] = (
                    field_type,
                    Field(
                        default=field.default,
                        description=field.description,
                        alias=field.alias,
                        title=field.title,
                        examples=field.examples,
                        **(field.json_schema_extra or {}),
                    ),
                )

    if is_response:
        for model in models:
            model.__pydantic_decorators__.model_validators = {}

    bases = tuple(models + [base])
    if PYDANTIC_VERSION > 10:
        model = create_model(
            merged_model_name,
            **fields,
            __config__=ConfigDict(arbitrary_types_allowed=True),
            __base__=bases,
        )
    else:
        model = create_model(
            merged_model_name,
            **fields,
            model_config=ConfigDict(arbitrary_types_allowed=True),
            __base__=bases,
        )

    if hasattr(model, "generate_validate_method"):
        model.generate_validate_method()

    return model


def _build_merged_name(
    models: list[Type[BaseModel]], request_or_response: str
) -> str:
    """Generate the merged model class name."""
    if len(models) == 1:
        return (
            models[0].__name__.replace("Input", "").replace("Output", "")
            + request_or_response.capitalize()
        )
    name = ""
    for m in models:
        model_name = m.__name__
        prefix = "From" if "Output" in model_name else ""
        postfix = "To" if "Input" in model_name else ""
        model_name = model_name.replace("Input", "").replace("Output", "")
        name += f"{prefix}{model_name}{postfix}"
    return name
