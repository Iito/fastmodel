import json
import warnings
from inspect import isclass
from io import BytesIO
from typing import Any, ForwardRef, Literal, Optional, Type, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, create_model

from ..config import IMG_FORMAT

HAS_TORCH = HAS_PIL = HAS_NUMPY = HAS_TRIMESH = False

try:
    from aiohttp.client import ClientResponse
except ImportError:
    ClientResponse = ForwardRef("ClientResponse")


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
    model: Type[BaseModel] | list[Type[BaseModel]],
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
    if not isinstance(model, list):
        models = [model]

    request_or_response = request_or_response.lower()

    fields = {}
    merged_model_name = ""
    inherited_classes: list[str] = [base.__name__]
    imports: dict[str, list[str]] = _get_base_imports(base)
    param_code = []
    model_repr = {
        "imports": imports,
        "param_code": param_code,
        "inherited_classes": inherited_classes,
    }
    skipping_keys = ["model_config", "validate", "_validate", "_args_signature"]
    skipping_keys.extend(base.model_fields.keys())

    skipping_keys
    for m in models:
        inherited_classes.append(m.__name__)
        _get_imports(imports, m)

        model_name = m.__name__
        prefix = "From" if "Output" in model_name else ""
        postfix = "To" if "Input" in model_name else ""
        model_name = model_name.replace("Input", "").replace("Output", "")
        merged_model_name += "".join([prefix, model_name, postfix])

        for field_name, field_type in get_type_hints(m).items():
            if field_name in skipping_keys or field_name.startswith("_"):
                continue
            elif field_name in base.model_fields.keys() and base.__qualname__ == "ApiBaseResponse":
                field_info = base.model_fields[field_name]
                new_field = (field_type, field_info)
            else:
                field = m.model_fields[field_name]
                mapping = field.json_schema_extra
                new_type = field_type
                if field.metadata and base.__qualname__ == "ApiBaseResponse":
                    new_type = field.metadata[0].__class__[field_type]

                new_field = (
                    new_type if base.__qualname__ == "ApiBaseRequest" else Optional[new_type],
                    Field(
                        default=field.default if base.__qualname__ == "ApiBaseRequest" else None,
                        description=field.description,
                        alias=field.alias,
                        title=field.title,
                        examples=field.examples,
                        **mapping if mapping else {},
                    ),
                )
            fields[field_name] = new_field
            if base.__qualname__ == "ApiBaseResponse":
                _get_param_code(param_code, field_name, new_field)
                _get_imports(imports, field_type)

    if len(models) == 1:
        merged_model_name = (
            models[0].__name__.replace("Input", "").replace("Output", "")
            + request_or_response.capitalize()
        )

    if request_or_response == "response":
        # Remove custom validators from the model
        # it will allow to use the model in the response with potential None values even if the model has validators
        for model in models:
            model.__pydantic_decorators__.model_validators = {}

    model = create_model(
        merged_model_name,
        **fields,
        model_config=ConfigDict(arbitrary_types_allowed=True),
        __base__=tuple([x for x in models] + [base]),
    )
    if hasattr(model, "generate_validate_method"):
        model_repr["method_code"], model_repr["method_imports"] = model.generate_validate_method()

    model._repr = _get_class_code(merged_model_name, model_repr)
    return model


def _get_class_code(name: str, model_repr: dict) -> str:
    """
    Generate the class code from the model representation.
    param name: The name of the class.
    param model_repr: The model representation.
    return: The class code.
    """
    source_code = ""
    for _module, _imports in model_repr["imports"].items():
        if not _imports:
            source_code += f"""
import {_module}\
"""
        else:
            source_code += f"""
from {_module} import {", ".join(sorted(_imports))}
"""
    if "method_imports" in model_repr:
        for _import in model_repr["method_imports"]:
            source_code += _import.strip().rstrip().replace("\n", "") + "\n"
    param_code = [
        "\t" + x.strip().rstrip().replace("\n", "") + "\n" for x in model_repr["param_code"]
    ]
    source_code += f"""
class {name}({", ".join(model_repr["inherited_classes"])}):
{"".join(param_code)}
"""
    if "method_code" in model_repr:
        for lines in model_repr["method_code"]:
            lines = lines.replace("\n", "\n\t")
            source_code += lines

    return source_code


def _get_inner_repr(var_type) -> str:
    """
    Get the inner representation of a variable type.
    eg. Optional[str] -> str or data: str | int -> str
    :param var_type: The variable type.
    :return: The inner representation.
    """
    if hasattr(var_type, "__args__"):
        names = [_get_inner_repr(x) for x in var_type.__args__]
        annotated = None
        if "NoneType" in names:
            annotated = "Optional"
        names = [x for x in names if x != "NoneType"]
        if len(names) == 1:
            var_type = names[0] if not annotated else f"{annotated}[{names[0]}]"
        else:
            var_type = (
                f"{annotated}[{', '.join(names)}]"
                if "NoneType" in names
                else f"{annotated}[{', '.join(names)}]"
            )
    elif hasattr(var_type, "__qualname__"):
        var_type = var_type.__qualname__
    else:
        var_type = var_type.__repr__()
    return var_type


def _get_param_code(class_code: list, var_name, var_field) -> None:
    """
    Generate the code for all the parameters of the class.
    :param class_code: The class code.
    :param var_name: The variable name.
    :param var_field: The variable field.
    """
    var_type = var_field[0]
    field_info = var_field[1]

    var_type = _get_inner_repr(var_type)
    _class_code = f"{var_name}: {var_type} = Field("

    for k in field_info.__slotnames__:
        if k.startswith("_") or k == "annotation":
            continue
        attr = getattr(field_info, k)
        if attr:
            if isclass(attr):
                attr = attr.__name__
            if isinstance(attr, (tuple, list)):
                attrs = []
                for a in attr:
                    if isclass(a):
                        attrs.append(a.__name__)
                    elif hasattr(a, "value"):
                        attrs.append(str(a.value))
                    else:
                        attrs.append(repr(a))
                attr = "[" + ", ".join(attrs) + "]"

            else:
                attr = repr(attr)
            _class_code += f"{k}={attr}, "
    _class_code += "), \n"
    class_code.append(_class_code)


def _get_imports(imports: dict, imported_class: Any) -> None:
    """
    Get all the import required for the class based on the type hints.
    :param imported_class: The imported class.
    :param imports: The imports dictionary.

    """
    if not hasattr(imported_class, "__name__"):
        if hasattr(imported_class, "__args__"):
            for arg in imported_class.__args__:
                _get_imports(imports, arg)
        return
    _module = imported_class.__module__
    _name = imported_class.__name__
    if _module == "builtins":
        return

    if _module not in imports:
        imports[_module] = []

    if imported_class.__name__ not in imports[_module]:
        imports[_module].append(_name)


def _get_base_imports(inherited_model: Type[BaseModel]) -> dict:
    """
    Helper function to get the base imports to generate FastAPI Request models.
    :param inherited_model: The inherited model.
    :return: The base imports.
    """
    return {
        "io": ["BytesIO"],
        "typing": ["Any", "ClassVar", "Literal", "Optional", "Type", "Union"],
        "json": [],
        "fastapi": ["File", "Form", "UploadFile", "status"],
        "fastapi.encoders": ["jsonable_encoder"],
        "fastapi.exceptions": ["HTTPException"],
        "fastapi.responses": ["JSONResponse", "StreamingResponse"],
        "pydantic": ["BaseModel", "ConfigDict", "Field", "ValidationError", "field_validator"],
        inherited_model.__module__: [inherited_model.__qualname__],
    }
