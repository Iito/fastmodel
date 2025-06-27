import hmac
import json
import warnings
from base64 import b64decode, b64encode
from copy import deepcopy
from http import HTTPStatus
from io import BytesIO
from pprint import pprint
from typing import (
    Any,
    ClassVar,
    ForwardRef,
    Generator,
    Literal,
    Optional,
    Type,
    Union,
    get_type_hints,
)
from urllib.error import HTTPError

from fastapi import File, Form, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, SkipValidation, ValidationError, create_model
from pydantic.errors import PydanticInvalidForJsonSchema

from ..config import IMG_FORMAT, STREAMING_BOUNDARY, STREAMING_CHUNK_SIZE
from ..status_code import StatusCode
from .utils import (  # create_optional_model,; create_request_model,; create_response_model,
    _get_serializable_data,
    _get_streaming_data,
    _merge_models,
    get_inner_data,
)

warnings_str = "\n"
try:
    from aiohttp.client import ClientResponse
except ImportError:
    warnings_str += "- aiohttp library not found, parsing streamed data will not work\n"
    ClientResponse = ForwardRef("ClientResponse")

try:
    import PIL
    from PIL import Image as ImageReader
    from PIL.Image import Image

    has_pil = True
except ImportError:
    has_pil = False
    warnings_str += "- PIL library not found, image validation will not work\n"

try:
    import torch
    from safetensors.torch import save_file
    from torch import Tensor

    has_torch = True
except ImportError:
    has_torch = False
    warnings_str += "- torch library not found, tensor validation will not work\n"

try:
    import numpy as np
    from numpy import ndarray

    has_numpy = True
except ImportError:
    has_numpy = False
    warnings_str += "- numpy library not found, numpy validation will not work\n"

try:
    import trimesh

    has_trimesh = True
except ImportError:
    has_trimesh = False
    warnings_str += "- trimesh library not found, trimesh validation will not work\n"

try:
    import cloudpickle as sio

except ImportError:
    import pickle as sio

    warnings_str += "- cloudpickle library not found, defaulting to pickle\n"

if warnings_str != "\n":
    warnings.warn(warnings_str)


class APIBase(BaseModel):
    """
    Base class for both Request and Response pydantic models
    """

    def process(self, **kwargs):
        # Method to convert the request to the required instance
        return self.returns.model_validate(BaseModel.model_dump(self, **kwargs))

    def get_model_arg_signature(self):
        return self._get_model_arg_signature(self)

    @classmethod
    def __get_model_arg_signature(cls, model: Type[BaseModel]):
        media_type = "application/json"
        base_model_form = cls._get_model_arg_signature(model)

        if all([v[1] == Form for k, v in base_model_form.items()]):
            form = (str, Form)
        elif all([v[1] == File for k, v in base_model_form.items()]):
            form = (UploadFile, File)
            media_type = "multipart/form-data"
        else:
            form = media_type = None
        return form, media_type

    @classmethod
    def _get_union_field_arg_signature(cls, field_type: tuple):
        media_type = "application/json"
        is_optional = True if field_type[-1] == type(None) else False
        extra_types = None

        if is_optional and len(field_type) == 2:
            # This is an Optional field, the first element is the actual type, the second one is None
            field_type = field_type[0]
            if field_type == BaseModel or type(field_type) == type(BaseModel):
                form, media_type = cls.__get_model_arg_signature(field_type)
            else:
                _type = cls.get_arg_signature(field_type)
                form = (_type, File if _type == UploadFile or _type == bytes else Form)
        else:
            # This is a proper Union field
            # Assert all type in Form or File type but not mixed
            form, media_type = cls._get_list_field_arg_signature(field_type)
            if not all([x == field_type[0] for x in field_type]):
                extra_types = field_type
        return form, media_type, extra_types

    @classmethod
    def _get_list_field_arg_signature(cls, field_type_list: Type[list]):
        arg_signature = []
        media_type = "application/json"

        for field_type in field_type_list:
            if (
                field_type == BaseModel
                or type(field_type) == type(BaseModel)
                or isinstance(field_type, BaseModel)
            ):
                model_form, media_type = cls.__get_model_arg_signature(field_type)
                if not model_form or not media_type:
                    continue
                _field_type = model_form[0]

            else:
                _field_type = cls.get_arg_signature(field_type)

            arg_signature.append(_field_type)

        if all([x == UploadFile for x in arg_signature]):
            form = (UploadFile, File)
            media_type = "multipart/form-data"
        elif all([x == bytes for x in arg_signature]):
            form = (bytes, File)
            media_type = "multipart/form-data"
        elif all([x != UploadFile for x in arg_signature]):
            form = (str, Form)
        else:
            form = media_type = None

        # Even tho Union will use that method, it is unlikely that multiple field will have the same type
        #  except if we expect a list of the same type
        if all([x == field_type_list[0] for x in field_type_list]):
            form = ([form[0]], form[1])

        return form, media_type

    @classmethod
    def _get_model_arg_signature(cls, model: Type[BaseModel]):
        args_signature = {}
        type_hints = get_type_hints(model)
        type_hints = {k: v for k, v in type_hints.items() if k in model.model_fields}

        for field_name, field_type in type_hints.items():
            media_type = "application/json"

            if (
                field_name.startswith("_")
                or field_name == "model_config"
                or field_name == "validate"
            ):
                continue
            if type(field_type) == type(ClassVar) or field_type == classmethod:
                continue
            if (
                field_type == BaseModel
                or type(field_type) == type(BaseModel)
                or isinstance(field_type, BaseModel)
            ):
                form, media_type = cls.__get_model_arg_signature(field_type)
                if not form or not media_type:
                    raise ValueError(
                        f"Form data cannot handled BaseModel fields with mixed Form and File types\n {field_name}:{field_type}"
                    )

            elif hasattr(field_type, "__origin__"):
                # This is an Optional or Union field

                form, media_type, extra_types = cls._get_union_field_arg_signature(
                    field_type.__args__
                )

                if not form or not media_type:
                    raise ValueError(
                        f"Form data cannot handled fields with mixed Form and File types\n{field_name}:{field_type}"
                    )
                field_type = extra_types or field_type.__args__[0]

            elif hasattr(field_type, "__args__"):
                # This is a list, tuple or Union made with | operator
                # eg: my_var: str | int or my_var: str | None
                form, media_type, extra_types = cls._get_union_field_arg_signature(
                    field_type.__args__
                )
                field_type = extra_types or field_type

            else:
                _type = cls.get_arg_signature(field_type)
                form = (_type, File if _type == UploadFile or _type == bytes else Form)
                media_type = None

            field_data = model.model_fields[field_name]
            if field_data.is_required():
                required = ...
            elif field_data.default:
                required = field_data.default.__repr__()
            else:
                required = None

            args_signature[field_name] = (
                form[0],
                form[1],
                field_type,
                {
                    "required": required,
                    "description": field_data.description,
                    "alias": field_data.alias,
                    "title": field_data.title,
                    "examples": field_data.examples,
                    "media_type": media_type,
                },
            )
        return args_signature

    @staticmethod
    def get_arg_signature(field_type):
        _type = field_type
        if field_type == BaseModel or type(field_type) == type(BaseModel):
            raise ValueError("BaseModel cannot be used as a field type")
        if has_pil and field_type == Image:
            _type = UploadFile
        if has_torch and field_type == Tensor:
            _type = UploadFile
        if has_numpy and field_type == ndarray:
            _type = UploadFile
        if has_trimesh and field_type == trimesh.base.Trimesh:
            _type = UploadFile
        return _type

    @staticmethod
    def _get_model_arg_signature_repr(args_signature: tuple) -> str:
        lines = ""
        for arg, value in args_signature.items():
            form_field_type = value[0]
            form_data = value[1]
            form_data_args: dict = value[3]

            form_field_type_str = (
                form_field_type.__qualname__
                if not isinstance(form_field_type, (list, tuple))
                else f"list[{form_field_type[0].__qualname__}]"
            )
            lines += f"{arg}: {form_field_type_str} = {form_data.__name__}("
            lines += f"{form_data_args.pop('required')}, "
            form_args = []
            for form_arg, form_value in form_data_args.items():
                if (form_arg == "example" or form_arg == "examples") and form_value:
                    if not isinstance(form_value, (list, tuple)):
                        form_value = [form_value]
                    for i in range(len(form_value)):
                        if issubclass(form_value[i].__class__, BaseModel):
                            form_value[i] = form_value[i].model_dump()
                form_args.append(f"{form_arg}={form_value.__repr__()}")
            lines += ", ".join(form_args)
            lines += "), \n"

        return lines


class ApiBaseRequest(APIBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _validate: classmethod
    _args_signature: ClassVar[dict]

    @staticmethod
    async def read_image(img_data: str | UploadFile) -> bytes:
        if isinstance(img_data, str):
            data = b64decode(img_data.split("base64,")[-1])
            img = ImageReader.open(BytesIO(data))
        else:
            content_type = img_data.content_type
            img_format = content_type.split("/")[-1]
            img_buff = BytesIO(await img_data.read())
            img_buff.seek(0)
            img = ImageReader.open(img_buff)
            if img.mode != "RGB":
                img = img.convert("RGB")

            if img_format != img.format.lower():
                img.save(img_buff, format=img_format)
                img_buff.seek(0)
                img = ImageReader.open(img_buff)

        return img

    @classmethod
    def generate_validate_method(cls, save: bool = False) -> None:
        args_signature_full = cls._get_model_arg_signature(cls)

        method_args_signature_str = cls._get_model_arg_signature_repr(args_signature_full)
        cls_args_signature_str = ", ".join([f"{k}={k}" for k in args_signature_full.keys()])
        img_fields = {}
        base_model_fields = {}
        for k, v in args_signature_full.items():
            expected_type = v[2]
            is_image = False
            is_base_model = False
            if isinstance(expected_type, (list, tuple)):
                if has_pil and all([x == Image for x in expected_type]):
                    is_image = True
                if all([issubclass(x, BaseModel) for x in expected_type]):
                    is_base_model = True
            else:
                if has_pil and issubclass(expected_type, Image):
                    is_image = True
                if issubclass(expected_type, BaseModel):
                    is_base_model = True
                    expected_type = [expected_type]
            assert not (is_image and is_base_model), "Cannot have mixed Image and BaseModel fields"

            if is_image:
                img_fields[k] = True

            if is_base_model:
                base_model_fields[k] = expected_type
        validation_method_body = []
        imports_str = []
        validation_method_body.append(
            f"""
async def validate(cls, {method_args_signature_str}):
"""
        )
        for _, v in args_signature_full.items():
            classes = v[2]
            if not isinstance(classes, (list, tuple)):
                classes = [classes]
            for _class in classes:
                _class_name = _class.__name__
                _module = _class.__module__
                if _module == "builtins":
                    continue
                imports_str.append(
                    f"""
    from {_module} import {_class_name}\
"""
                )

        if any(img_fields):
            for field_name in img_fields:
                validation_method_body.append(
                    f"""
    if isinstance({field_name}, (list, tuple)):
        if {field_name}:
            {field_name} = [await cls.read_image(img) for img in {field_name}]
    elif {field_name}:
        {field_name} = await cls.read_image({field_name})
"""
                )
        if any(base_model_fields):
            for field_name, baseModelClasses in base_model_fields.items():
                validation_method_body.append(
                    f"""
    if isinstance({field_name}, (list, tuple)):
        if {field_name}:\
"""
                )
                for baseModelClass in baseModelClasses:
                    validation_method_body.append(
                        f"""
            try:
                {field_name} = [json.loads(x) if isinstance(x, str) else x for x in {field_name}]
                {field_name} = [{baseModelClass.__qualname__}.model_validate(x) for x in {field_name}]
            except ValidationError as e:
                pass\
"""
                    )
                validation_method_body.append(
                    f"""
    elif {field_name}:\
"""
                )
                for baseModelClass in baseModelClasses:
                    validation_method_body.append(
                        f"""
        try:
            {field_name} = json.loads({field_name}) if isinstance({field_name}, str) else {field_name}
            {field_name} = {baseModelClass.__qualname__}.model_validate({field_name})
                
        except ValidationError as e:
            pass
"""
                    )
                validation_method_body.append(
                    f"""
        if not {field_name}:
            raise HTTPException(
                detail=jsonable_encoder(e.errors()) if e else None,
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )
"""
                )
        validation_method_body.append(
            f"""
    return cls({cls_args_signature_str})
"""
        )

        local_namespace = {}
        method_body = deepcopy(validation_method_body)
        method_body.insert(1, "".join(imports_str))
        method_body = "".join(method_body)
        if save:
            with open(f"{cls.__name__}_validate.py", "w") as f:
                f.write(method_body)
        exec(method_body, globals(), local_namespace)
        cls._validate = classmethod(local_namespace["validate"])
        return validation_method_body, imports_str


class ApiBaseResponse(APIBase):
    status: int = Field(
        default=0,
        description="The status of the response",
        examples=[2000],
    )
    message: Optional[str] = Field(
        default=None,
        description="The message of the response",
        examples=["Success"],
    )
    _version: ClassVar[str]
    version: Optional[str] = Field(
        default=None,
        description="The version of the API set by the internal variable '_version'",
        examples=["0.1.0"],
    )
    _streaming_media_type: ClassVar[str] = f"multipart/form-data; boundary={STREAMING_BOUNDARY}"
    _requires_streaming: bool = False
    force_json: ClassVar[bool] = False

    def __init__(self, **data) -> None:
        if "version" not in data:
            data["version"] = self._version if hasattr(self, "_version") else None
        super().__init__(**data)

    @classmethod
    def from_model(
        cls,
        model,
        status: StatusCode = StatusCode.UnsetStatus,
        message: str = None,
    ) -> "ApiBaseResponse":
        if not isinstance(model, cls.returns):
            if isinstance(model, dict):
                model = cls.returns.model_validate(model)
            elif isinstance(model, str):
                model = cls.returns.model_validate_json(model)
            else:
                raise ValueError(f"Expected {cls.returns.__name__}, got {model.__class__.__name__}")
        new_model = model.model_dump()
        new_model.update({"status": status, "message": message or status.msg})
        return cls.model_validate(new_model)

    @classmethod
    async def from_response(cls, response: ClientResponse):
        """
        Parse the HTTP response and return the model instance

        Args:
            response (ClientResponse): [description]

        Returns:
            [type]: [description]

        """

        headers = dict(response.headers)

        if "Content-Type" in headers:
            if "boundary" in headers["Content-Type"]:
                return cls._from_streamed_data(await response.read())
            elif "application/json" in headers["Content-Type"]:
                return cls.model_validate(await response.json())

    @classmethod
    def _from_streamed_data(cls, data: bytes):
        """
        Unwrap the data that were streamed by the generate_streaming_response method.
        """
        model_fields = {x: None for x in cls.model_fields}
        boundary = f"--{STREAMING_BOUNDARY}"
        parts = data.split(boundary.encode("utf-8"))
        received_data = {}
        for part in parts:
            if not part.strip():
                continue
            headers, body = part.split(b"\r\n\r\n", 1)
            headers = headers.decode("utf-8").lower()
            if f"Content-Type: image/{IMG_FORMAT}".lower() in headers:
                img_field = headers.split('name="')[1].split('"')[0]
                img_bytes = BytesIO(body)
                if has_pil:
                    img = ImageReader.open(img_bytes)
                else:
                    img = img_bytes
                received_data[img_field] = img
            else:
                field_name = headers.split('name="')[1].split('"')[0]
                if "application/json" in headers:
                    received_data[field_name] = json.loads(body.decode("utf-8"))
                else:
                    # This body is a bytes object, we do not decode it, it will be sent as is
                    received_data[field_name] = body

        # Reconstruct nested fields
        if "parsed_inner" in received_data:
            parsed = received_data["parsed_inner"]

        if parsed:
            for field_name, _ in get_type_hints(cls).items():
                if field_name.startswith("_"):
                    continue
                inner_type = (
                    cls.model_fields[field_name].annotation.__args__[0]
                    if hasattr(cls.model_fields[field_name].annotation, "__args__")
                    else cls.model_fields[field_name].annotation
                )
                if field_name in received_data and isinstance(
                    received_data[field_name], (inner_type, bytes)
                ):
                    model_fields[field_name] = received_data[field_name]
                elif "super_" + field_name in received_data:
                    if isinstance(received_data["super_" + field_name], (list, tuple)):
                        expected_type, loose_data = received_data["super_" + field_name]
                        if expected_type == list.__name__ or expected_type == tuple.__name__:
                            model_fields[field_name] = [
                                received_data[f"{field_name}_{i}"] for i in range(loose_data)
                            ]
                            if expected_type == tuple:
                                model_fields[field_name] = tuple(model_fields[field_name])

                        elif expected_type == dict or expected_type == set:
                            nested_field = {}
                            for i in range(loose_data):
                                inner_field_name = f"{field_name}_{i}"
                                if inner_field_name in received_data:
                                    nested_field[inner_field_name] = received_data[inner_field_name]
                            model_fields[field_name] = nested_field
                            if expected_type == set:
                                model_fields[field_name] = set(model_fields[field_name])
                else:
                    model_fields[field_name] = received_data[field_name]
        else:
            for field_name, _ in cls.model_fields.items():
                model_fields[field_name] = received_data.get(field_name)
        return cls.model_validate(model_fields)

    @classmethod
    def model_json_schema(cls) -> dict:
        """
        Get the JSON schema of the model
        """
        try:
            return super().model_json_schema()
        except PydanticInvalidForJsonSchema:
            json_schema = {}
            for key, value in cls._get_model_arg_signature(cls).items():
                examples = value[3].get("examples")
                field_type = value[2]

                if examples:
                    json_schema[key] = examples[0]
                else:
                    if isinstance(field_type, (list, tuple)):
                        _description = []
                        for x in field_type:
                            _description_type = cls.get_inner(x)
                            if isinstance(_description_type, dict):
                                _description.append(_description_type["properties"])
                            else:
                                _description.append(_description_type)

                        json_schema[key] = " | ".join(
                            [json.dumps(x) if isinstance(x, dict) else x for x in _description]
                        )
                    else:
                        json_schema[key] = cls.get_inner(field_type)
        return json_schema

    @classmethod
    def get_inner(cls, field):
        if issubclass(field, BaseModel):
            try:
                _json_schema = field.model_json_schema()
            except PydanticInvalidForJsonSchema:
                model_x = cls._get_model_arg_signature(field)
                _json_schema = {k: v[2] for k, v in model_x.items()}
                for key in _json_schema.keys():
                    if isinstance(_json_schema[key], (tuple, list)):
                        _json_schema[key] = " | ".join([x.__qualname__ for x in _json_schema[key]])
        else:
            _json_schema = field.__qualname__
        return _json_schema

    def generate_streaming_response(self, parse_inner=False) -> StreamingResponse:
        """
        Generate the streaming response from the model as a multipart form-data
        """
        serialized_status = self.check_json_serializable()
        if not self._requires_streaming:
            return JSONResponse(self.model_dump())
        elif self.force_json:
            return JSONResponse(self.convert_to_base64(serialized_status))
        else:
            return StreamingResponse(
                self._stream_data(parse_inner=parse_inner, serialized_status=serialized_status),
                media_type=self._streaming_media_type,
            )

    def convert_to_base64(self, serialized_status: dict) -> dict:
        output_data = {}
        for field_name, serializable in serialized_status.items():
            if serializable:
                output_data[field_name] = getattr(self, field_name)

            else:
                data = getattr(self, field_name)
                json_friendly_data = None
                for method in [
                    "json",
                    "dict",
                    "to_dict",
                    "tojson",
                    "to_json",
                    "jsonable_encoder",
                    "getvalue",
                    "getdata",
                    "get_data",
                    "tobytes",
                    "tojson",
                    "to_bytes",
                    "read",
                ]:
                    try:
                        json_friendly_data = getattr(data, method)()
                        if not isinstance(
                            json_friendly_data, (dict, list, tuple, set, str, int, float, bool)
                        ):
                            json_friendly_data = b64encode(json_friendly_data).decode("utf-8")
                        break
                    except AttributeError:
                        pass
                if not isinstance(
                    json_friendly_data, (dict, list, tuple, set, str, int, float, bool)
                ):
                    for method in ["save", "export"]:
                        try:
                            buff = BytesIO()
                            getattr(data, method)(buff)
                            buff.seek(0)
                            json_friendly_data = b64encode(buff.read()).decode("utf-8")
                            break
                        except AttributeError:
                            pass

                output_data[field_name] = json_friendly_data

        return output_data

    async def _stream_data(self, serialized_status: dict, parse_inner=False):
        # Get bytes, images, tensor and non serialized data from the model
        yield f"--{STREAMING_BOUNDARY}\r\n".encode("utf-8")
        data_to_stream = []
        for field_name, serializable in serialized_status.items():
            if serializable:
                data_to_stream.append(_get_serializable_data(field_name, getattr(self, field_name)))
            else:
                data_to_stream.extend(
                    _get_streaming_data(
                        field_name, getattr(self, field_name), parse_inner=parse_inner
                    )
                )

        parsed_inner_chunk = _get_serializable_data("parsed_inner", parse_inner)
        data_to_stream.insert(0, parsed_inner_chunk)

        for headers, raw_data in data_to_stream:
            yield headers.encode("utf-8")
            if isinstance(raw_data, BytesIO):
                for chunk in iter(lambda: raw_data.read(STREAMING_CHUNK_SIZE), b""):
                    yield chunk
            else:
                yield raw_data
            yield f"\r\n--{STREAMING_BOUNDARY}\r\n".encode("utf-8")
        # yield f"--{STREAMING_BOUNDARY}--\r\n".encode("utf-8")

    def check_json_serializable(self) -> dict[str, bool]:
        """
        Checks the JSON serializability of the model's fields.

        :return: A dictionary where the keys are the field names and the values are booleans
                 indicating if the field is JSON serializable.
        """
        serializable_status = {}
        for field_name, _ in self.model_fields.items():
            value = getattr(self, field_name)
            try:
                # Try to serialize the value to JSON
                json.dumps(value)
                serializable_status[field_name] = True
            except TypeError:
                serializable_status[field_name] = False
                self._requires_streaming = True
        return serializable_status


def create_request_model(model: Type[BaseModel] | list[Type[BaseModel]]) -> Type[BaseModel]:
    return _merge_models(model, "request", ApiBaseRequest)


def create_response_model(model: Type[BaseModel] | list[Type[BaseModel]]) -> Type[BaseModel]:
    return _merge_models(model, "response", ApiBaseResponse)


__all__ = [
    ApiBaseRequest,
    ApiBaseResponse,
    create_request_model,
    create_response_model,
]
