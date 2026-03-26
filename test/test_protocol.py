"""Tests for the ServingModule protocol and is_serving_module check."""

from typing import ClassVar

from pydantic import BaseModel

from fastmodel.protocol import ServingModule, is_serving_module
from test.conftest import (
    ProtocolModel,
    LegacyModel,
    AltArgModel,
    SimpleInput,
    SimpleOutput,
)


def test_protocol_model_satisfies_check():
    assert is_serving_module(ProtocolModel) is True


def test_legacy_model_does_not_satisfy():
    assert is_serving_module(LegacyModel) is False


def test_alt_arg_model_satisfies():
    assert is_serving_module(AltArgModel) is True


def test_missing_one_classvar():
    class Incomplete:
        MODULE_NAME: ClassVar[str] = "test"
        MODULE_VERSION: ClassVar[str] = "1.0"
        INPUT_TYPE: ClassVar[type] = SimpleInput
        # Missing OUTPUT_TYPE

        def __call__(self, input: SimpleInput) -> SimpleOutput:
            ...

    assert is_serving_module(Incomplete) is False


def test_not_callable():
    class NotCallable:
        MODULE_NAME: ClassVar[str] = "test"
        MODULE_VERSION: ClassVar[str] = "1.0"
        INPUT_TYPE: ClassVar[type] = SimpleInput
        OUTPUT_TYPE: ClassVar[type] = SimpleOutput

    # Still returns True because class itself is callable (can be instantiated)
    # The protocol checks callable(cls), which is True for any class
    assert is_serving_module(NotCallable) is True


def test_classvars_are_accessible():
    assert ProtocolModel.MODULE_NAME == "test.model"
    assert ProtocolModel.MODULE_VERSION == "1.0.0"
    assert ProtocolModel.INPUT_TYPE is SimpleInput
    assert ProtocolModel.OUTPUT_TYPE is SimpleOutput


def test_protocol_model_is_functional():
    model = ProtocolModel()
    result = model(SimpleInput(text="hello", count=3))
    assert result.result == "hellohellohello"
    assert result.length == 15
