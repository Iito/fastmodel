"""Tests for the importer: protocol resolution, legacy resolution, and parsing."""

import warnings

import pytest

from fastmodel.protocol import is_serving_module
from fastmodel.utils.importer import (
    ImportFromStringError,
    _get_call_input_arg,
    _resolve_from_annotations,
    _resolve_from_protocol,
    parse_import_string,
)
from tests.conftest import (
    AltArgModel,
    LegacyModel,
    ProtocolModel,
    SimpleInput,
    SimpleOutput,
)


# ── parse_import_string ──


def test_parse_module_only():
    module, inp, out = parse_import_string("mypackage.MyModel")
    assert module == "mypackage.MyModel"
    assert inp is None
    assert out is None


def test_parse_with_io():
    module, inp, out = parse_import_string("mypackage.MyModel:pkg.Input->pkg.Output")
    assert module == "mypackage.MyModel"
    assert inp == "pkg.Input"
    assert out == "pkg.Output"


def test_parse_missing_output():
    with pytest.raises(ImportFromStringError):
        parse_import_string("mypackage.MyModel:pkg.Input->")


def test_parse_missing_input():
    with pytest.raises(ImportFromStringError):
        parse_import_string("mypackage.MyModel:->pkg.Output")


def test_parse_empty():
    with pytest.raises(ImportFromStringError):
        parse_import_string(":")


# ── _get_call_input_arg ──


def test_input_arg_protocol_model():
    assert _get_call_input_arg(ProtocolModel) == "input"


def test_input_arg_alt_model():
    assert _get_call_input_arg(AltArgModel) == "input_data"


def test_input_arg_legacy_model():
    assert _get_call_input_arg(LegacyModel) == "input"


def test_input_arg_no_params():
    class NoParams:
        def __call__(self):
            pass

    with pytest.raises(ImportFromStringError, match="must accept at least one parameter"):
        _get_call_input_arg(NoParams)


# ── _resolve_from_protocol ──


def test_protocol_resolution_basic():
    cls, (in_arg, in_type), (ret_key, out_type) = _resolve_from_protocol(
        ProtocolModel, None, None
    )
    assert cls is ProtocolModel
    assert in_arg == "input"
    assert in_type is SimpleInput
    assert out_type is SimpleOutput
    assert ret_key == "return"


def test_protocol_resolution_alt_arg():
    cls, (in_arg, in_type), (ret_key, out_type) = _resolve_from_protocol(
        AltArgModel, None, None
    )
    assert in_arg == "input_data"
    assert in_type is SimpleInput
    assert out_type is SimpleOutput


# ── _resolve_from_annotations (legacy) ──


def test_legacy_resolution_input():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        key, model = _resolve_from_annotations(None, LegacyModel, "input")
    assert key == "input"
    assert model is SimpleInput


def test_legacy_resolution_return():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        key, model = _resolve_from_annotations(None, LegacyModel, "return")
    assert key == "return"
    assert model is SimpleOutput


def test_legacy_no_call_annotations():
    """A class without a user-defined __call__ has no useful annotations."""

    class NoCall:
        pass

    # The default __call__ (type's metaclass) has no __annotations__,
    # so this should fail with an AttributeError or ImportFromStringError
    with pytest.raises((ImportFromStringError, AttributeError)):
        _resolve_from_annotations(None, NoCall, "input")


def test_legacy_no_annotations():
    class Bare:
        def __call__(self, x):
            pass

    with pytest.raises(ImportFromStringError, match="No input model found"):
        _resolve_from_annotations(None, Bare, "input")


# ── Both paths produce same result ──


def test_protocol_and_legacy_agree():
    """Protocol and legacy paths resolve to the same IO types for equivalent models."""
    _, (p_arg, p_in), (_, p_out) = _resolve_from_protocol(ProtocolModel, None, None)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        l_arg, l_in = _resolve_from_annotations(None, LegacyModel, "input")
        _, l_out = _resolve_from_annotations(None, LegacyModel, "return")

    assert p_in is l_in is SimpleInput
    assert p_out is l_out is SimpleOutput
    assert p_arg == l_arg == "input"
