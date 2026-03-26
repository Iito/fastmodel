"""Tests for StatusCode enum."""

from fastmodel.status_code import StatusCode


def test_status_code_values():
    assert StatusCode.UnsetStatus == 0
    assert StatusCode.Success == 100
    assert StatusCode.Error == 1000
    assert StatusCode.InvalidInput == 1001
    assert StatusCode.InvalidOutput == 1002
    assert StatusCode.ModelNotReady == 1003


def test_status_code_messages():
    assert StatusCode.Success.msg == "Success"
    assert StatusCode.Error.msg == "Error"
    assert StatusCode.InvalidInput.msg == "Invalid Input"
    assert StatusCode.ModelNotReady.msg == "Model Not Ready"


def test_status_code_is_int():
    assert isinstance(StatusCode.Success, int)
    assert StatusCode.Success + 0 == 100
