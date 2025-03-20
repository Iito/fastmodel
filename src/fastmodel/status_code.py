# This is the default status code that can be replaced by the user

from enum import IntEnum, unique


@unique
class StatusCode(IntEnum):
    def __new__(cls, value, message) -> "StatusCode":
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.msg = message

        return obj

    UnsetStatus = (0, "Unset Status")
    Success = (100, "Success")
    Error = (1000, "Error")
    InvalidInput = (1001, "Invalid Input")
    InvalidOutput = (1002, "Invalid Output")
    ModelNotReady = (1003, "Model Not Ready")
