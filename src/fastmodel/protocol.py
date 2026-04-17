from typing import Any, ClassVar, Protocol, Type, runtime_checkable

from pydantic import BaseModel


@runtime_checkable
class ServingModule(Protocol):
    """Contract for fastmodel-compatible model classes.

    Models satisfying this protocol get direct type resolution
    from ClassVars without annotation introspection. Models that
    don't satisfy it fall back to __call__ annotation discovery.

    Example:
        class MyModel:
            MODULE_NAME: ClassVar[str] = "my_model"
            MODULE_VERSION: ClassVar[str] = "1.0.0"
            INPUT_TYPE: ClassVar[type[BaseModel]] = MyInput
            OUTPUT_TYPE: ClassVar[type[BaseModel]] = MyOutput

            def __call__(self, input: MyInput) -> MyOutput:
                ...
    """

    MODULE_NAME: ClassVar[str]
    MODULE_VERSION: ClassVar[str]
    INPUT_TYPE: ClassVar[Type[BaseModel]]
    OUTPUT_TYPE: ClassVar[Type[BaseModel]]

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


def is_serving_module(cls: type) -> bool:
    """Check if a class satisfies the ServingModule protocol.

    runtime_checkable only verifies methods, not ClassVars,
    so we check the required attributes manually.
    """
    required = ("MODULE_NAME", "MODULE_VERSION", "INPUT_TYPE", "OUTPUT_TYPE")
    return all(hasattr(cls, attr) for attr in required) and callable(cls)
