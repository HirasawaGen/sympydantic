from pydantic._internal._validate_call import ValidateCallWrapper
import pydantic_core
from typing import Any


__all__ = []


# original method
def __call__(self, *args: Any, **kwargs: Any) -> Any:
    if not self.__pydantic_complete__:
        self._create_validators()

    res = self.__pydantic_validator__.validate_python(pydantic_core.ArgsKwargs(args, kwargs))
    if self.__return_pydantic_validator__:
        return self.__return_pydantic_validator__(res)
    else:
        return res

def __new_call__(self, *args: Any, **kwargs: Any) -> Any:
    if not self.__pydantic_complete__:
        self._create_validators()

    res = self.__pydantic_validator__.validate_python(pydantic_core.ArgsKwargs(args, kwargs), context={})
    if self.__return_pydantic_validator__:
        return self.__return_pydantic_validator__(res)
    else:
        return res


_patched_doc = '!!! This method is patched.\n'
ValidateCallWrapper.__call__ = __new_call__  # type: ignore[method-assign]
__call_doc__ = ValidateCallWrapper.__call__.__doc__
ValidateCallWrapper.__call__.__doc__ = _patched_doc + (__call_doc__ if __call_doc__ is not None else '')  # type: ignore[attr-defined]
