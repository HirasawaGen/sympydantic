from pydantic._internal._validate_call import ValidateCallWrapper
from sympy import Symbol, Expr  # type: ignore[import-untyped]


def test_patchs():
    patched_doc = '!!! This method is patched.'
    call_doc = ValidateCallWrapper.__call__.__doc__
    assert isinstance(call_doc, str)
    assert call_doc.startswith(patched_doc)
    assert hasattr(Symbol, '__get_pydantic_core_schema__')
    assert hasattr(Expr, '__get_pydantic_core_schema__')
