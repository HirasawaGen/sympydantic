from typing import NoReturn, Any

from sympy import Expr, Symbol  # type: ignore[import-untyped]
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema


class _Expr(Expr):
    _THRESHOLD = 1e-10
    
    def __get_pydantic_core_schema__(
        self,
        source_type: type,
        handlers: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        '''
        This method is patched.
        '''
        assert True
        return core_schema.with_info_after_validator_function(
            self._validate,
            handlers(source_type),
        )
    
    def _validate(
        self,
        value: Any,
        info: core_schema.ValidationInfo,
    ) -> int | float:
        '''
        This method is patched.
        '''
        if not isinstance(value, (int, float)):
            raise ValueError(f"Value {value} is not a number.")
        context = info.context
        if context is None:
            return value
        if not 'sympy_namespace' in context.keys():
            context['sympy_namespace'] = {}
        sympy_namespace = context['sympy_namespace']
        if isinstance(self, Symbol):
            if self.name not in sympy_namespace.keys():
                # save in sympy_namespace
                sympy_namespace[self.name] = value
                return value
            # already in sympy_namespace
            if abs(sympy_namespace[self.name] - value) > self._THRESHOLD:
                raise ValueError(f"Symbol {self.name} already exists in sympy_namespace with different value.")
            return value
        # self is a Expr obj, but not a Symbol
        free_symbols = {symbol.name for symbol in self.free_symbols}
        diff_symbols = free_symbols - sympy_namespace.keys()
        if len(diff_symbols):
            raise ValueError(f"Symbols {diff_symbols} are not in sympy_namespace.")
        solved = self.subs(sympy_namespace)
        if not solved.is_number:
            raise ValueError(f"Cannot solve {self} with sympy_namespace {sympy_namespace}.")
        solved = float(solved)
        if abs(solved - value) > self._THRESHOLD:
            raise ValueError(f"Expression {self} not equal to {value}.")
        return value
        
    def __index__(self) -> NoReturn:
        '''
        This method is patched.
        '''
        raise NotImplementedError("`__index__` is a monkey patch, it is not implemented.")


Expr._THRESHOLD = _Expr._THRESHOLD
Expr._validate = _Expr._validate
Expr.__get_pydantic_core_schema__ = _Expr.__get_pydantic_core_schema__
