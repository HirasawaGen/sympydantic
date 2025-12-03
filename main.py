from pydantic_core import PydanticCustomError
from pydantic import BaseModel, field_validator


class MyError(PydanticCustomError):
    _error_type: str = 'my_error'
    _message_template: str = 'number is {num}'
    def __new__(cls, num: int):
        return super().__new__(
            cls,
            cls._error_type,
            cls._message_template,
            {'num': num}
        )
    

class MyModel(BaseModel):
    a: int
    b: str
    @field_validator("b")
    def validate_b(cls, v):
        if len(v) < 3:
            raise MyError(len(v))
        return v


try:
    obj = MyModel(a='ab', b="abcccc")
except Exception as e:
    print(e)
    print(f'{isinstance(e, MyError)=}')
    print(f'{isinstance(e, PydanticCustomError)=}')