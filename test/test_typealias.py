type StrDict[T] = dict[str, T]
type MyType = int | float


def test_mytype():
    assert isinstance(1, MyType.__value__)
    assert isinstance(1.0, MyType.__value__)
    assert not isinstance('a', MyType.__value__)

def test_strdict():
    try:
        assert isinstance({'a': 1}, StrDict[int].__value__)
        assert isinstance({'a': 1.0}, StrDict[float].__value__)
        assert not isinstance({'a': 'b'}, StrDict[int].__value__)
    except TypeError as e:
        # TypeError: isinstance() argument 2 cannot be a parameterized generic
        print(str(e))