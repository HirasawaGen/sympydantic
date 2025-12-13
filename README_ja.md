<p align="right">
  <a href="./README.md">🇬🇧 English</a> |
  <a href="./README_zh.md">🇨🇳 简体中文</a> |
  <a href="./README_ja.md">🇯🇵 日本語</a>
</p>

# Sympydantic プロジェクトガイド

## プロジェクトの背景

こんな経験ありませんか？🧐  
午後ずーっとCNNを訓練させてきたのに、**行列の転置を忘れただけ**で前のテンソルの列数と次のテンソルの行数が合わなくてエラーが出て、**すべてがパー！**😭

あるいは……**同僚**が書いたユーティリティ関数が `numpy` 配列を返すのに型ヒントは `np.int8` だし、あなたはその要素を**リストのインデックス**に使おうとして実行したら、実際の型は `np.float64` だった！🙀

あるいは、**強化学習**の環境が状態を `tuple[int]` で返しているのに、開発者が**型アノテーションを書いていなくて**、つい `numpy` 配列だと思って `arr.argmax()` なんて呼んだら、**真っ赤なエラー**が出てうんざり😨

あなたは大手企業の**シニア技術アドバイザー**かもしれないし、**一流大学**の博士課程の学生かもしれない。複雑な**数式**なんてお手のもの、楽勝です。でも、ここで `unsqueeze` がなくて、そこで `reshape` を忘れて……こんな単純なのに、こんなに怒りたくなるミス！🤬

GitHubの某所で三流不**有名プログラマー・HirasawaGen**曰く：
> 「ディープラーニングで**形状の問題**をクリアすれば、すべての問題の半分は解決する。」😎

`sympypdantic`を使いましょう！有名なPythonライブラリ `pydantic` と `sympy` を組み合わせて、`numpy` や `torch` のテンソル形状を**関数実行前に検証**します。  
数時間後にアニメを見ながらマクドナルド🍔🍟を食べてるときに出るはずのエラーを、**実行ボタン▶を押して数秒後**に出して、あなたが**従従容容で遊刃有餘に**対処できるように！

## 依存関係

- Python ≥ 3.12  
- Pydantic（データ検証ライブラリ）≥ 2.12.4  
- SymPy（数式処理ライブラリ）≥ 1.14.0  

> 注意：`sympydantic`は現在開発中のため、**Python 3.12以上**のみサポートしています😩。将来的に3.8〜3.11にも対応予定です。

## 主要機能

### 自動形状検証

#### デモ1

名前の通り、`sympy` と `pydantic` の良いとこ取り！

```python
from typing import Annotated
import numpy as np
import sympy as sp
from pydantic import validate_call

from sympydantic import TensorLike   # torch.Tensor と numpy.ndarray の両方に対応
from sympydantic import tensorshape  # テンソル形状を指定するメタデータ

X = sp.symbols('X')

@validate_call
def foo(
    arg: Annotated[TensorLike, tensorshape[2, X, X + 2]],
) -> None:
    # !! もし arg.shape が表示されなければ、pydanticが関数実行前に検証で弾いたということ
    print(arg.shape)
    assert arg.shape[0] == 2
    assert arg.shape[1] + 2 == arg.shape[2]

if __name__ == '__main__':
    arg1 = np.random.rand(2, 3, 5)
    arg2 = np.random.rand(1, 3, 5)  # 第1次元が2でない
    arg3 = np.random.rand(2, 3, 4)  # 第2・3次元が条件を満たさない

    foo(arg1)  # OK

    try:
        foo(arg2)
    except Exception as e:
        print(e)  # dimension 0 has length 1, expected 2 (int)

    try:
        foo(arg3)
    except Exception as e:
        print(e)  # The expression 'X + 2' is solved as 5, which conflicts with the provided value 4.

''' Terminal Output:
(2, 3, 5)
1 validation error for foo
0
  dimension 0 has length 1, expected 2 (int) 
  [type=shape_conflict, input_value=array([[[0.22684143, 0.50...66766634, 0.46905961]]]), input_type=ndarray]
1 validation error for foo
0
  The expression 'X + 2' is solved as 5, which is conflict with the provided value 4. 
  [type=expr_conflict, input_value=array([[[0.59563589, 0.36...08101385, 0.58254737]]]), input_type=ndarray]
'''
```

`arg1` の形状 `(2, 3, 5)` は `(2, X, X+2)` を満たすため通過。  
`arg2`, `arg3` は違反のため、関数実行前にエラーを出します。

#### デモ2

`sympy.Symbol` を使いたくない場合は `TypeVar` でもOK：

```python
from typing import Annotated
import numpy as np
from pydantic import validate_call

from sympydantic import TensorLike, tensorshape

@validate_call
def foo[X](
    arg: Annotated[TensorLike, tensorshape[X, X]],
) -> None:
    print(arg.shape)
    assert arg.shape[0] == arg.shape[1]

if __name__ == '__main__':
    arg1 = np.random.rand(3, 3)
    arg2 = np.random.rand(3, 4)

    foo(arg1)  # OK

    try:
        foo(arg2)
    except Exception as e:
        print(e)  # The symbol 'X' is already set to 3. You provided a conflicting value 4.

''' Terminal Output:
(3, 3)
1 validation error for foo
0
  The symbol 'X' is already set to 3. you provide a conflict value 4.
  [type=symbol_redefined, input_value=array([[0.40639904, 0.541....92482645, 0.0740373 ]]), input_type=ndarray]
'''
```

#### デモ3

`TypeVar` では四則演算が使えません。それでも `sympy` をインポートしたくない場合は、  
`tensorshape['X', 'X+1']` と直接書けます。数値、文字列、`slice` オブジェクトも混在可能です：

```python
from typing import Annotated
import numpy as np
from sympy.abc import X, Y
from pydantic import validate_call

from sympydantic import TensorLike, tensorshape

@validate_call
def foo(
    value_Y: Annotated[int, Y],              # symbol Y の値を引数にバインド
    arg1: Annotated[TensorLike, tensorshape[X, X:10, '*']],
    arg2: Annotated[TensorLike, tensorshape[..., '2 * Y - 1']],
) -> None:
    print(arg1.shape)
    print(arg2.shape)
    _solve_X = arg1.shape[0]
    assert _solve_X <= arg1.shape[1] < 10   # スライスに数値を混在
    assert arg2.shape[-1] == 2 * value_Y - 1  # 最終次元のみ検証
    # pydantic は arg1.ndim == 3 を確認するが、arg2.ndim は問わない
```

備考  
- `'*'` とアノテートされた次元は**完全に無視**され、値の検証も保存も行われません。  
- 通常、sympydantic はまず**ndim**をチェックします。例えば期待形状 `(X, X:10, Y)` に対して `(1,2,3,4,5)` を渡すと、次元数が違うため即座に拒否されます。  
- エリプシス `...` は「前後だけをチェックし、中間は自由」と言う意味です。  
  例：`(1, ..., X+2)` は最初と最後の次元を検証し、中間は何でもOK。  
- もし**ちょうど 5 次元**を保ちつつ両端だけ検証したい場合は `(X, '*', '*', '*', 2*X)` と書けます。

### 自動型変換

#### デモ4

`TensorLike` は変換せず検証のみ。  
変換を使いたい場合は `NDArray` や `Tensor` を使います：

```python
from typing import Annotated
import numpy as np
import torch
from pydantic import validate_call

from sympydantic import TensorLike, Tensor, NDArray
from sympydantic.metadatas.device import CUDA

@validate_call
def foo(
    original_arr: TensorLike,
    numpy_arr: Annotated[NDArray[np.bool], 'meta'],
    torch_arr: Annotated[Tensor, CUDA],
):
    print(original_arr)  # 変換なし
    print(numpy_arr)     # numpy.ndarray に変換 + dtypeもboolに
    print(torch_arr)     # torch.Tensor に変換 + CUDAに移動
    
''' Terminal Output:
[0.71413676 0.09614301 0.04009426]
[ True  True  True]
tensor([0.7141, 0.0961, 0.0401], device='cuda:0', dtype=torch.float64)
tensor([0.1790, 0.4157, 0.8533])
[ True  True  True]
tensor([0.1790, 0.4157, 0.8533], device='cuda:0')
'''

```

#### デモ5

リストやスカラも渡せます：

```python
from typing import Annotated
import numpy as np
import torch
from pydantic import validate_call

from sympydantic import TensorLike, Tensor, NDArray
from sympydantic.metadatas.device import CUDA

@validate_call
def foo(
    numpy_arr: Annotated[NDArray[np.bool], 'meta'],
    torch_arr: Annotated[Tensor, CUDA],
) -> None:
    print(numpy_arr)
    print(torch_arr)

if __name__ == '__main__':
    arr = [1, 2, 3]
    foo(arr, arr)
    foo(3, 9)


''' Terminal Output:
[ True  True  True]
tensor([1, 2, 3], device='cuda:0')
True
tensor(9, device='cuda:0')
'''

```

強化学習環境が `tuple` を返しても `ndarray` を返しても、もう心配不要！

## TODOs

- 自動`squeeze`対応：たとえば `(1,3,4,5)` を期待しているのに `(3,1,4,5,1)` が来たら自動でリサイズ  
- ブロードキャスト対応：たとえば `(3,4,4)` を期待しているのにスカラが来たら自動でブロードキャスト  
- Python 3.8〜3.11 への対応
