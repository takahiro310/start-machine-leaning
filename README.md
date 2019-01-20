# start-machine-leaning
機械学習を理解するためのサンプル

実行にはnumpyとmatplotlibが必要なので、未インストール時は下記のようにインストールする。
```
python -m pip install -U numpy --user
python -m pip install -U matplotlib --user
```

### 回帰のサンプルコード

| ソース | 内容 |
| :-- | :-- |
| click_plot.py | data/click.csvのデータをプロット |
| linear_function.py | １次関数の学習モデル |
| polynomial_regression.py | 多項式回帰の学習モデル |
| polynomial_regression_mse.py | 多項式回帰の学習モデルに対して平均二乗誤差をプロット |
| polynomial_regression_sgd | 多項式回帰のパラメータ更新に確率的勾配降下法を適用 |

### 分類のサンプルコード

| ソース | 内容 |
| :-- | :-- |
| images1_plot.py | data/click.csvのデータをプロット |
| perceptron.py | パーセプトロンの実装 |
| logistic_regression.py | ロジスティック回帰の実装 |
| linear_inseparable.py | 線形分離不可能な分類 |
| linear_inseparable_sgd.py | 線形分離不可能な分類のパラメータ更新に確率的勾配下降法を適用 |
| regularization.py | 正則化 |
