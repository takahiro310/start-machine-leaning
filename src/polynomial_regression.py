import numpy as np
import matplotlib.pyplot as pyplot

train = np.loadtxt('data/click.csv', delimiter=",", skiprows=1)
train_x = train[:,0]
train_y = train[:,1]

# 標準化
mu = train_x.mean()
sigma = train_x.std()
def standardize(x):
    return (x - mu) / sigma

train_z = standardize(train_x)

# パラメータ初期化
theta = np.random.rand(3)

# 学習データの行列を作成
def to_matrix(x):
    return np.vstack([np.ones(x.shape[0]), x, x ** 2]).T

X = to_matrix(train_z)

# 予測関数
def f(x):
    return np.dot(x, theta)

# 目的関数
def E(x, y):
    return 0.5 * np.sum((y-f(x)) ** 2)

# 学習率
ETA = 1e-3

# 誤差の差分
diff = 1

# 学習
error = E(X, train_y)
while diff > 1e-2:
    # 額数データを並べ替えるためにランダムな順列を用意
    p = np.random.permutation(X.shape[0])
    # 確率的勾配降下法でパラメータ更新
    for x, y in zip(X[p,:], train_y[p]):
        theta = theta - ETA * (f(x) -y) * x
    # 前回の誤差との差分を計算
    current_error = E(X, train_y)
    diff = error - current_error
    error = current_error

pyplot.plot(train_z, train_y, 'o')
x = np.linspace(-3, 3, 100)
pyplot.plot(x, f(to_matrix(x)))
pyplot.show()
