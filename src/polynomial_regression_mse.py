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

# 平均二乗誤差
def MSE(x, y):
    return (1 / x.shape[0]) * np.sum((y - f(x)) ** 2)
# 平均二乗誤差の履歴
errors = []

# 学習
errors.append(MSE(X, train_y))
while diff > 1e-2:
    # パラメータ更新
    theta = theta - ETA * np.dot(f(X) - train_y, X)
    errors.append(MSE(X, train_y))
    # 前回の誤差との差分を計算
    diff = errors[-2] - errors[-1]

x = np.arange(len(errors))
pyplot.plot(x, errors)
pyplot.show()
