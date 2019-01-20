import numpy as np
import matplotlib.pyplot as pyplot

train = np.loadtxt('data/click.csv', delimiter=",", skiprows=1)
train_x = train[:,0]
train_y = train[:,1]

theta0 = np.random.rand()
theta1 = np.random.rand()

# 予測関数
def f(x):
    return theta0 + theta1 * x

# 目的関数
def E(x, y):
    return 0.5 * np.sum((y-f(x)) ** 2)

# 標準化
mu = train_x.mean()
sigma = train_x.std()
def standardize(x):
    return (x - mu) / sigma

train_z = standardize(train_x)

# 学習率
ETA = 1e-3

# 誤差の差分
diff = 1

# 更新回数
count = 0

# 学習
error = E(train_z, train_y)
while diff > 1e-2:
    # 更新結果を一時変数に退避
    tmp0 = theta0 - ETA * np.sum((f(train_z) - train_y))
    tmp1 = theta1 - ETA * np.sum((f(train_z) - train_y) * train_z)
    # パラメータ更新
    theta0 = tmp0
    theta1 = tmp1
    # 前回の誤差との差分を計算
    current_error = E(train_z, train_y)
    diff = error - current_error
    error = current_error
    # ログ出力
    count += 1
    log = '{}回目: theta0 = {:.3f}, theta1 = {:.3f}, 差分 = {:.4f}'
    print(log.format(count, theta0, theta1, diff))

pyplot.plot(train_z, train_y, 'o')
x = np.linspace(-3, 3, 100)
pyplot.plot(x, f(x))
pyplot.show()
