import numpy as np
import matplotlib.pyplot as pyplot

train = np.loadtxt('data/images1.csv', delimiter=",", skiprows=1)
train_x = train[:,0:2]
train_y = train[:,2]

pyplot.plot(train_x[train_y == 1, 0], train_x[train_y == 1, 1], 'o')
pyplot.plot(train_x[train_y == -1, 0], train_x[train_y == -1, 1], 'x')
pyplot.axis('scaled')
pyplot.show()