import numpy as np
import matplotlib.pyplot as pyplot

train = np.loadtxt('data/click.csv', delimiter=",", skiprows=1)
train_x = train[:,0]
train_y = train[:,1]

pyplot.plot(train_x, train_y, 'o')
pyplot.show()