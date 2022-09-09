import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import part1 as prt1

xs, ys = prt1.read_data('data/linearX.csv', 'data/linearY.csv')
xs, ys = prt1.normalize_data(xs, ys)
theta1s = np.linspace(-2,2,10)
theta0s = np.linspace(-2,2,10)

Theta1, Theta0 = np.meshgrid(theta1s, theta0s)
Js = np.array([prt1.cost(xs, ys, theta1, theta0) for theta1, theta0 in zip(np.ravel(Theta1), np.ravel(Theta0))])
J = Js.reshape(Theta1.shape)

plt.contour(Theta0, Theta1, J)
#plt.show()
df = pd.read_csv('vals.csv')
theta0, theta1 = df['bias'], df['theta']
plt.plot(theta0, theta1, color='black')
plt.scatter(theta0, theta1, color='red')
plt.show()