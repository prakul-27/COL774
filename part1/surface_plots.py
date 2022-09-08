import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import part1 as prt1
import collections

xs, ys = prt1.read_data()
theta1s = np.linspace(-1e-2,1e-2,10)
theta0s = np.linspace(0,1,10)

xs, ys = prt1.normalize_data(xs, ys)
Theta1, Theta0 = np.meshgrid(theta1s, theta0s)
Js = np.array([prt1.cost(xs, ys, theta1, theta0) for theta1, theta0 in zip(np.ravel(Theta1), np.ravel(Theta0))])
J = Js.reshape(Theta1.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Theta1, Theta0, J, rstride=1, cstride=1, color='b', alpha=0.5)

ax.set_xlabel('theta1')
ax.set_ylabel('theta0')
ax.set_zlabel('cost')

#plt.show()

df = pd.read_csv('vals.csv')
theta, bias, cost = df['theta'].values, df['bias'].values, df['cost']
ax.plot(theta, bias, cost, 'red')
plt.show()