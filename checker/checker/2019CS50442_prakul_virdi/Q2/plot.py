import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

#plt.show()

df = pd.read_csv('r1000000.csv')
theta0, theta1, theta2 = df['theta0'].values, df['theta1'].values, df['theta2']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('theta0')
ax.set_ylabel('theta1')
ax.set_zlabel('theta2')
ax.plot(theta0, theta1, theta2, 'red')
plt.show()