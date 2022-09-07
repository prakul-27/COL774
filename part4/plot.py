import numpy as np
import matplotlib.pyplot as plt

with open('data/q4x.dat', 'r') as datFile:   
    X = []
    for data in datFile:
        X.append([float(data.split()[0]), float(data.split()[1])])        
    X = np.array(X)

with open('data/q4y.dat', 'r') as datFile:
    Y = np.array([label.strip() for label in datFile])

for i in range(2):
    X[:,i] = (X[:, i] - np.mean(X[:,i]))/np.std(X[:,i])

x1, x2, x11, x21 = [], [], [], []
for i in range(len(Y)):
    if Y[i] == 'Alaska':
        x1.append(X[i][0])
        x2.append(X[i][1])
    else:
        x11.append(X[i][0])
        x21.append(X[i][1])
plt.scatter(x1, x2, color='red')
plt.scatter(x11, x21, color='blue')
plt.show()
