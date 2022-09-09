import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

X = pd.read_csv('data\logisticX.csv').values.tolist()    
Y = pd.read_csv('data\logisticY.csv').values    
x1, x2 = [], []
Ylist = []

for x_1, x_2 in X:
    x1.append(x_1)
    x2.append(x_2)
for y in Y:
    Ylist.append(y[0])

def normalize_data(xs, ys):
    mu, sigma = np.mean(xs), np.std(xs)
    norm_x = np.array([(x - mu)/sigma for x in xs])
    mu, sigma = np.mean(ys), np.std(ys)
    norm_y = np.array([(y - mu)/sigma for y in ys])
    return norm_x, norm_y

x1, x2 = normalize_data(x1, x2)
tmpx1, tmpx2 = [], []
for y in range(len(Ylist)):
    if Ylist[y] == 0:
        tmpx1.append(x1[y])
        tmpx2.append(x2[y])
plt.scatter(tmpx1, tmpx2, color='blue')
tmpx1, tmpx2 = [], []
for y in range(len(Ylist)):
    if Ylist[y] == 1:
        tmpx1.append(x1[y])
        tmpx2.append(x2[y])
plt.scatter(tmpx1, tmpx2, color='red')

theta = [0.41325506, 2.4309434, -2.64095593]
x1 = np.linspace(-3,3,100) # theta1 * x1 + theta2 * x2 = 0.5 - theta0
x2 = ((0.5-theta[0]) - theta[1]*x1)/theta[2]
plt.plot(x1, x2, color='black')
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()