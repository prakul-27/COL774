import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

theta, bias =   0.0013579397686593693, 0.99663414141414

xs = list(pd.read_csv('data/linearX.csv').values.ravel())
ys = list(pd.read_csv('data/linearY.csv').values.ravel())

def normalize_data(xs, ys):
    mu, sigma = np.mean(xs), np.std(xs)
    norm_x = np.array([(x - mu)/sigma for x in xs])
    mu, sigma = np.mean(ys), np.std(ys)
    norm_y = np.array([(y - mu)/sigma for y in ys])
    return norm_x, ys

xs, ys = normalize_data(xs, ys)

plt.plot(xs, ys, 'ro', color='blue')
x = np.linspace(min(xs), max(xs), 100)
y = theta*x+bias    
plt.plot(x, y, color='red')
plt.xlabel("x")
plt.ylabel("y")

plt.show()