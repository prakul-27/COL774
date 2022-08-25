import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

theta, bias =   0.6619554709942264 , 0.01

xs = list(pd.read_csv('data/linearX.csv').values.ravel())
ys = list(pd.read_csv('data/linearY.csv').values.ravel())

def normalize_data(xs, ys):
    mu, sigma = np.mean(xs), np.std(xs)
    norm_x = np.array([(x - mu)/sigma for x in xs])
    mu, sigma = np.mean(ys), np.std(ys)
    norm_y = np.array([(y - mu)/sigma for y in ys])
    return norm_x, norm_y

xs, ys = normalize_data(xs, ys)

for x, y in zip(xs, ys):
    print(x, y)

plt.plot(xs, ys, 'ro', color='blue')
x = np.linspace(min(xs), max(xs), 100)
y = theta*x+bias    
plt.plot(x, y, color='red')

plt.show()