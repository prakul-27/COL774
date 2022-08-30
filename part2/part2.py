import csv
import numpy as np
import pandas as pd

def h(theta, x):
    epsilon = np.random.normal(0, 2, 1)[0]
    return np.dot(theta, x) + epsilon

def sample_data():
    theta = np.array([3, 1, 2])
    x = np.zeros(len(theta),)

    with open('data/train_data.csv', 'w', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=['X_1', 'X_2', 'Y'])
        csv_writer.writeheader()

        for _ in range(int(1e6)):            
            x1, x2 = np.random.normal(3, 4, 1)[0], np.random.normal(-1, 4, 1)[0]            
            x = np.array([1, x1, x2])
            y = h(theta, x)
            
            row = {
                'X_1': x1,
                'X_2': x2,
                'Y'  : y
            }
            csv_writer.writerow(row)
    return

def cost(theta, xs, ys): 
    return (1/(2*len(xs))) * np.sum([np.power(y - h(theta, x), 2) for x, y in zip(xs, ys)])

def stochastic_gradient_descent(r, eta=0.001):
    theta = np.zeros(3,)
    xs = pd.read_csv('data/train_data.csv').values.ravel()
    ys = pd.read_csv('data/train_data.csv').values.ravel()
    
    index = 0
    for epoch in range(len(xs)/r):        
        last, x, y = index+r, [], []
        while index < min(last, len(xs)):
            x.append(xs[index])
            y.append(ys[index])
            index += 1
        x, y = np.array(x), np.array(y)

    return