import numpy as np
import pandas as pd

def read_data():
    xs = pd.read_csv('data/linearX.csv').values
    ys = pd.read_csv('data/linearY.csv').values
    return xs.ravel(), ys.ravel()

def normalize_data(xs, ys):
    mu, sigma = np.mean(xs), np.std(xs)
    norm_x = np.array([(x - mu)/sigma for x in xs])
    norm_y = np.array([(y - mu)/sigma for y in ys])
    return norm_x, norm_y

def h(x, theta, bias): # hypothesis
    return theta*x + bias

def cost(xs, ys, theta, bias): # J(theta)
    return (1/(2*len(xs))) * np.sum([np.power(y - h(x, theta, bias), 2) for x, y in zip(xs, ys)])
    
def batch_grad_descent(xs, ys, theta, bias, eta = 0.01, epsilon = 0.01, iters = 100, which=0):
    if which == 0:
        n = 0
        while True:
            n += 1
            new_theta = theta + eta * np.sum([(y - h(x, theta, bias)) * x for x, y in zip(xs, ys)])         
            if abs(new_theta - theta) < epsilon:
                print(abs(new_theta-theta))
                break
            theta = new_theta
            if n % 5 == 0:
                print('theta = ')
                print(theta, end=',')
                print('cost = ')
                print(cost(xs, ys, theta, bias))
    else:
        for i in range(iters):
            theta = theta + eta * np.sum([(y - h(x, theta, bias)) * x for x, y in zip(xs, ys)])
            if i % 10 == 0:
                print('theta = '+str(theta))
                print('cost = '+str(cost(xs, ys, theta, bias)))
                print()
    return theta

if __name__ == '__main__':
    xs,ys = read_data()
    xs,ys = normalize_data(xs, ys)
    theta, bias = 0.0, 1.0
    theta = batch_grad_descent(xs, ys, theta, bias, iters = 1000, which=0, eta=0.005, epsilon=1/1e15)
    print('final theta = '+str(theta))
    print('final cost = '+str(cost(xs, ys, theta, bias)))