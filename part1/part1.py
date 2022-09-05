import numpy as np
import pandas as pd
import csv

def read_data():
    xs = pd.read_csv('data/linearX.csv').values
    ys = pd.read_csv('data/linearY.csv').values
    return xs.ravel(), ys.ravel()

def normalize_data(xs, ys):
    mu, sigma = np.mean(xs), np.std(xs)
    norm_x = np.array([(x - mu)/sigma for x in xs])
    mu, sigma = np.mean(ys), np.std(ys)
    norm_y = np.array([(y - mu)/sigma for y in ys])
    return norm_x, norm_y

def h(x, theta, bias): # hypothesis
    return theta*x + bias

def cost(xs, ys, theta, bias): # J(theta)
    return (1/(2*len(xs))) * np.sum([np.power(y - h(x, theta, bias), 2) for x, y in zip(xs, ys)])
    
def batch_grad_descent(xs, ys, theta, bias, eta = 0.01, epsilon = 0.01, iters = 100):    
    n = 0
    costs, thetas, biases = [],[],[]    
    while True:
        costs.append(cost(xs, ys, theta, bias))
        thetas.append(theta)
        biases.append(bias)
        n += 1
        new_theta = theta + eta * np.sum([(y - h(x, theta, bias)) * x for x, y in zip(xs, ys)])
        new_bias = bias + eta * np.sum([(y - h(x, theta, bias)) for x, y in zip(xs, ys)])                
        if abs(new_theta - theta) < epsilon and abs(new_bias-bias) < epsilon:                
            break
        theta = new_theta
        bias = new_bias
        if n % 5 == 0:
            print('theta =', end=' ')
            print(theta, end=',')
            print('cost =', end= ' ')
            print(cost(xs, ys, theta, bias))    
    with open('train_data.csv', w) as csv_file:
        
    return theta, bias

if __name__ == '__main__':
    xs,ys = read_data()
    xs,ys = normalize_data(xs, ys)
    theta, bias = 0.0, 0.0
    theta, bias = batch_grad_descent(xs, ys, theta, bias, iters = 1000, which=0, eta=0.005, epsilon=1/1e15)
    print('final theta = '+str(theta))
    print('final_bias = '+str(bias))
    print('final cost = '+str(cost(xs, ys, theta, bias)))