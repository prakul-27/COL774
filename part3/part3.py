import pandas as pd
import numpy as np

def g(z):
    return 1/(1 + np.power(np.e, -z))

def h(theta, X):    
    return g(np.matmul(X, theta))

def cost(theta, X, Y):
    return np.matmul(np.transpose(Y), np.log(h(theta, X))) + np.matmul(np.transpose(1-Y), np.log(1-h(theta, X)))

def newton(epsilon=0.1):
    df = pd.read_csv('data\logisticX.csv')    
    X = np.array([[1.0, x1, x2] for x1, x2 in df.values])
    Y = pd.read_csv('data\logisticY.csv').values    
    
    #data normalize
    for i in range(1, 3):
        X[:,i] = (X[:,i] - np.mean(X[:,i]))/np.std(X[:,i])
        
    theta = np.zeros((3,1))
    def H_inv(theta):
        D = np.zeros((len(X), len(X)))
        for i in range(len(X)):
            z = np.matmul(np.transpose(theta), np.reshape(X[i], (3,1)))                          
            D[i][i] = g(z)*(1-g(z))        
        H = np.matmul(np.matmul(np.transpose(X), D), X)        
        return np.linalg.inv(H)

    def grad(theta):
        return np.transpose(np.matmul(np.transpose(Y - h(theta, X)), X))

    while True:        
        new_theta = theta + np.matmul(H_inv(theta), grad(theta))
        diff = np.sum([(old-new)**2 for old, new in zip(theta, new_theta)])
        print(diff)
        if diff < epsilon:
            break
        theta = new_theta        

    return theta

print(newton())