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


def gda(one): 
    m = len(X)
    phi = (1/m) * sum([1 if y == one else 0 for y in Y])
    mu_0, mu_1, sigma = np.zeros(2), np.zeros(2), np.zeros((2, 2)) 
    cnt = 0
    for x, y in zip(X, Y):                
        if y != one:
            cnt += 1
            mu_0 = mu_0 + x
    mu_0 = mu_0 / cnt
    cnt = 0
    for x, y in zip(X, Y):
        if y == one:
            cnt += 1
            mu_1 = mu_1 + x
    mu_1 = mu_1/cnt
    np.reshape(mu_0, (2,1))
    np.reshape(mu_1, (2,1))    
    for x_, y in zip(X, Y):
        x = np.reshape(x_, (2,1))        
        if y == one:
            sigma += np.matmul(x - mu_1, np.transpose(x - mu_1))            
        else:
            sigma += np.matmul(x - mu_0, np.transpose(x - mu_0))                    
    sigma *= (1/m)
    return phi, mu_0, mu_1, sigma

def gda_diff_cov(one):
    m = len(X)
    phi = (1/m) * sum([1 if y == one else 0 for y in Y])
    mu_0, mu_1, sigma0, sigma1 = np.zeros(2), np.zeros(2), np.zeros((2,2)), np.zeros((2,2)) 
    cnt = 0
    for x, y in zip(X, Y):                
        if y != one:
            cnt += 1
            mu_0 = mu_0 + x
    mu_0 = mu_0 / cnt
    cnt = 0
    for x, y in zip(X, Y):
        if y == one:
            cnt += 1
            mu_1 = mu_1 + x
    mu_1 = mu_1/cnt
    np.reshape(mu_0, (2,1))
    np.reshape(mu_1, (2,1))    
    cnt = 0
    for x_, y in zip(X, Y):
        x = np.reshape(x_, (2,1))        
        if y != one:
            sigma0 += np.matmul(x - mu_0, np.transpose(x - mu_0))                                       
            cnt += 1
    sigma0 *= (1/cnt)
    cnt = 0
    for x_, y in zip(X, Y):
        x = np.reshape(x_, (2,1))        
        if y == one:
            sigma1 += np.matmul(x - mu_1, np.transpose(x - mu_1))                                       
            cnt += 1
    sigma1 *= (1/cnt)
    return phi, mu_0, mu_1, sigma0, sigma1

def plot_cov_same():
    phi, mu_0, mu_1, sigma = gda('Alaska')
    print(phi, mu_0, mu_1, sigma)

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
    #plt.show()

    sigma_inv = np.linalg.inv(sigma)
    a, b, c, d = sigma_inv[0,0], sigma_inv[0,1], sigma_inv[1,0], sigma_inv[1,1]
    e, f = mu_0[0], mu_0[1]
    g, h = mu_1[0], mu_1[1]

    c1 = -2*e*a - b*f - c*f + b*h + 2*a*g + c*h
    c2 = -b*e - e*c - 2*d*f + b*g + c*g + 2*h*d
    c3 = a*g*g + g*h*b + c*g*h - a*e*e - b*e*f - c*e*f - d*f*f + d*h*h

    # x2 = (c3 - x1*c1)/c2
    x1 = np.linspace(-3,3, 100)
    x2 = (c3 - x1*c1)/c2
    plt.plot(x1, x2, color='black')
    plt.show()

def plot_cov_diff():
    phi, mu_0, mu_1, sigma0, sigma1 = gda_diff_cov('Alaska')
    print(phi, mu_0, mu_1, sigma0, sigma1)

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
    #plt.show()

    sigma0_inv = np.linalg.inv(sigma0)
    sigma1_inv = np.linalg.inv(sigma1)

    p, q = mu_0[0], mu_0[1]
    r, s = mu_1[0], mu_1[1]
    a, b, c, d = sigma0_inv[0,0], sigma0_inv[0,1], sigma0_inv[1,0], sigma0_inv[1,1]
    e, f, g, h = sigma1_inv[0,0], sigma1_inv[0,1], sigma1_inv[1,0], sigma1_inv[1,1]

    x1 = np.linspace(-3,3,100)
    A = d-h
    B = (b+c-g-f)*x1 + (-p*b - p*c - 2*d*q + r*f + r*g + 2*h*s)
    C = x1*x1*(a-e) + x1*(-2*a*p - b*q - c*q + 2*e*r + s*f + s*g) + (a*p*p + b*p*q + c*p*q + d*q*q - e*r*r - s*r*f - g*r*s - h*s*s)

    x21 = (-B + pow(B*B - 4*A*C, 0.5))/(2*A)
    #x22 = (-B - pow(B*B - 4*A*C, 0.5))/(2*A)

        
    plt.plot(x1, x21, color='black')
    #plt.plot(x1, x22, color='green')
    plt.show()

plot_cov_diff()