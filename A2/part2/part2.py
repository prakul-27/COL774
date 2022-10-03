import time
import numpy as np
import pandas as pd
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import matplotlib.pyplot as plt
from scipy.spatial import distance

def load_data(file_path, d=2, norm=True):
    objects = pd.read_pickle(file_path)
    pos_images, neg_images = [], []    
    for image, label in zip(objects['data'], objects['labels']):        
        if label == d%5 or label == (d+1)%5:
            image = image.flatten()
            if norm:
                image = image/255.0
            if label == d%5:                
                pos_images.append(image)                    
            else:
                neg_images.append(image)
    pos_images = np.array(pos_images)
    neg_images = np.array(neg_images)
    pos_labels = np.array([1 for _ in range(len(pos_images))])
    neg_labels = np.array([-1 for _ in range(len(neg_images))])
    labels = np.concatenate((pos_labels, neg_labels))
    images = np.vstack((pos_images, neg_images))
    return images, np.reshape(labels, (len(labels), 1))

def solve(images, labels, kernel=0):
    def get_cvxopt_solver_parameters(X, y, C=1):
        m, n = X.shape
        y = y.reshape(len(y),1) * 1.0        
        if kernel == 0:              
            X_dash = X*y
            H = np.dot(X_dash, X_dash.T) * 1.0
        else:            
            gamma = 0.001
            pairwise_distance = distance.cdist(X, X, 'sqeuclidean')                        
            H = np.exp(-gamma*pairwise_distance)*(y @ y.T)             
        P = cvxopt_matrix(H)
        q = cvxopt_matrix(-np.ones((m, 1)))
        G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
        h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
        A = cvxopt_matrix(y.T)
        b = cvxopt_matrix(np.zeros(1))
        return P, q, G, h, A, b
    P, q, G, h, A, b = get_cvxopt_solver_parameters(images, labels)
    solution = cvxopt_solvers.qp(P, q, G, h, A, b)    
    return solution

def linear_kernel():
    X, y = load_data("part2_data/part2_data/train_data.pickle",norm=True)
    start_time = time.time()
    solution = solve(X, y, kernel=0)
    end_time = time.time()
    print('training time = ', end_time-start_time)
    alphas = np.ravel(solution['x'])

    threshold = 1e-3
    n_samples, n_features = X.shape
    w = np.zeros((n_features, 1))
    support_vectors = 0

    for i in range(n_samples):
        w += (alphas[i] * y[i]) * np.reshape(X[i], (n_features, 1))
        if alphas[i] > threshold:
            support_vectors += 1

    b =  0.0
    maxterm, minterm  = float('-inf'), float('inf') 
    for i in range(n_samples):               
        if y[i] == -1:
            maxterm = max(maxterm, np.dot(w.T, X[i]))
        else:
            minterm = min(minterm, np.dot(w.T, X[i]))
    b = -1/2 * (maxterm + minterm)
    print(b)
    print(support_vectors, support_vectors/n_samples * 100)

    def predict(x, w, b):
        return 1 if np.dot(w.T, x) + b > 0 else -1

    def accuracy(X, y, w, b):
        n_samples, n_features = X.shape
        correct = 0
        for i in range(n_samples):
            if y[i] == predict(X[i], w, b):
                correct += 1
        return (correct / n_samples) * 100.0    

    def get_images(X, w, alpha, b, k=5): 
        n_samples, n_features = X.shape
        top_k_alpha = sorted(alpha)[-k:]
        top_k_images = []
        for i in range(n_samples):
            if alpha[i] in top_k_alpha:
                top_k_images.append(np.reshape(X[i], (32, 32, 3)))                
        return top_k_images, np.reshape(w, (32, 32, 3))     
    
    #top_k_images, w_reshaped = get_images(X, w, alphas, b)
    #print(top_k_images)
    #print(w_reshaped)

    #for image in top_k_images:
    #    plt.imshow(image)
    #    plt.show()

    #plt.imshow(w_reshaped)
    #plt.show()
    
    X_test, y_test = load_data("part2_data/part2_data/test_data.pickle")
    print(accuracy(X, y, w, b))
    print(accuracy(X_test, y_test, w, b))

def gaussian_kernel():
    def solve(images, labels, kernel=0, gamma=0.001):
        def get_cvxopt_solver_parameters(X, y, C=1):
            m, n = X.shape
            y = y.reshape(len(y),1) * 1.0        
            if kernel == 0:              
                X_dash = X*y
                H = np.dot(X_dash, X_dash.T) * 1.0
            else:                            
                pairwise_distance = distance.cdist(X, X, 'sqeuclidean')                        
                H = np.exp(-gamma*pairwise_distance)*(y @ y.T)             
            P = cvxopt_matrix(H)
            q = cvxopt_matrix(-np.ones((m, 1)))
            G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
            h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
            A = cvxopt_matrix(y.T)
            b = cvxopt_matrix(np.zeros(1))
            return P, q, G, h, A, b, H
        P, q, G, h, A, b, H = get_cvxopt_solver_parameters(images, labels)
        solution = cvxopt_solvers.qp(P, q, G, h, A, b)    
        return solution, H

    X, y = load_data("part2_data/part2_data/train_data.pickle")      
    solution, K = solve(X, y, kernel=1)
    alphas = np.ravel(solution['x'])
    #np.savetxt("alphas.csv", alphas, delimiter=',')

    # Parameters
    threshold = 1e-3
    n_samples, n_features = X.shape    
    support_vectors = 0
    for i in range(n_samples):        
        if alphas[i] > threshold:
            support_vectors += 1

    #Gaussian Kernel    
    print(alphas)
    print(max(alphas))
    
    w_t_x = []
    b =  0.0
    maxterm, minterm  = float('-inf'), float('inf') 
    for i in range(n_samples):       
        def compute(j):
            sum = 0.0
            for k in range(n_samples):
                sum += alphas[k]*y[k]*K[k][j]
            return sum
        w_t_x.append(compute(i))
        if y[i] == -1:
            maxterm = max(maxterm, w_t_x[-1])
        else:
            minterm = min(minterm, w_t_x[-1])
    b = -1/2 * (maxterm + minterm)
    print(b)
    print(support_vectors, support_vectors/n_samples * 100)

    def predict(x, b):
        return 1 if x+b > 0 else -1
    def accuracy(y, b):#train accuracy
        correct = 0
        for i in range(n_samples):
            if y[i] == predict(w_t_x[i], b):
                correct += 1
        return correct / n_samples
    
    #print('train accuracy = ', accuracy(y, b))
    X_test, y_test = load_data("part2_data/part2_data/test_data.pickle")    
    n_samples, n_features = X_test.shape
    gamma = 0.001
    pairwise_distance = distance.cdist(X, X_test, 'sqeuclidean')                        
    K_test = np.exp(-gamma*pairwise_distance)*(y@ y_test.T)    

    w_t_x = []
    for i in range(n_samples):
        def compute(j):
            sum = 0.0
            for k in range(n_samples):
                sum += alphas[k]*y[k]*K_test[k][j]
            return sum
        w_t_x.append(compute(i))    

    correct = 0
    for i in range(n_samples):        
        if y_test[i] == predict(w_t_x[i], b):            
            correct += 1
    print('test accuracy = ', correct/n_samples)

    def get_images(X, w, alpha, b, k=5): 
        n_samples, n_features = X.shape
        top_k_alpha = sorted(alpha)[-k:]
        top_k_images = []
        for i in range(n_samples):
            if alpha[i] in top_k_alpha:
                top_k_images.append(np.reshape(X[i], (32, 32, 3)))                
        return top_k_images, np.reshape(w, (32, 32, 3))              

    #top_k_images, w_reshaped = get_images(X, w, alphas, b)
    #print(top_k_images)
    #print(w_reshaped)

    #for image in top_k_images:
    #    plt.imshow(image)
    #    plt.show()

    #plt.imshow(w_reshaped)
    #plt.show()

gaussian_kernel()