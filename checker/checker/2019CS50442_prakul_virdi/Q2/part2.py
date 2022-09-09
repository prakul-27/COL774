import csv
import numpy as np
import pandas as pd
import random
import pprint
random.seed()


def sample_data():
    theta = np.array([3, 1, 2])
    x = np.zeros(len(theta),)

    with open('data/train_data.csv', 'w', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=['X_1', 'X_2', 'Y'])
        csv_writer.writeheader()

        for _ in range(int(1e6)):            
            x1, x2 = np.random.normal(3, 4, 1)[0], np.random.normal(-1, 4, 1)[0]            
            x = np.array([1, x1, x2])
            epsilon = np.random.normal(0, 2, 1)[0]
            y = np.dot(theta, x) + epsilon
            
            row = {
                'X_1': x1,
                'X_2': x2,
                'Y'  : y
            }
            csv_writer.writerow(row)
    return

def cost(theta, x, y):
    return 1/(2*len(x))*sum((y-np.matmul(x, theta))**2)

def batches(x, r):
	batch_list = []
	current_batch = []
	for xi in x:
		current_batch.append(xi)				
		if len(current_batch) == r:
			batch_list.append(current_batch)
			current_batch = []		
	if current_batch:
		batch_list.append(current_batch)
	return batch_list

def stochastic_gradient_descent(r, eta=0.001, epsilon=0.001, conv_iters=1000):
    theta = np.zeros(3,)
    df = pd.read_csv('data/train_data.csv')
    x1, x2, Y = df['X_1'].ravel(), df['X_2'].ravel(), df['Y'].ravel()
    X = np.array([[1.0, x1i, x2i] for x1i, x2i in zip(x1, x2)])
    
    X = np.array(batches(X, r))
    Y = np.array(batches(Y, r))    
    
    total_iters = 0
    has_converged = False
    iters, cur_theta_sum, prev_avg_theta = 0, np.zeros(3,), np.zeros(3,)
        		
    
    with open('r100.csv', 'w', newline='') as csv_file:        
        csv_writer = csv.DictWriter(csv_file, fieldnames=['theta0', 'theta1', 'theta2', 'cost'])	    
        csv_writer.writeheader()       
        while not has_converged:                        
            for x, y in zip(X, Y):
                total_iters += 1
                iters += 1
                #update theta                         
                theta = theta + eta*(1/len(x))*np.matmul(np.transpose(y-np.matmul(x, theta)), x)            
                #print(theta)
                cur_theta_sum += theta                 
                csv_writer.writerow({"theta0":theta[0], "theta1":theta[1], "theta2":theta[2], "cost":cost(theta, x, y)})
                if iters == conv_iters:                
                    iters = 0                                
                    cur_avg_theta = cur_theta_sum
                    diff = pow(np.sum([(old-new)**2 for old, new in zip(prev_avg_theta, cur_avg_theta)]), 0.5)                 
                    #print(diff)
                    if diff < epsilon:
                        has_converged = True
                        break
                    prev_avg_theta = cur_avg_theta
                    cur_theta_sum = np.zeros(3,)                
        return theta, total_iters

# models
# r = 1, epsilon = 1, conv_iters=100, iters = 11700, conv_val = [2.90096727, 1.00398103, 1.9958781 ]
# r = 100, epsilon = 0.1, conv_iters=50, iters = 7250, conv_val = [2.97415026, 1.00340122, 1.99621081]
# r = 10000, epsilon = 0.001, conv_iters=10, iters = 8190, conv_val = [2.98299967, 1.00181044, 1.99969435]
# r = 1000000, epsilon = 0.00001, conv_iters=1, iters = 8560, conv_val = [2.98714405, 1.00147602, 1.99980244] 

#theta = np.array([2.98714405, 1.00147602, 1.99980244])
def test(theta):
    df = pd.read_csv('data/q2test.csv')
    x1, x2, Y = df['X_1'].ravel(), df['X_2'].ravel(), df['Y'].ravel()
    X = np.array([[1.0, x1i, x2i] for x1i, x2i in zip(x1, x2)])
    return cost(theta, X, Y)

# costs
# r = 1, cost = 0.9895715244767835 
# r = 100, cost = 0.9848810836329218 
# r = 10000, cost = 0.9832557604388013
# r = 1000000, cost = 0.9831320884069167
# original hypothesis, cost = 0.9829469215000091

def test_with_original():
    theta = np.array([3, 1, 2])    
    df = pd.read_csv('data/q2test.csv')
    x1, x2, Y = df['X_1'].ravel(), df['X_2'].ravel(), df['Y'].ravel()
    X = np.array([[1.0, x1i, x2i] for x1i, x2i in zip(x1, x2)])    
    Y = df['Y'].ravel()
    return cost(theta, X, Y)

def run(test_path_x):    
    theta = [2.97415026, 1.00340122, 1.99621081]
    xs = pd.read_csv(test_path_x,header=None).values
    with open('result_2.txt', 'w') as file:
        for x in xs:            
            h = theta[0] + x[0]*theta[1] + x[1]*theta[2]
            file.write(str(h)+"\n")            