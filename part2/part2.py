import csv
import numpy as np

theta = np.array([3, 1, 2])
x = np.zeros(len(theta),)

def h(theta, x):
    epsilon = np.random.normal(0, 2, 1)[0]
    return np.dot(theta, x) + epsilon

def sample_data():
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

sample_data()