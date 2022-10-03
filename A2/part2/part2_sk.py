import time
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def load_data(file_path, d=2, norm=True):
    objects = pd.read_pickle(file_path)
    pos_images, neg_images = [], []    
    for image, label in zip(objects['data'], objects['labels']):        
        if label == d%5 or label == (d+1)%5:
            image = image.flatten()
            if norm:
                image = image/255.0
            if label == d%5:
                pos_images.append(image/255)                    
            else:
                neg_images.append(image/255)
    pos_images = np.array(pos_images)
    neg_images = np.array(neg_images)
    pos_labels = np.array([1 for _ in range(len(pos_images))])
    neg_labels = np.array([-1 for _ in range(len(neg_images))])
    labels = np.concatenate((pos_labels, neg_labels))
    images = np.vstack((pos_images, neg_images))
    return images, np.reshape(labels, (len(labels), 1))

X, y = load_data("part2_data/part2_data/train_data.pickle")
X_test, y_test = load_data("part2_data/part2_data/test_data.pickle")

def linear_kernel():
    print('SVC Linear Kernel')
    svc = SVC(kernel='linear')
    clf = make_pipeline(StandardScaler(), svc)
    start_time = time.time()
    clf.fit(X, y.ravel())
    end_time = time.time()
    print('training time = ', end_time-start_time)

    def predict(x):
        return clf.predict(x)

    def accuracy(X, y):
        return clf.score(X, y)

    print('number of support vectors = ', svc.n_support_)
    #print('support vectors = ', svc.support_vectors_)    
    print('train_accuracy = ', accuracy(X, y))
    print('test_accuracy = ', accuracy(X_test, y_test))
    print('w = ', svc.coef_)
    print('b = ', svc.intercept_)


def gaussian_kernel():
    svc = SVC(kernel='rbf', gamma=0.001)
    clf = make_pipeline(StandardScaler(), svc)
    clf.fit(X, y.ravel())

    def predict(x):
        return clf.predict(x)

    def accuracy(X, y):
        return clf.score(X, y)

    print(svc.n_support_)
    print(svc.support_vectors_)
    print(accuracy(X, y))
    print(accuracy(X_test, y_test))

gaussian_kernel()