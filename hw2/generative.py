import numpy as np
from numpy.linalg import inv
import sys
import math
sigmoid = lambda s: (1.0 / (1 + np.exp(-s)))
#def sigmoid(s):

if __name__ == "__main__":
    # read training data
    raw_x = np.genfromtxt(sys.argv[1], delimiter=',', dtype=np.float64)
    raw_y = np.genfromtxt(sys.argv[2], delimiter=',', dtype=int)
    x = raw_x[1:,:]
    y = raw_y[1:,np.newaxis]
    
    num_data, dim = x.shape
    
    #normalization
    mean = np.mean(x,axis=0)
    std = np.std(x, axis=0)
        
    for i in range(dim):
        x[:,i] = (x[:,i] - mean[i] )/(1 if std[i] == 0 else std[i])
    
    # classify 1, 0
    x_c1, x_c0 = [], []
    for i in range(num_data):
        if y[i][0] == 1:
            x_c1.append(x[i])
        else: # y[i][0] == 0
            x_c0.append(x[i])

    x_c1 = np.array(x_c1)
    x_c0 = np.array(x_c0)

    num_c1 = x_c1.shape[0]
    num_c0 = x_c0.shape[0]
    
    mean_c1 = np.mean(x_c1, axis=0)
    mean_c0 = np.mean(x_c0, axis=0)
    
    n = x_c1.shape[1]
    cov_0 = np.zeros((n,n))
    cov_1 = np.zeros((n,n))
        
    for i in range(num_c0):
        cov_0 += np.dot(np.transpose([x_c0[i] - mean_c0]), [(x_c0[i] - mean_c0)]) / num_c0

    for i in range(num_c1):
        cov_1 += np.dot(np.transpose([x_c1[i] - mean_c1]), [(x_c1[i] - mean_c1)]) / num_c1

    cov = (cov_1*num_c1 + cov_0*num_c0) / (num_c1+num_c0)
    w_no_b = np.transpose((mean_c1-mean_c0).dot(inv(cov)))
    b = -0.5 * mean_c1.dot(inv(cov)).dot(mean_c1) \
        + 0.5 * mean_c0.dot(inv(cov)).dot(mean_c0) \
            +np.log(float(num_c1) / num_c0)
    w = np.array([b] + w_no_b.tolist()).reshape(-1,1)
    np.save('mean_gen.npy', mean)
    np.save('std_gen.npy', std)
    np.save('weight_gen.npy', w)
    