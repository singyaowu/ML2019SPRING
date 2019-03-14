import numpy as np
from scipy.stats import multivariate_normal
import sys
import math
sigmoid = lambda s: (1.0 / (1 + np.exp(-s)))
#def sigmoid(s):

#bash ./hw2_logistic.sh train.csv test.csv X_train Y_train X_test prediction.csv
if __name__ == "__main__":
    # read training data
    raw_x = np.genfromtxt(sys.argv[1], delimiter=',', dtype=np.float64)
    raw_y = np.genfromtxt(sys.argv[2], delimiter=',', dtype=int)
    x = raw_x[1:,:]
    y = raw_y[1:,np.newaxis]
    

    num_data, dim = x.shape
    print('read input finish')
    
    #normalization
    mean = np.mean(x,axis=0)
    std = np.std(x, axis=0)
        
    for i in range(dim):
        x[:,i] = (x[:,i] - mean[i] )/(1 if std[i] == 0 else std[i])
    
    #x = np.concatenate((np.ones(shape=(num_data,1)), x), axis=1).astype(np.float64)
    #dim += 1

    # classify 1, 0
    x_true, x_false = [], []
    for i in range(num_data):
        if y[i][0] == 1:
            x_true.append(x[i])
        else: # y[i][0] == 0
            x_false.append(x[i])

    x_true = np.array(x_true).T
    x_false = np.array(x_false).T

    num_true = x_true.shape[1]
    num_false = x_false.shape[1]
    print('# of false:', x_false.shape, '# of true:', x_true.shape)
    
    mean_true = np.mean(x_true, axis=1)
    mean_false = np.mean(x_false, axis=1)
    print(mean_false.shape, mean_true.shape)
    print(mean_true)
    cov_true = np.cov(x_true)
    cov_false = np.cov(x_false)

    cov = (cov_true*num_true + cov_false*num_false) / (num_true+num_false)
    print(cov.shape)
    print('is covariance LI: ', np.linalg.matrix_rank(cov) == cov.shape[0])
    np.save('mean_true.npy', mean_true)
    np.save('mean_false.npy', mean_false)
    np.save('cov.npy', cov)
    np.save('mean_gen.npy', mean)
    np.save('std_gen.npy', std)

    y_test = multivariate_normal.pdf(x_true.T, mean=mean_true, cov=cov, allow_singular=True)
    #print(y_test.reshape(-1).tolist())
    