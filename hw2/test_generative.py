import numpy as np
from scipy.stats import multivariate_normal
import sys
sigmoid = lambda s: (1.0 / (1 + np.exp(-s)))

if __name__ == "__main__":  
    mean_true = np.load('mean_true.npy')
    mean_false = np.load('mean_false.npy')
    mean = np.load('mean_gen.npy')
    std = np.load('std_gen.npy')
    cov = np.load('cov.npy')
    raw_x = np.genfromtxt(sys.argv[1], delimiter=',', dtype=np.float64)
    x = raw_x[1:,:]

    num_data, dim = x.shape

    print('read input finish')
    for i in range(dim):
        x[:,i] = (x[:,i] - mean[i] )/(1 if std[i] == 0 else std[i])
    #gaussian = multivariate_normal(mean=mean_true, cov=cov)
    y = multivariate_normal.pdf(x, mean=mean_false, cov=cov, allow_singular=True)
    
    #print(y)
    expect_y = y#np.around(y).astype(int)
    print(expect_y.shape)
    # write file
    output_file = open(sys.argv[2], 'w')    
    output_file.write("id,label\n")
    
    for i in range(num_data):
        output_file.write( str(i+1) + ',' + str(expect_y[i]) + '\n')