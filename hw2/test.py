import numpy as np
import sys
sigmoid = lambda s: (1.0 / (1 + np.exp(-s)))
# python3 test.py Xtest predict.csv
if __name__ == "__main__":  
    w = np.load('weight.npy')
    mean = np.load('mean.npy')
    std = np.load('std.npy')

    raw_x = np.genfromtxt(sys.argv[1], delimiter=',', dtype=np.float64)
    x = raw_x[1:,:]

    num_data, dim = x.shape
    
    for i in range(dim):
        x[:,i] = (x[:,i] - mean[i] )/(1 if std[i] == 0 else std[i])
    x = np.concatenate((np.ones(shape=(num_data,1)), x), axis=1).astype(np.float64)
    dim += 1
    expect_y = np.around(sigmoid(x.dot(w))).astype(int)

    # write file
    output_file = open(sys.argv[2], 'w')    
    output_file.write("id,label\n")
    
    for i in range(num_data):
        output_file.write( str(i+1) + ',' + str(expect_y[i][0]) + '\n')