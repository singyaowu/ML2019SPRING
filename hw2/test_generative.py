import numpy as np
import sys
sigmoid = lambda s: (1.0 / (1 + np.exp(-s)))

if __name__ == "__main__":  
    mean_true = np.load('mean_true.npy')
    mean_false = np.load('mean_false.npy')
    cov = np.load('cov.npy')

    raw_x = np.genfromtxt(sys.argv[1], delimiter=',', dtype=np.float64)
    x = raw_x[1:,:]

    num_data, dim = x.shape
    
    print('read input finish')
    
    
    expect_y = np.around(sigmoid(x.dot(w))).astype(int)

    # write file
    output_file = open(sys.argv[2], 'w')    
    output_file.write("id,label\n")
    
    for i in range(num_data):
        output_file.write( str(i+1) + ',' + str(expect_y[i][0]) + '\n')