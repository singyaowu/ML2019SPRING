import numpy as np

if __name__ == "__main__":  
    w = np.load('weight.npy')
    mean = np.load('mean.npy')
    std = np.load('std.npy')
    
    test_raw_data = np.genfromtxt('test.csv', delimiter=',')
    test_datas = test_raw_data[:,2:]
    test_datas[np.isnan(test_datas)] = 0.0

    num_data = test_datas.shape[0] // 18
    test_x = np.empty( shape=(num_data, 18 * 9),  dtype=float)
    for i in range(num_data):
        # feature extraction
        x = test_datas[i*18:(i+1)*18,:]
        for j in range(9):
            if x[9, j] < 0:
                if j == 8: x[9,j] = x[9, 7]
                elif j == 0: x[9,j] = max(0, x[9, 1])
                else: x[9,j] = max(0, (x[9, j+1] + x[9,j-1]) / 2)

        test_x[i,:] = x.reshape(1, -1)
    
    for i in range(test_x.shape[1]):
        test_x[:,i] = (test_x[:,i] - mean[i] )/(1 if std[i] == 0 else std[i])
    
    test_x = np.concatenate( (np.ones(shape=(num_data,1)),test_x), axis=1).astype(float)
    expect_y = test_x.dot(w)

    # write file
    output_file = open('submission.csv', 'w')    
    output_file.write("id,value\n")
    
    for i in range(num_data):
        output_file.write('id_' + str(i) + ',' + str(expect_y[i][0]) + '\n')
    
    print("output file is built!!")