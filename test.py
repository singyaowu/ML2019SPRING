import numpy as np

if __name__ == "__main__":  
    w = np.load('weight.npy')
    test_raw_data = np.genfromtxt('test.csv', delimiter=',')
    test_datas = test_raw_data[:,2:]
    test_datas[np.isnan(test_datas)] = 0.0

    test_file = open('test.csv', 'r')
    output_file = open('submission.csv', 'w')

    output_file.write("id,value\n")

    for id, i in enumerate(range(0, len(test_datas), 18)):
        test_x = test_datas[i: i+18]
        test_x = test_x[4:18,:]
        test_x = np.concatenate((test_x,test_x[np.newaxis:8]**2, test_x[np.newaxis:9]**2), axis=0)
    
        test_x = np.array(test_x).T.reshape(-1).tolist()
        test_x = np.array([1] + test_x)
        expect_y = test_x.dot(w)
        output_file.write('id_' + str(id) + ',' + str(expect_y) + '\n')
    print('submission.csv is build')