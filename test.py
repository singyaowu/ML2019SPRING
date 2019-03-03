import numpy as np

if __name__ == "__main__":  
    w = np.loadtxt('model.npy')
    test_file = open('test.csv', 'r')
    output_file = open('submission.csv', 'w')

    test_datas = test_file.readlines()
    output_file.write("id,value\n")

    for i in range(0, len(test_datas), 18):
        test_data = test_datas[i: i+18]
        test_x = []
        for line in test_data:
            features = []
            for data in line.split(',')[-9:]:
                feature = 0
                try: feature = float(data)
                except ValueError: pass
                features.append(feature)
            test_x.append(features)
    
        test_x = np.array(test_x).T.reshape(-1).tolist()
        test_x = np.array([1] + test_x)
        expect_y = test_x.dot(w)
        id = test_data[0].split(',')[0]
        output_file.write(id + ',' + str(expect_y) + '\n')