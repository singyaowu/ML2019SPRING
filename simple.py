import numpy as np
if __name__ == "__main__":    
    # read training data
    train_file = open('train.csv', 'r', encoding='big5')
    train_datas = [ [] for i in range(18)]

    for i, line in enumerate(train_file.readlines()[1:]):
        features = []
        for data in line.split(',')[-24:]:
            feature = 0
            try: feature = float(data)
            except ValueError: pass
            features.append(feature)
        train_datas[i%18] += features
        #train_datas[i%18] += [ float(data) if data.isnumeric() else 0.0 for data in line.split(',')[-24:] ]
    train_file.close()

    # preprocessing
    train_datas = np.array(train_datas).T # (numData, features)
    '''
    for i in range(0, len(train_datas)-9):
        train_data = train_datas[i: i+9]
        test_x = []
        
        test_x = np.array(test_x).T.reshape(-1).tolist()
        test_x = np.array([1] + test_x)
        expect_y = test_x.dot(w)
        id = test_data[0].split(',')[0]
        output_file.write(id + ',' + str(expect_y) + '\n')
    '''
    train_datas = train_datas.reshape(-1)
    num_feature = 18
    train_x, train_y = [], []
    for i in range(0, len(train_datas) - num_feature * 10, num_feature):
        x = [1.0] + train_datas[i: i+num_feature*9].tolist()
        y = train_datas[i + 18*9 + 9]
        if y > 0:
            train_x.append(x)
            train_y.append(train_datas[i + 18*9 + 9])
    train_x = np.array(train_x)
    train_y = np.array(train_y).T
    
    # training
    w = np.array([-2.0] * len(train_x[0])).T#np.zeros(len(train_x[0]) ).T
    b = 0
    lr = 100.0
    iteration = 100000
    sum_grad = np.zeros(len(train_x[0]) ).T
    epsilon = np.array([1e-8] * len(train_x[0])).T
    for i in range(iteration):
        expect_y = train_x.dot(w)
        loss = np.sqrt( np.sum((train_y - expect_y)**2) / len(train_x) )
        print('loss:', loss)
        gradient = np.zeros(len(train_x[0]) ).T
        #for n in range(len(train_x)):
        #    gradient += -2 * (train_y[n] - train_x[n].dot(w)) * train_x[n].T
        gradient = -2.0 * train_x.T.dot(train_y - train_x.dot(w))
        sum_grad += gradient**2
        w -= lr * gradient / (np.sqrt(sum_grad + epsilon))
 
    np.savetxt('model.npy', w, delimiter=',')