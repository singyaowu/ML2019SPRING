import numpy as np
import sys

if __name__ == "__main__":

    # read training data
    raw_data = np.genfromtxt(sys.argv[1],encoding='big5', delimiter=',')
    data = raw_data[1:,3:]
    data[np.isnan(data)] = 0
    
    month_datas = []
    for month in range(12):
        a_month_data = data[(month)*18*20:(month+1)*18*20,:].tolist()
        sample = [ [] for i in range(18)]
        for i, row in enumerate(a_month_data):
            sample[i%18] += row
        month_datas.append( np.array(sample) )
    month_datas = np.array(month_datas)

    # preprocessing
    train_x, train_y = [], []
    
    for month_data in month_datas:
        # extract features
        print(month_data.shape)
        #month_data[0:4,:] = np.zeros(shape=month_data[0:4,:].shape)
        #month_data[10:17,:] = np.zeros(shape=month_data[10:17,:].shape)
        train_datas = month_data.T.reshape(-1)
        num_feature = 18
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
        if i % 100 == 0:
            loss = np.sqrt( np.sum((train_y - expect_y)**2) / len(train_x) )
            print('loss:', loss)
        gradient = np.zeros(len(train_x[0]) ).T
        gradient = -2.0 * train_x.T.dot(train_y - train_x.dot(w))
        sum_grad += gradient**2
        w -= lr * gradient / (np.sqrt(sum_grad + epsilon))
    print('final loss:', np.sqrt( np.sum((train_y - expect_y)**2) / len(train_x) ))
    np.savetxt('model.npy', w, delimiter=',')
    