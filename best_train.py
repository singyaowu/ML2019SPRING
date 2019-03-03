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
        month_data = month_data[4:18,:]
        ind_pm25 = 5
        month_data = np.concatenate((month_data,month_data[np.newaxis:8]**2, month_data[np.newaxis:9]**2), axis=0)
        #month_data[0:4,:] = np.zeros(shape=month_data[0:4,:].shape)
        #month_data[10:17,:] = np.zeros(shape=month_data[10:17,:].shape)
        train_datas = month_data.T.reshape(-1)
        dim_day = month_data.shape[0]
        for i in range(0, len(train_datas) - dim_day * 10, dim_day):
            x = train_datas[i: i+dim_day*9].tolist()
            y = train_datas[i + dim_day*9 + ind_pm25]
            if y > 0:
                train_x.append(x)
                train_y.append(train_datas[i + dim_day*9 + ind_pm25])
    
    train_x = np.array(train_x)
    train_y = np.array(train_y).T
    
    dim = len(train_x[0])
    num_data = len(train_x)
    
    verify_x = np.copy(train_x)
    verify_x = np.concatenate((np.ones(shape=(num_data,1)), train_x), axis=1)
    # normalization
    
    mean = np.mean(train_x,axis=0)
    std = np.std(train_x, axis=0)
    
    for i in range(dim):
        train_x[:,i] = (train_x[:,i] - mean[i] )/(1 if std[i] == 0 else std[i])
    
    # add bias
    dim += 1
    train_x = np.concatenate((np.ones(shape=(num_data,1)), train_x), axis=1)
    
    # training
    w = np.array([-2.0] * dim).T#np.zeros(len(train_x[0]) ).T
    b = 0
    lr = 200.0
    iteration = 10000
    sum_grad = np.zeros(dim).T
    epsilon = np.array([1e-8] * dim).T
    for i in range(iteration):
        if i % 500 == 0:
            loss = np.sqrt( np.sum((train_y - train_x.dot(w))**2) / num_data )
            print('loss:', loss)
        gradient = np.zeros(dim).T
        gradient = -2.0 * train_x.T.dot(train_y - train_x.dot(w))
        sum_grad += gradient**2
        w -= lr * gradient / (np.sqrt(sum_grad + epsilon))
    
    # put std, mean back
    
    w[1:] = w[1:] / std
    w[0] -= mean.dot(w[1:])
    
    print('final loss:', np.sqrt( np.sum((train_y - verify_x.dot(w))**2) / num_data ))
    np.save('weight.npy', w)
    