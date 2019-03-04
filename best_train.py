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


    raw_x = np.empty(shape=(12*471 , 18*9), dtype=float)
    raw_y = np.empty(shape=(12*471 , 1), dtype=float)
    
    for month, month_data in enumerate(month_datas):
        for hour in range(0, 20*24 - 9):
            raw_x[month * 471 + hour, :] = month_data[:, hour:hour+9].reshape(1, -1)
            raw_y[month*471 + hour, 0] = month_data[9, hour+9]
        dim_day = month_data.shape[0]
    
    # preprocessing
    neg_y = []
    for i in range(raw_y.shape[0]):
        if raw_y[i][0] < 0: 
            neg_y.append(i)
    raw_x = np.delete(raw_x, neg_y, axis=0)
    raw_y = np.delete(raw_y, neg_y, axis=0)
    del neg_y
    
    train_x, train_y = [], []

    for x, y in zip(raw_x, raw_y):
   
        extract_x = x.copy().reshape(18,9)
        
        for i in range(9):
            if extract_x[9,i] < 0:
                break
        else:
            # pick NO2(5), 03(7),PM10(8), PM2.5(9), SO2(12)
            #extract_x = x.copy().reshape(18,9)[0:18, :]
            #extract_x = np.concatenate((extract_x,extract_x[np.newaxis:8]**2, extract_x[np.newaxis:9]**2), axis=0)
            #extract_x[0:2, :] = np.zeros(shape=extract_x[0:2, :].shape)
            #extract_x[3:5, :] = np.zeros(shape=extract_x[3:5, :].shape)
            #extract_x[10:14, :] = np.zeros(shape=extract_x[10:14, :].shape)
            #extract_x[15:, :] = np.zeros(shape=extract_x[15:, :].shape)
            
            train_x.append(extract_x.reshape(-1))
            train_y.append(y)

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    dim = len(train_x[0])
    num_data = len(train_x)
    
    # normalization
    
    mean = np.mean(train_x,axis=0)
    std = np.std(train_x, axis=0)
    
    for i in range(dim):
        train_x[:,i] = (train_x[:,i] - mean[i] )/(1 if std[i] == 0 else std[i])
    
    # add bias
    dim += 1
    train_x = np.concatenate((np.ones(shape=(num_data,1)), train_x), axis=1)
    
    # training
    w = np.zeros(shape= (dim, 1))
    lr = 200.0
    iteration = 10000
    sum_grad = np.zeros(shape=(dim,1))
    for i in range(iteration):
        if i % 500 == 0:
            loss = np.sqrt( np.sum((train_y - train_x.dot(w))**2) / num_data )
            print('loss:', loss)
        gradient = np.zeros(dim).T
        gradient = -2.0 * train_x.T.dot(train_y - train_x.dot(w))
        sum_grad += gradient**2
        w -= lr * gradient / (np.sqrt(sum_grad + 1e-8))
    
    # put std, mean back
    
    #w[1:] = w[1:] / std
    #w[0] -= mean.dot(w[1:])
    
    for i in range(raw_x.shape[1]):
        raw_x[:,i] = (raw_x[:,i] - mean[i] )/(1 if std[i] == 0 else std[i])
    raw_x = np.concatenate((np.ones(shape=(raw_x.shape[0],1)), raw_x), axis=1)
    #print('raw loss:', np.sqrt( np.sum((raw_y - raw_x.dot(w))**2) / raw_x.shape[0] ))
    print('train_loss',np.sqrt( np.sum((train_y - train_x.dot(w))**2) / num_data ))
    np.save('weight.npy', w)
    np.save('mean.npy', mean)
    np.save('std.npy', std)
    