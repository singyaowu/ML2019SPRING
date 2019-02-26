import numpy as np
if __name__ == "__main__":    
    # read training data
    train_file = open('train.csv', 'r', encoding='big5')
    train_data = [ [] for i in range(18)]

    for i, line in enumerate(train_file.readlines()[1:]):
        train_data[i%18] += [ float(data) if data.isnumeric() else 0 for data in line.split(',')[-24:] ]
    train_file.close()

    # preprocessing
    train_data = np.array(train_data).T
    train_data = train_data.reshape(-1)
    
    train_x, train_y = [], []
    for i in range(0, len(train_data) - 180, 18):
        train_x.append([1.0] + train_data[i: i+18*9].tolist() )
        train_y.append(train_data[i + 18*9 + 9])
    train_x = np.array(train_x)
    train_y = np.array(train_y).T
    
    # training
    w = np.array([-10.0] * len(train_x[0])).T#np.zeros(len(train_x[0]) ).T
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