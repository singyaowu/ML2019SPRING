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
        train_x.append(train_data[i: i+18*9].tolist() + [1.0])
        train_y.append(train_data[i + 18*9 + 9])
    train_x = np.array(train_x)
    train_y = np.array(train_y).T
    
    # training
    w = np.zeros(len(train_x[0]) ).T
    lr = 0.0000000003
    iteration = 1000000
    for i in range(iteration):#
        expect_y = train_x.dot(w)
        loss = np.sum((train_y - expect_y)**2) / len(train_x)
        print('loss:', loss)
        gradient = np.zeros(len(train_x[0]) ).T
        #for n in range(len(train_x)):
        #    gradient += -2 * (train_y[n] - train_x[n].dot(w)) * train_x[n].T
        gradient = -2 * train_x.T.dot(train_y - train_x.dot(w))
        print(gradient.shape)
        w -= lr * gradient
    print(w)