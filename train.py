import numpy as np
import sys

def strnumeric(data):
    feature = 0.0
    try: feature = float(data)
    except ValueError: pass
    return feature

if __name__ == "__main__":

    # read training data
    train_file = open('train.csv', 'r', encoding='big5')
    train_datas = [ [] for i in range(18)]

    for i, line in enumerate(train_file.readlines()[1:]):
        train_datas[i%18] += [ strnumeric(data) for data in line.split(',')[-24:] ]
    train_file.close()
    
    # preprocessing
    train_datas = np.array(train_datas).T # (numData, features)
    print(train_datas.shape)
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
    print(train_x.shape)
    
    
    # training
    w = np.array([-2.0] * len(train_x[0])).T#np.zeros(len(train_x[0]) ).T
    b = 0
    lr = 100.0
    iteration = 100000
    sum_grad = np.zeros(len(train_x[0]) ).T
    epsilon = np.array([1e-8] * len(train_x[0])).T
    for i in range(iteration):
        expect_y = train_x.dot(w)

        if i % 1000 == 0:
            loss = np.sqrt( np.sum((train_y - expect_y)**2) / len(train_x) )
            print('loss:', loss)
        gradient = np.zeros(len(train_x[0]) ).T
        gradient = -2.0 * train_x.T.dot(train_y - train_x.dot(w))
        sum_grad += gradient**2
        w -= lr * gradient / (np.sqrt(sum_grad + epsilon))
    print('final loss:', np.sqrt( np.sum((train_y - expect_y)**2) / len(train_x) ))
    np.savetxt('model.npy', w, delimiter=',')