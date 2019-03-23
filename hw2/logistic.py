import numpy as np
import sys
sigmoid = lambda s: (1.0 / (1 + np.exp(-s)))

if __name__ == "__main__":
    # read training data
    raw_x = np.genfromtxt(sys.argv[1], delimiter=',', dtype=np.float64)
    raw_y = np.genfromtxt(sys.argv[2], delimiter=',', dtype=np.float64)
    
    x = raw_x[1:,:]
    y = raw_y[1:,np.newaxis]

    print('read input finish')
    num_data, dim = x.shape
    #normalization
    mean = np.mean(x,axis=0)
    std = np.std(x, axis=0)
    for i in range(dim):
        x[:,i] = (x[:,i] - mean[i] )/(1 if std[i] == 0 else std[i])
    x = np.concatenate((np.ones(shape=(num_data,1)), x), axis=1).astype(np.float64)
    dim += 1
    
    # training
    w = 0.1 * np.ones(shape= (dim, 1), dtype=np.float64)
    lr = 0.1
    iteration = 3000
    sum_grad = np.zeros(shape=(dim,1),dtype=np.float64)
    for i in range(iteration):
        f_x = sigmoid(x.dot(w))
        if i % 1000 == 0:            
            loss = -np.sum(y * (np.log(f_x + 1e-8)) + (1.0 - y) * np.log(1.0 - f_x + 1e-8))
            y_hat = np.around(f_x +1e-8).reshape(-1).astype(int)
            y_test = y.reshape(-1).astype(int)
            accuracy = np.sum([ y1 == y2 for y1, y2 in zip(y_test, y_hat)]) / num_data
            print('iteration:',i,'loss:', loss, 'accuracy:', accuracy)
        
        gradient = np.zeros(shape=(dim, 1), dtype=np.float64)
        gradient = -x.T.dot(y - f_x)
        sum_grad += gradient**2
        
        w -= lr * gradient / (np.sqrt(sum_grad + 1e-8))

    np.save('weight.npy', w)
    np.save('mean.npy', mean)
    np.save('std.npy', std)