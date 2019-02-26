import numpy as np
if __name__ == "__main__":
    train_file = open('train.csv', 'r', encoding='big5')
    train_data = [ [] for i in range(18)]

    for i, line in enumerate(train_file.readlines()[1:]):
        #line = line.decode('big5')
        train_data[i%18] += [ float(data) if data.isnumeric() else 0 for data in line.split(',')[-24:] ]
    train_file.close()
    array_data = np.array(train_data)
    print(len(array_data.T))
    train_x, train_y = [], []
    for i in range(10, len(array_data.T)):
        