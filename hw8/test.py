import numpy as np
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import Model
BATCH_SIZE = 256

def readfile(path):
    print("Reading File: %s..."%path)
    img_test = []

    raw_train = np.genfromtxt(path, delimiter=',', dtype=str, skip_header=1)
    for i in range(len(raw_train)):
        tmp = np.array(raw_train[i, 1].split(' ')).reshape(1, 48, 48)
        img_test.append(tmp)

    img_test = np.array(img_test, dtype=float) / 255.0
    img_test = torch.FloatTensor(img_test)

    return img_test

if __name__ == "__main__":
    model = Model.MyMobileCNN()
    model.load_state_dict(torch.load('mobile_model_params.pkl'))
    model.float()
    model.cuda()
    model.eval()
    # test
    output_file = open(sys.argv[2], 'w')
    output_file.write("id,label\n")
    
    test_imgs = readfile(sys.argv[1])
    num_test_data = test_imgs.size()[0]
    print('num_test_data=', num_test_data)
    test_set = Data.TensorDataset(test_imgs)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    predict_y = None
    for step, img in enumerate(test_loader):
        #print(img)
        img_cuda = img[0].cuda()
        output = model(img_cuda)
        predict = torch.max(output, 1)[1]
        if predict_y is None:
            predict_y = predict
        else:
            predict_y = torch.cat((predict_y, predict), 0)
  

    print(predict_y.size())

    for i in range(num_test_data):
        output_file.write( str(i) + ',' + str(int(predict_y[i])) + '\n')
    print('finish')