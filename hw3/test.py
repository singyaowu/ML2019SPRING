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
def parse_csv(label_path):
    raw_data_fp = open(label_path,'r')
    lines = raw_data_fp.readlines()[1:]
    num_data = len(lines)
    raw_imgs = np.empty(shape=(num_data,1,48*48), dtype=float)
    raw_y = np.zeros(shape=(num_data),dtype=np.int64)
    #raw_y = np.zeros(shape=(num_data,7),dtype=np.int64)
    for i, line in enumerate(lines):
        nums = line.split(',')
        raw_y[i] = int(nums[0])
        #raw_y[i][int(nums[0])] = 1
        raw_imgs[i,:,:] = np.array([float(num) for num in nums[1].split(' ')]) /255
    
    raw_imgs = raw_imgs.reshape((num_data,1,48,48))
    
    #raw_y = raw_y.reshape((num_data,1))
    return raw_imgs, raw_y

if __name__ == "__main__":
    model = Model.MyCNN()
    model.load_state_dict(torch.load('model_params.pkl'))
    device = torch.device('cuda')
    model.to(device)
    model.eval()
    # test
    output_file = open(sys.argv[2], 'w')
    output_file.write("id,label\n")
    
    test_imgs, ids = parse_csv(sys.argv[1])
    test_imgs = torch.tensor(test_imgs).type(torch.FloatTensor)
    num_test_data = test_imgs.size()[0]
    print('num_test_data=', num_test_data)
    test_set = Data.TensorDataset(test_imgs)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    predict_y = None
    for step, (img) in enumerate(test_loader):
        img_cuda = img[0].to(device, dtype=torch.float)
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