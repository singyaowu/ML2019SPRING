import numpy as np
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms as tf
import Model
from myDataset import TrainDataset
# parameters
EPOCH = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.001
Validation = True

def readfile(path):
    print("Reading File: %s..."%path)
    img_train = []
    img_label = []
    img_val = []
    val_label = []

    raw_train = np.genfromtxt(path, delimiter=',', dtype=str, skip_header=1)
    for i in range(len(raw_train)):
        tmp = np.array(raw_train[i, 1].split(' ')).reshape(1, 48, 48)
        if (i % 10 == 0):
            img_val.append(tmp)
            val_label.append(raw_train[i][0])
        else:
            img_train.append(tmp)
            img_train.append(np.flip(tmp, axis=2))    # simple example of data augmentation
            img_label.append(raw_train[i][0])
            img_label.append(raw_train[i][0])

    img_train = np.array(img_train, dtype=float) / 255.0
    img_val = np.array(img_val, dtype=float) / 255.0
    img_label = np.array(img_label, dtype=int)
    val_label = np.array(val_label, dtype=int)

    img_train = torch.FloatTensor(img_train)
    img_val = torch.FloatTensor(img_val)
    img_label = torch.LongTensor(img_label)
    val_label = torch.LongTensor(val_label)

    return img_train, img_label, img_val, val_label

'''def parse_csv(label_path):
    raw_data_fp = open(label_path,'r')
    lines = raw_data_fp.readlines()[1:]
    num_data = len(lines)

    raw_imgs = np.empty(shape=(num_data,1,48*48), dtype=float)
    raw_y = np.zeros(shape=(num_data),dtype=np.int64)
    for i, line in enumerate(lines):
        nums = line.split(',')
        raw_y[i] = int(nums[0])
        raw_imgs[i,:,:] = np.array([float(num) for num in nums[1].split(' ')]) /255.0
    
    raw_imgs = raw_imgs.reshape((num_data,1,48,48))
    
    return raw_imgs, raw_y
'''
if __name__ == "__main__":
    img_train, img_label, img_val, val_label = readfile(sys.argv[1])

    training_set = TrainDataset(img_train, img_label)
    val_set = Data.TensorDataset(img_val, val_label)
    train_loader = DataLoader(
        training_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    
    # train
    model = Model.MyMobileCNN()
    teacher_model = Model.MyCNN()
    teacher_model.load_state_dict(torch.load('model_params0.6896.pkl'))

    model.cuda()
    model.train()
    teacher_model.cuda()
    teacher_model.eval()

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_func = nn.CrossEntropyLoss()

    print('start training...')

    high_val_acc = 0.62
    for epoch in range(EPOCH):
        train_loss, train_acc = [], []
        torch.cuda.empty_cache()
        for step, (img, target) in enumerate(train_loader):
            #print(img.size(), target.size())           
            img_cuda = img.cuda()
            target_cuda = target.cuda()

            optimizer.zero_grad()
            output = model(img_cuda)
            #teacher_output = teacher_model(img_cuda)
            #print(output.size(), target_cuda.size())
            loss = loss_func(output, target_cuda)
            loss.backward()
            optimizer.step()

            predict = torch.max(output, 1)[1]
            acc = np.mean((target_cuda == predict).cpu().numpy())
            train_acc.append(acc)
            train_loss.append(loss.item())
        acc = np.mean(train_acc)
        val_acc = 0
        
        if Validation:
            model.eval()
            for _, (img, target) in enumerate(val_loader):
                img_cuda = img.cuda()
                target_cuda = target.cuda()
                output = model(img_cuda)
                predict = torch.max(output, 1)[1]
                val_acc += np.sum((target_cuda == predict).cpu().numpy())
            val_acc /= val_set.__len__()
            if val_acc > high_val_acc:
                high_val_acc = val_acc
                torch.save(model.state_dict(), 'mobile_model_params.pkl')
                print('saved new parameters')
            model.train()
        if epoch % 10 == 0:
            torch.save(model.state_dict(), 'mobile_model_params.pkl')
            print('saved new parameters')
        print("Epoch: {}| Loss: {:.4f}| Acc: {:.4f}| Val Acc: {:.4f}"\
            .format(epoch + 1, np.mean(train_loss), acc, val_acc))
    
    model.eval()
    # save parameters
    # torch.save(model, 'model.pkl') # entire net
    torch.save(model.state_dict(), 'mobile_model_params.pkl') # parameters
    

