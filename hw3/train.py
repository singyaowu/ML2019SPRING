import numpy as np
import sys
import torch
import csv
import torch.nn as nn
from torch.optim import Adam
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import transforms as tf
import Model
   
# parameters
EPOCH = 50
BATCH_SIZE = 108
LEARNING_RATE = 0.001

def flipped_data(imgs, y):
    new_imgs = imgs.copy()
    new_y = y.copy()
    num_data = new_imgs.shape[0]
    for i in range(num_data):
        new_imgs[i,0,:,:] = new_imgs[i,0,:,::-1]
    return new_imgs, new_y

def parse_csv(label_path):
    raw_data_fp = open(label_path,'r')
    lines = raw_data_fp.readlines()[1:]
    num_data = len(lines)

    raw_imgs = np.empty(shape=(num_data,1,48*48), dtype=float)
    raw_y = np.zeros(shape=(num_data),dtype=np.int64)
    for i, line in enumerate(lines):
        nums = line.split(',')
        raw_y[i] = int(nums[0])
        raw_imgs[i,:,:] = np.array([float(num) for num in nums[1].split(' ')]) /255
    
    raw_imgs = raw_imgs.reshape((num_data,1,48,48))
    
    return raw_imgs, raw_y
class TrainDataset(Dataset):
    def __init__(self, raw_imgs, raw_y):
        aug_imgs, aug_y = flipped_data(raw_imgs, raw_y)
        imgs = raw_imgs#np.concatenate((raw_imgs, aug_imgs), axis=0)
        self.imgs = torch.tensor(imgs).type(torch.FloatTensor)
        #y = np.concatenate((raw_y, aug_y), axis=0)
        self.y = raw_y#torch.tensor(y).type(torch.LongTensor)
        self.transform = tf.Compose([
            tf.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25),
            tf.RandomHorizontalFlip(),
            tf.RandomRotation(5),            
            tf.RandomResizedCrop(48,scale=(0.95,1)),
            tf.ToTensor()
        ])
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):

        PIL_img = tf.ToPILImage()(self.imgs[idx])
        #return self.imgs[idx], self.y[idx]
        return self.transform(PIL_img).type(torch.FloatTensor), self.y[idx]

if __name__ == "__main__":
    raw_imgs, raw_y = parse_csv(sys.argv[1])
    
    num_val_data = 0#raw_imgs.size()[0] // 4
    val_imgs = raw_imgs[:num_val_data,:,:]
    val_y = raw_y[:num_val_data]

    print(raw_imgs.shape)
    training_set = TrainDataset(raw_imgs[num_val_data:,:,:,:], raw_y[num_val_data:])#Data.TensorDataset(train_imgs[num_val_data:,:,:], train_y[num_val_data:])
    val_set = Data.TensorDataset(
        torch.tensor(raw_imgs[num_val_data:,:,:,:]).type(torch.FloatTensor), 
        torch.tensor(raw_y[num_val_data:]).type(torch.LongTensor))
    train_loader = DataLoader(
        training_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    #print(training_set[1])
    #quit()
    # train
    device = torch.device('cuda')
    model = Model.MyCNN()
    model.load_state_dict(torch.load('model_params.pkl'))
    model.to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_func = nn.CrossEntropyLoss()

    print('start training...')
    model.train()


    for epoch in range(EPOCH):
        train_loss, train_acc = [], []
        torch.cuda.empty_cache()
        for step, (img, target) in enumerate(train_loader):
            #print(img.size(), target.size())           
            img_cuda = img.to(device, dtype=torch.float)
            target_cuda = target.to(device)

            optimizer.zero_grad()
            output = model(img_cuda)
            #print(output.size(), target_cuda.size())
            loss = loss_func(output, target_cuda)
            loss.backward()
            optimizer.step()

            predict = torch.max(output, 1)[1]
            acc = np.mean((target_cuda == predict).cpu().numpy())
            train_acc.append(acc)
            train_loss.append(loss.item())
        
        val_acc = 0
        if num_val_data > 0:
            model.eval()
            for step, (img, target) in enumerate(val_loader):
                img_cuda = img.to(device, dtype=torch.float)
                target_cuda = target.to(device)
                output = model(img_cuda)
                predict = torch.max(output, 1)[1]
                val_acc = np.sum((target_cuda == predict).cpu().numpy())
            val_acc /= val_set.__len__()
            model.train()

        print("Epoch: {}, Loss: {:.4f}, Acc: {:.4f}, Val Acc: {:.4f}"\
            .format(epoch + 1,
             np.mean(train_loss), np.mean(train_acc), val_acc
             ))
    
    model.eval()
    # save parameters
    # torch.save(model, 'model.pkl') # entire net
    torch.save(model.state_dict(), 'model_params.pkl') # parameters
    # validation
    # test
    output_file = open(sys.argv[3], 'w')
    output_file.write("id,label\n")
    
    test_imgs, ids = parse_csv(sys.argv[2])
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
  
    #predict = model(test_imgs)
    #predict_y = torch.max(predict, 1)[1]
    print(predict_y.size())
    #print(predict_y.size())
    #print(predict_y)
    for i in range(num_test_data):
        output_file.write( str(i) + ',' + str(int(predict_y[i])) + '\n')
    print('finish')
