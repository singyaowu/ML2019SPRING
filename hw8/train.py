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

# parameters
EPOCH = 1
BATCH_SIZE = 256
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
        raw_imgs[i,:,:] = np.array([float(num) for num in nums[1].split(' ')]) /255.0
    
    raw_imgs = raw_imgs.reshape((num_data,1,48,48))
    
    return raw_imgs, raw_y

class TrainDataset(Dataset):
    def __init__(self, raw_imgs, raw_y):
        aug_imgs, aug_y = flipped_data(raw_imgs, raw_y)
        imgs = np.concatenate((raw_imgs, aug_imgs), axis=0)
        self.imgs = torch.tensor(imgs).type(torch.FloatTensor)
        y = np.concatenate((raw_y, aug_y), axis=0)
        self.y = torch.tensor(y).type(torch.LongTensor)
        self.transform = tf.Compose([
            tf.ToPILImage(),
            tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            tf.RandomRotation(30),            
            tf.RandomResizedCrop(48,scale=(0.8,1)),
            tf.ToTensor()
        ])
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        #return self.imgs[idx], self.y[idx]
        return self.transform(self.imgs[idx]).type(torch.FloatTensor), self.y[idx]

if __name__ == "__main__":
    raw_imgs, raw_y = parse_csv(sys.argv[1])
    imgs_shape = raw_imgs.shape
    y_shape = raw_y.shape
    c =  np.concatenate((raw_imgs.reshape(len(raw_imgs), -1), raw_y.reshape(len(raw_y),1)), axis=1)
    np.random.shuffle(c)
    raw_imgs = (c[:, :-1]).reshape(imgs_shape)
    raw_y = (c[:, -1]).reshape(y_shape)
    
    num_val_data = 0#raw_imgs.shape[0] // 12
    val_imgs = raw_imgs[:num_val_data,:,:]
    val_y = raw_y[:num_val_data]

    train_imgs = raw_imgs[num_val_data:,:,:,:]
    train_y = raw_y[num_val_data:]
    a = train_imgs.shape[0]

    training_set = TrainDataset(train_imgs, train_y)
    val_set = Data.TensorDataset(
        torch.tensor(val_imgs).type(torch.FloatTensor), 
        torch.tensor(val_y).type(torch.LongTensor))
    train_loader = DataLoader(
        training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    
    # train
    device = torch.device('cuda')
    model = Model.MyCNN()
    try:
        model.load_state_dict(torch.load('model_params.pkl'))
        print('use exist parameters')
    except:
        print('new model, no exist parameters')
        pass
    model.to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_func = nn.CrossEntropyLoss()

    print('start training...')
    model.train()

    high_val_acc = 0.67
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
        acc = np.mean(train_acc)
        val_acc = 0
        if num_val_data > 0:
            model.eval()
            for _, (img, target) in enumerate(val_loader):
                img_cuda = img.to(device, dtype=torch.float)
                target_cuda = target.to(device)
                output = model(img_cuda)
                predict = torch.max(output, 1)[1]
                val_acc += np.sum((target_cuda == predict).cpu().numpy())
            val_acc /= val_set.__len__()
            if val_acc > high_val_acc:
                high_val_acc = val_acc
                torch.save(model.state_dict(), 'model_params.pkl')
                print('saved new parameters')
            model.train()
        if epoch % 10 == 0:
            torch.save(model.state_dict(), 'model_params.pkl')
            print('saved new parameters')
        print("Epoch: {}| Loss: {:.4f}| Acc: {:.4f}| Val Acc: {:.4f}"\
            .format(epoch + 1, np.mean(train_loss), acc, val_acc))
    
    model.eval()
    # save parameters
    # torch.save(model, 'model.pkl') # entire net
    torch.save(model.state_dict(), 'model_params.pkl') # parameters
    

