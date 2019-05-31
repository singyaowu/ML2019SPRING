import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms as tf
from skimage import io
from PIL import Image

class TrainDataset(Dataset):
    def __init__(self, img_train, img_label):
        self.img_train = img_train
        self.img_label = img_label

        self.transform = tf.Compose([
            tf.ToPILImage(),
            tf.RandomHorizontalFlip(),
            tf.RandomAffine(30),
            #tf.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            tf.RandomRotation(30),            
            tf.RandomResizedCrop(48,scale=(0.8,1)),
            tf.ToTensor(),
            #tf.Normalize(mean=[0.5],std=[0.5]),
        ])
    def __len__(self):
        return len(self.img_train)

    def __getitem__(self, idx):
        #.type(torch.FloatTensor)
        return self.transform(self.img_train[idx]), self.img_label[idx]

class OldTrainDataset(Dataset):
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
