import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms as tf
from skimage import io
from PIL import Image
import os
class TrainDataset(Dataset):
    def __init__(self, imgs_range, images_path):
        self.imgs_range = imgs_range
        self.num_imgs = imgs_range[1]-imgs_range[0] 
        self.images_path = images_path

        self.transform = tf.Compose([
            #tf.ToPILImage(),
            tf.RandomHorizontalFlip(),
            #tf.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            #tf.RandomRotation(5),            
            #tf.RandomResizedCrop(32,scale=(0.8,1)),
            tf.ToTensor(),
            tf.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]),
        ])
    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        idx_img = self.imgs_range[0] + idx
        assert idx_img < self.imgs_range[1]
        
        img_path = os.path.join(self.images_path, '%06d.jpg'%idx_img)
        #img = io.imread(img_path)
        img = Image.open(img_path)
        
        img = self.transform(img).type(torch.FloatTensor)
        #print('img:', img)
        return img


class TestDataset(Dataset):
    def __init__(self, imgs_range, images_path):
        self.imgs_range = imgs_range
        self.num_imgs = imgs_range[1]-imgs_range[0] 
        self.images_path = images_path

        self.transform = tf.Compose([
            tf.ToTensor(),
            tf.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]),
        ])
    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        idx_img = self.imgs_range[0] + idx
        assert idx_img < self.imgs_range[1]
        
        img_path = os.path.join(self.images_path, '%06d.jpg'%idx_img)
        #img = io.imread(img_path)
        img = Image.open(img_path)
        img = self.transform(img).type(torch.FloatTensor)
        return img