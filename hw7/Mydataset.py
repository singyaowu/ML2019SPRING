import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms as tf
from skimage import io
import os
class TrainDataset(Dataset):
    def __init__(self, imgs_range, images_path):
        self.imgs_range = imgs_range
        self.num_imgs = imgs_range[1]-imgs_range[0] 
        self.images_path = images_path

        self.transform = tf.Compose([
            #tf.ToPILImage(),
            #tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            #tf.RandomRotation(30),            
            #tf.RandomResizedCrop(48,scale=(0.8,1)),
            tf.ToTensor()
        ])
    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        idx_img = self.imgs_range[0] + idx
        assert idx_img < self.imgs_range[1]
        
        img_path = os.path.join(self.images_path, '%06d.jpg'%idx_img)
        img = io.imread(img_path)
        #print('img:', img.shape)
        img = self.transform(img).type(torch.FloatTensor)
        #print('tensor:', img.size())
        return img