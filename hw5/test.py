import sys
import csv
import os
import numpy as np
import torch 
import torch.nn as nn
import torchvision.transforms as transform
from torch.autograd.gradcheck import zero_gradients
from PIL import Image
from torchvision.models import resnet50

def read_categories(label_path, cat_path):
    with open(label_path, 'r') as labfp:
        lab_csv = csv.DictReader(labfp)
        return np.array([int(row['TrueLabel']) for row in lab_csv], dtype=np.int64)

def read_img_dir(dir_path, num_imgs):
    img_path = [os.path.join(dir_path, '%03d'%i + '.png') for i in range(num_imgs)]
    return [Image.open(img_path[i]) for i in range(num_imgs)]

if __name__ == "__main__":
    model = resnet50(pretrained=True)
    model.eval()
    # read files
    tar_labels = read_categories('labels.csv', 'categories.csv') # shape=200
    attack_imgs = read_img_dir(sys.argv[2], tar_labels.shape[0])
    raw_imgs = read_img_dir(sys.argv[1], tar_labels.shape[0])

    pred_labels = np.empty(shape=tar_labels.shape)
    
    for i in range(tar_labels.shape[0]):
        img = attack_imgs[i]
        
        tar_label = torch.tensor([tar_labels[i]]).type(torch.LongTensor)
        print(tar_label)
        trans = transform.Compose([transform.ToTensor()])
        inv_trans = transform.Compose([transform.ToPILImage()])
        img = trans(img)
        img = img.unsqueeze(0)
        output = model(img)
        pred_labels[i] = torch.max(output, 1)[1]
        
        img = img.squeeze(0)
        img = np.array(inv_trans(img))
        

        
    
