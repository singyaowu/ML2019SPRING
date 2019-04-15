import sys
import csv
import os

import numpy as np
from numpy import linalg as LA

import torch 
import torch.nn as nn
from torch.autograd.gradcheck import zero_gradients

import torchvision.transforms as tf
from torchvision.models import resnet50

from PIL import Image

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
    # *** parameters ***
    epsilon = 0.035
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    loss_func = nn.CrossEntropyLoss()
    trans = tf.Compose([tf.ToTensor(), tf.Normalize(mean=mean, std=std)])
    inv_trans = tf.Compose([tf.Normalize(mean=(-mean/std), std=(1/std)), tf.ToPILImage()])
    
    # read files
    tar_labels = read_categories('labels.csv', 'categories.csv') # shape=200
    raw_imgs = read_img_dir(sys.argv[1], tar_labels.shape[0])
    
    pred_labels = np.empty(shape=tar_labels.shape)
    L_infs = np.empty(shape=tar_labels.shape)
    
    for i in range(tar_labels.shape[0]):
        output_path = os.path.join(sys.argv[2], '%03d'%i + '.png')
        print('processing %s'%output_path)

        img = raw_imgs[i].copy()
        tar_label = torch.tensor([tar_labels[i]]).type(torch.LongTensor)
        
        img = trans(img).type(torch.FloatTensor)
        img = img.unsqueeze(0)
        img.requires_grad = True
        
        zero_gradients(img)
        output = model(img)
        loss = -loss_func(output, tar_label)
        loss.backward()
        
        img = img - epsilon * img.grad.sign_() #(1, 3, 224, 224)
        img = img.detach()
        pred_labels[i] = torch.max(model(img), 1)[1]
        
        # inverse transforms
        img= img.squeeze(0)
        img = tf.Normalize(mean=(-mean/std), std=(1/std))(img)
        img = torch.clamp(img, min=0.0, max=1.0)
        img = tf.ToPILImage()(img)
        
        img.save(output_path)
        
        L_inf = max(abs( np.array(img, dtype=float).reshape(-1) - np.array(raw_imgs[i], dtype=float).reshape(-1)))
        L_infs[i] = L_inf
        print('ground Truth: %d, predict: %d, L-infinity:%f'%(tar_labels[i], pred_labels[i], L_infs[i]))
    
    success_rate = np.mean((pred_labels != tar_labels))
    L_inf = np.mean(L_infs)
    print('success rate:%f, L_infinity:%f'%(success_rate, L_inf))
        