import sys
import csv
import os
import pandas as pd
import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients
import torchvision.transforms as tf
from torchvision.models import resnet50
import matplotlib.pyplot as plt
from skimage import io
from matplotlib import gridspec
def read_labels(label_path, cat_path):
    with open(label_path, 'r') as labfp:
        lab_csv = csv.DictReader(labfp)
        return np.array([int(row['TrueLabel']) for row in lab_csv], dtype=np.int64)

def read_img_dir(dir_path, num_imgs):
    img_path = [os.path.join(dir_path, '%03d'%i + '.png') for i in range(num_imgs)]
    return [io.imread(img_path[i]) for i in range(num_imgs)]

if __name__ == "__main__":
    model = resnet50(pretrained=True)
    model.eval()
    # *** parameters ***
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    loss_func = nn.CrossEntropyLoss()
    trans = tf.Compose([tf.Normalize(mean=mean, std=std)])
    inv_trans = tf.Compose([tf.Normalize(mean=(-mean/std), std=(1/std))])
    
    # read files
    tar_labels = read_labels('labels.csv', 'categories.csv') # shape=200
    cat = pd.read_csv('categories.csv', sep=',', dtype={'CategoryId':int, "CategoryName":str})
    imgs = read_img_dir(sys.argv[1], tar_labels.shape[0])
    out_imgs = read_img_dir(sys.argv[2], tar_labels.shape[0])
    print(cat)
    def predict(img):
        img = tf.ToTensor()(img).type(torch.FloatTensor)
        img = trans(img)
        img= img.unsqueeze(0)
        img.requires_grad = False

        predict = F.softmax(model(img), dim = 1).squeeze(0).detach().numpy()
        img= img.squeeze(0)
        img = inv_trans(img)
        img = torch.clamp(img, min=0.0, max=1).numpy()
        img = np.transpose(img, (1,2,0))

        return img, predict

    def bar_input(pred):
        inx = np.argsort(-pred)
        x = inx[:3].tolist()
        y = [pred[i] for i in x]
        x = ['%s(%d)'%(cat["CategoryName"][i], i) for i in x]
        print(x, y)
        print(tar_labels[i])
        return x, y
    rat = [1,2, 3]
    for i in rat:
        output_path = os.path.join(sys.argv[1], '%03d'%i + '.png')
        print('processing %s'%output_path)

        o_img, o_pred = predict(imgs[i])
        att_img, att_pred = predict(out_imgs[i])
        plt.figure(figsize=(23,4))
        gs = gridspec.GridSpec(1, 3, width_ratios=[0.5, 1, 1]) 
        plt.subplot(gs[0])
        plt.imshow(o_img)
        plt.axis('off')
        plt.subplot(gs[1])
        plt.title('Original Image')
        x, y = bar_input(o_pred)
        plt.bar(x, y)

        plt.subplot(gs[2])
        plt.title('Adversarial Image')
        x2, y2 = bar_input(att_pred)
        plt.bar(x2, y2)
        
        plt.suptitle('%03d'%i + '.png')
        plt.savefig('%d.jpg'%i )
        plt.close()
        
    
        