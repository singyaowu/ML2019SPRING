import sys
import os
import csv
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import Model
from lime import lime_image
import skimage.color as color
from skimage.segmentation import slic
import numpy as np
from torch.optim import Adam
import Model
EPOCH = 50
LEARNING_RATE = 0.1
if __name__ == "__main__":

    model = Model.MyCNN().cuda().eval()
    model.load_state_dict(torch.load('model_params0.6896.pkl'))
    part_model = model.conv1[:5]
    print(part_model)
    numFilter = 64
    for filterID in range(numFilter):
        img = np.uint8(np.random.uniform(150, 180, (1, 1, 48, 48)))/255 
        img_var = Variable(torch.tensor(img).type(torch.FloatTensor).cuda()
            , requires_grad=True)
        optimizer = Adam([img_var], lr=LEARNING_RATE, weight_decay=1e-6)
        for epoch in range(EPOCH):
            optimizer.zero_grad()
            output = part_model(img_var)
            output = output[0, filterID, :, :]
            loss = -torch.mean(output)
            loss.backward()
            optimizer.step()
        img_new = img_var.cpu().detach().numpy()
        img_new = img_new.reshape(48, 48)
        layer = '2nd Conv2d'
        plt.subplot(8, numFilter//8, filterID+1)
        plt.imshow(img_new, cmap=plt.cm.gray)
        plt.xlabel("filter" + str(filterID))
        plt.axis('off')
    plt.suptitle('Filters of layer conv2d_2')
    plt.savefig(os.path.join(sys.argv[1], 'fig2-1.jpg'))
    plt.close()
    imgs = np.load('cimgs.npy')
    for i in range(2, 3):
        img = imgs[i:(i+1)]
        img_var = Variable(torch.tensor(img).type(torch.FloatTensor).cuda()
            , requires_grad=True)
        output = part_model(img_var)
        
        for filterID in range(numFilter):
            output = part_model(img_var)
            output = output[0, filterID, :, :]
            fiter_img = output.cpu().detach().numpy().reshape(48, 48)
            layer = '2nd Conv2d'
            plt.subplot(8, numFilter//8, filterID+1)
            plt.imshow(fiter_img, cmap=plt.cm.gray)
            plt.xlabel("filter" + str(filterID))
            plt.axis('off')
        plt.suptitle('Filters of layer conv2d_2')
        plt.savefig(os.path.join(sys.argv[1], 'fig2-2.jpg'))
        plt.close()