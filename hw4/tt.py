import sys
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import Model
from copy import deepcopy
from lime import lime_image
import skimage.color as color
from skimage.segmentation import slic
REPORT_MODE = True
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

def show_saliency_maps(x, y, model):
    x_org = x.squeeze().numpy()
    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(x, y, model)

    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.detach().cpu().numpy()
    
    num_pics = x_org.shape[0]
    
    for i in range(num_pics):
        # You need to save as the correct fig names
        #plt.imsave('p3/pic_L'+str(int(y[i]))+'_i'+ str(i+offset)+'.png', x_org[i], cmap=plt.cm.gray)
        #plt.imsave('p3/pic_L'+str(int(y[i]))+'_i'+ str(i+offset)+'s.png', saliency[i], cmap=plt.cm.jet)
        if REPORT_MODE:
            plt.suptitle('Original / Saliency Map / Mask')
            ax = plt.subplot(1, 3, 1)
            plt.imshow(x_org[i], cmap=plt.cm.gray)
            plt.axis('off')
            divider = make_axes_locatable(ax)
            #cax = divider.append_axes("right", size="5%", pad=0.05)
            
            ax = plt.subplot(1, 3, 2)
            im = plt.imshow(saliency[i], cmap=plt.cm.jet)
            divider = make_axes_locatable(ax)
            plt.axis('off')
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            #plt.axis('off')
            ax = plt.subplot(1, 3, 3)
            e_max = np.amax(saliency[i])
            e_min = np.amin(saliency[i])
            thres = (e_max-e_min) / 6
            img3 = x_org[i] * (saliency[i] > thres) + (saliency[i] <= thres) * 0.1
            plt.axis('off')
            plt.imshow(img3, cmap=plt.cm.gray)
            divider = make_axes_locatable(ax)
            #cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.savefig('rep/test'+str(i)+'.jpg')
            #plt.show()
            plt.close()
        else:
            ax =  plt.gca()
            im = plt.imshow(saliency[i], cmap=plt.cm.jet)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

            plt.savefig(sys.argv[2]+'fig1_'+str(i)+'.jpg')
            plt.close()

def compute_saliency_maps(x, y, model):
    model.eval()
    x.requires_grad_()
    y_pred = model(x.cuda())
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(y_pred, y.cuda())
    loss.backward()

    saliency = x.grad.abs().squeeze().data
    return saliency
def explain(instance, predict_fn, **kwargs):
    np.random.seed(16)
    return exp.explain_instance(instance, predict_fn, **kwargs)

def predict(img):
    img = color.rgb2gray(img)
    img = img.reshape(img.shape[0], 1, 48, 48)
    t_img = torch.tensor(img).type(torch.FloatTensor).cuda()
    return model(t_img).detach().cpu().numpy()

def segmentation(img):
    return slic(img)

def gray2rgb(imgs):
    numData = imgs.shape[0]
    tmp = imgs.reshape(numData, 48,48)
    return color.gray2rgb(tmp)

def classify(imgs, label):
    img_list = [[] for i in range(7)]
    y_list = [[] for i in range(7)]
    for i in range(len(imgs)):
        y_list[label[i]].append(label[i])
        img_list[label[i]].append(imgs[i])
    num_pick = min([len(i) for i in img_list])
    num_pick = min(num_pick, 10)
    print('num_pick: ',num_pick)
    img = np.array(np.array(img_list[0][0:num_pick]) )
    lab = np.array(np.array(y_list[0][0:num_pick]) )
    for i in range(1, 7):
        img = np.concatenate((img, np.array(img_list[i][0:num_pick])), axis=0)
        lab = np.concatenate((lab, np.array(y_list[i][0:num_pick])), axis=0)
    return img, lab

if __name__ == "__main__":
    try:
        imgs = np.load('imgs.npy')
        label = np.load('label.npy')
    except:
        imgs, label = parse_csv(sys.argv[1])
        np.save('imgs.npy', imgs)
        np.save('label.npy', label)
    try:
        imgs = np.load('cimgs.npy')
        label = np.load('clabel.npy')
    except:
        imgs, label = classify(imgs, label)
        np.save('cimgs.npy', imgs)
        np.save('clabel.npy', label)
        
    imgs_tensor = torch.tensor(imgs).type(torch.FloatTensor)
    label_tensor = torch.tensor(label).type(torch.LongTensor)
    model = Model.MyCNN()
    model.load_state_dict(torch.load('model_params0.6896.pkl'))
    model.cuda()
    model.eval()
    print(label_tensor)
    s_imgs = np.concatenate((imgs[2:3],imgs[12:13],imgs[22:23],imgs[37:38]\
        ,imgs[45:46],imgs[52:53],imgs[65:66]), axis=0)
    s_imgs_tensor = torch.tensor(s_imgs).type(torch.FloatTensor)
    s_label_tensor = torch.tensor(np.arange(7)).type(torch.LongTensor)
 
    
    lime_imgs = np.concatenate((imgs[5:6],imgs[18:19],imgs[23:24],imgs[30:31]\
        ,imgs[41:42],imgs[52:53],imgs[60:61]), axis=0)

    for i in range(7):
        
        box_size = 8
        stripe = 8
        file_name = str(i)+'-face.npy'
        try:
            myimgs = np.load(file_name)
        except:    
            myimgs = deepcopy(lime_imgs[i:i+1])
            for up in range(0,48, stripe):
                for left in range(0, 48, stripe):
                    img_now = deepcopy(lime_imgs[i:i+1])
                    right = min(48, left+box_size)
                    bot = min(48, up+box_size)
                    img_now[0, 0, up:bot, left:right] = 0.1
                    myimgs = np.concatenate((myimgs, img_now), axis=0)
            np.save(file_name, myimgs)
        
        print(myimgs.shape)
        imgs_tensor = torch.tensor(myimgs).type(torch.FloatTensor).cuda()
        output = model(imgs_tensor)
        #output = output.cpu().detach().numpy()
        print(output.shape)
        f = plt.figure()
        f.suptitle("ground truth: %d"%i)
        f.set_figheight(10)
        f.set_figwidth(18)
        for j in range(myimgs.shape[0]):
            ax = plt.subplot(4, myimgs.shape[0]//4+1, j+1)
            plt.imshow(myimgs[j].reshape(48,48), cmap=plt.cm.gray)
            ax.set_title('score:%.3f\npredict: %d' %(output[j, i],torch.max(output,1)[1][j]))
            plt.axis('off')
        plt.savefig(str(i)+'-face.jpg')
        plt.close()



    