import os
import sys
import numpy as np 
from skimage.io import imread, imsave

IMAGE_PATH = sys.argv[1] #'Aberdeen'

# Images for compression & reconstruction
test_image = [sys.argv[2]] #['1.jpg','10.jpg','22.jpg','37.jpg','72.jpg'] 
# Number of principal components used
k = 5

def process(M): 
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    return M

if __name__ == '__main__':
    filelist = os.listdir(IMAGE_PATH)
    filelist = [filename for filename in filelist if filename[0] != '.']
    # Record the shape of images
    img_shape = imread(os.path.join(IMAGE_PATH,filelist[0])).shape 

    img_data = []
    for filename in filelist:
        tmp = imread(os.path.join(IMAGE_PATH,filename))  
        img_data.append(tmp.flatten())

    training_data = np.array(img_data).astype('float32')

    # Calculate mean & Normalize
    mean = np.mean(training_data, axis = 0)  
    training_data -= mean 
    print('mean shape:',mean.shape)
    # Use SVD to find the eigenvectors 
    u, s, v = np.linalg.svd(training_data.T, full_matrices = False, compute_uv=True)
    u_principle = u[:,:k] #(m,k)
    s_principle = s[:k]

    for x in test_image: 
        # Load image & Normalize
        picked_img = imread(os.path.join(IMAGE_PATH,x))  
        X = picked_img.flatten().astype('float32') 
        X -= mean
        
        # Compression
        #weight = np.array([picked_img.dot(u_principle.T[i]) for i in range(k)])  
        weight = (X.reshape(1, -1)).dot(u_principle)
        print('weight shape:', weight.shape)
        print(weight)
        # Reconstruction
        reconstruct = process(weight.dot(u_principle.T).flatten() + mean)
        imsave(sys.argv[3], reconstruct.reshape(img_shape))
        #imsave(x[:-4] + '_reconstruction.jpg', reconstruct.reshape(img_shape)) 

    #average = process(mean)
    #imsave('average.jpg', average.reshape(img_shape))

    #for x in range(k):
    #    eigenface = process(u_principle.T[x])
    #    imsave(str(x) + '_eigenface.jpg', eigenface.reshape(img_shape))

    #for i in range(k):
    #    number = s[i] * 100 / sum(s)
    #    print(number)