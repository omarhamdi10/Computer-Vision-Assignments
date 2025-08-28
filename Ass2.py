import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import requests
from io import BytesIO

def createkernel(size, direction):
    kernel = np.zeros((size, size))
    mid = size // 2
    
    if direction == 'x':
        kernel[:mid, :] = -1
        kernel[mid+1:, :] = 1
    elif direction == 'y':
        kernel[:, :mid] = -1
        kernel[:, mid+1:] = 1
    
    kernel[mid, :] = 0
    kernel[:, mid] = 0
    return kernel

def applyconvolution(image, kernel):
    imgheight, imgwidth = image.shape
    ksize = kernel.shape[0]
    pad = ksize // 2
    paddedimage = np.pad(image, pad, mode='constant')
    convolved = np.zeros_like(image, dtype=float)
    
    for i in range(imgheight):
        for j in range(imgwidth):
            region = paddedimage[i:i+ksize, j:j+ksize]
            convolved[i, j] = np.sum(region * kernel)
    return convolved

def edgedetection(image, maxkernelsize, threshold):
    image = np.array(image, dtype=float) / 255.0
    imgheight, imgwidth = image.shape
    
    magnitudemap = np.zeros((imgheight, imgwidth))
    kernelsizemap = np.zeros((imgheight, imgwidth), dtype=int)
    
    for size in range(3, maxkernelsize + 1, 2):
        kernelx = createkernel(size, 'x')
        kernely = createkernel(size, 'y')
        
        convx = applyconvolution(image, kernelx)
        convy = applyconvolution(image, kernely)
        
        for i in range(imgheight):
            for j in range(imgwidth):
                mag = math.sqrt((convx[i, j] ** 2 + convy[i, j] ** 2) / size**2)
                if mag > threshold and mag > magnitudemap[i, j]:
                    magnitudemap[i, j] = mag
                    kernelsizemap[i, j] = size
    return magnitudemap, kernelsizemap

def loadandprocessimage(imageurl, maxkernelsize=13, threshold=0.1):
    response = requests.get(imageurl)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content)).convert("L")
    else:
        raise Exception(f"Failed to download image. Status code: {response.status_code}")
    
    magnitudemap, kernelsizemap = edgedetection(img, maxkernelsize, threshold)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')
    
    plt.subplot(1, 3, 2)
    plt.title("Magnitude Map")
    plt.imshow(magnitudemap, cmap='gray')
    
    plt.subplot(1, 3, 3)
    plt.title("Kernel Size Map")
    plt.imshow(kernelsizemap, cmap='viridis')
    plt.colorbar()
    plt.show()

#imageurl = "https://networkcameratech.com/wp-content/uploads/2016/10/HIKVISION-DS-2CD2142FWD-I_2016-Nov-09_21_59_05.png"
#imageurl = "https://networkcameratech.com/wp-content/uploads/2016/10/HIKVISION-DS-2CD2142FWD-I_2016-Nov-09_21_52_01.png"
#imageurl = "https://networkcameratech.com/wp-content/uploads/2016/10/AXISP3364_2016-Oct-27_03_50_22.png"
imageurl = "https://www.mathworks.com/help/examples/matlab/win64/DisplayGrayscaleRGBIndexedOrBinaryImageExample_02.png"
loadandprocessimage(imageurl, maxkernelsize=13, threshold=0.1)
