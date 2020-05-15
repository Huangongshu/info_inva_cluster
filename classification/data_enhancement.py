# -*- coding: utf-8 -*-
#@author: huan

from PIL import Image, ImageEnhance
import cv2
import numpy as np

def random_flip(image, mode=Image.FLIP_LEFT_RIGHT):
    random_model = np.random.randint(0, 2)
    if random_model == 0:
        return image
    else:
        return image.transpose(mode)
 
def random_rotation(image, rotate_range, mode=Image.BILINEAR):
    random_angle = np.random.uniform(-rotate_range, rotate_range)
    return image.rotate(random_angle, mode)
 
def random_shift(image, shift_range = 0.1):
    w, h = image.size
    shift = np.ceil(w * shift_range)
    val = np.random.uniform(0, shift)
    random_shift = np.random.randint(0, 4)
    crop_region = [(val, 0, w + val, h), #left shift
                   (-val, 0, w - val, h), #top shift                   
                   (0, val, w, h + val), #right shift
                   (0, -val, w, h - val)] #right
    image = image.crop(crop_region[random_shift])
    return image  

def random_zoom(image, zoom_range = 0.1):
    w, h = image.size
    val = np.random.uniform(-zoom_range, zoom_range)
    var_w = int((val * w) / 2)
    var_h = int((val * h) / 2)
    new_w = w + var_w * 2
    new_h = h + var_h * 2    
    image = image.resize((new_w, new_h))        
    image = image.crop((var_w, var_h, w + var_w, h + var_h))
    return image     

def random_crop(image, crop_range = 0.1, mode = Image.BILINEAR):
    w, h = image.size
    if isinstance(crop_range, float):
        crop_im_w = np.ceil(w * crop_range)
        crop_im_h = np.ceil(h * crop_range)
        preseve_w = int(w * (1 - crop_range)) 
        preseve_h = int(h * (1 - crop_range))
    elif isinstance(crop_range, tuple) or isinstance(crop_range, list):    
        crop_im_w = np.ceil(w - crop_range[0])
        crop_im_h = np.ceil(h - crop_range[1])        
        preseve_w = crop_range[0]
        preseve_h = crop_range[1]
    x = np.random.uniform(0, crop_im_w)
    y = np.random.uniform(0, crop_im_h)
    crop_region = (x, y, x + w + preseve_w, y + preseve_h)
    return image.crop(crop_region).resize((w, h))
 
def random_centre_crop(image, crop_range = 0.1, mode = Image.BILINEAR):
    w, h = image.size
    crop_im_w = np.ceil(w * crop_range)
    crop_im_h = np.ceil(h * crop_range)
    x = np.random.uniform(0, crop_im_w)
    y = np.random.uniform(0, crop_im_h) 
    crop_region = (x // 2, y // 2, int(w - x // 2), int(h - y // 2))
    return image.crop(crop_region).resize((w, h))

def random_color(image, color_range = 0.2, contrast_range = 0.2, brightness_range = 1.3, Sharpness = 3):
    random_factor = np.random.uniform(1 - color_range, 1 + color_range)  # random factor
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # saturation
    random_factor = np.random.uniform(0.5, brightness_range)
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # brighness
    random_factor = np.random.uniform(1 - contrast_range, 1 + contrast_range)
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # contrast
    random_factor = np.random.uniform(0, Sharpness) 
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # Sharpness
 
def random_noisy(image, noise_sigma = 25): #Gaussian noisy
    w, h = image.size
    noisy = (np.random.randn(w, h).astype(np.uint8) * noise_sigma).astype(np.uint8)
    noisy = np.stack([noisy,noisy, noisy], axis = -1)
    image = np.asarray(image)
    image = np.add(image, noisy)
    return Image.fromarray(image)

def get():
    # im_path = '/home/zhangyun/EyeImagepreprocessed/dataset_2/afterResize/' + path
    # heatmap = '/home/zhangyun/EyeImagepreprocessed/dataset_2/heatmap2/' + path
    im = cv2.imread('C:/Users/huan/Desktop/2.png')
    heatmap = cv2.imread('C:/Users/huan/Desktop/0a09aa7356c0.png')  
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
    heatmap = heatmap / 255
    test_m = heatmap.copy()  
    test_m[test_m >= 0.35] = 1
    test_m[test_m < 0.35] = 0
    test_m = np.stack([test_m, test_m, test_m], axis = 2)
    
    mask = test_m.copy()
    mask[test_m == 1] = 0
    mask[test_m == 0] = 1
    
    im = im.astype(float)
    k = cv2.getGaussianKernel(10, 0)
    kern = np.dot(k, k.T)
    temp = cv2.filter2D(im.copy(), 3, kern)
    goal_im = temp * mask + im * test_m
    goal_im = np.clip(goal_im, 0, 255).astype(np.uint8) 
    return Image.fromarray(cv2.cvtColor(goal_im, cv2.COLOR_BGR2RGB))


if __name__=='__main__':  
    im = get()
    im = im.resize((1000, 1200))
    im = random_flip(im)
    im = random_shift(im, shift_range = 0.2)
    im = random_crop(im, crop_range = 0.2)
    # im = random_rotation(im, rotate_range = 180)
    # im = random_color(im)   
    # im = random_noisy(im, noise_sigma = 10)  
    im = np.asarray(im)  
    # goal_im = cv2.addWeighted(np.array(im), 4, cv2.GaussianBlur(np.array(im), (0,0) ,10), -4, 128)
    # cv2.imshow('', goal_im)
