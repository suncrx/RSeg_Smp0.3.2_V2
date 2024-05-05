# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 14:44:03 2022

@author: renxi

https://albumentations.ai/docs/getting_started/transforms_and_targets/
https://albumentations.ai/docs/introduction/why_you_need_a_dedicated_library_for_image_augmentation/


Here is an example definition of an augmentation pipeline. This pipeline will 
first crop a random 512px x 512px part of the input image. Then with probability 
30%, it will randomly change brightness and contrast of that crop. 
Finally, with probability 50%, it will horizontally flip the resulting image.

import albumentations as A

transform = A.Compose([
    A.RandomCrop(512, 512),
    A.RandomBrightnessContrast(p=0.3),
    A.HorizontalFlip(p=0.5),
])

image = cv2.imread("/path/to/image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#For semantic segmentation, you usually read one mask per image. 
#Albumentations expects the mask to be a NumPy array. 
#The height and width of the mask should have the same values 
#as the height and width of the image.

mask = cv2.imread("/path/to/mask.png")

transformed = transform(image=image, mask=mask)
transformed_image = transformed['image']
transformed_mask = transformed['mask']



"""

import albumentations as albu

# ---------------------------------------------------------------
def get_train_aug(height=256, width=256):
    train_transform = [
        
        # spatial level augumentation--------------------
        albu.PadIfNeeded(min_height=height, min_width=width, 
                         border_mode=0),
        
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
                
        albu.RandomCrop(height=height, width=width, p=0.2),
        
        albu.Rotate(p=0.5),
        
        #albu.IAAPerspective(p=0.1),
        albu.Perspective(p=0.1),
        
        
        # pixel level augumentation--------------------
        albu.Downscale(scale_min=0.25, scale_max=0.25, p=0.1),
        albu.ToGray(p=0.2),
        #albu.IAAAdditiveGaussianNoise(p=0.2),
        #albu.GaussianNoise(p=0.2),
        albu.FancyPCA(alpha=0.1, p=0.2),
        
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.5
        ),

        albu.OneOf(
            [
                #albu.IAASharpen(p=1),
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.5,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.5,
        ),
    ]
    return albu.Compose(train_transform, p=0.8)


def get_val_aug(height=256, width=256):
    """resize the image to the times of 32"""
    test_transform = [
        albu.PadIfNeeded(height, width)
    ]
    return albu.Compose(test_transform, p=0.5)



if __name__ == '__main__':
    import cv2
    from matplotlib import pyplot as plt

    image = cv2.imread('data/0.jpg')
    mask = cv2.imread('data/0_gt.png', cv2.IMREAD_GRAYSCALE)

    aug = get_train_aug(256, 256)

    augmented = aug(image=image, mask=mask)
    aimage = augmented['image']
    amask = augmented['mask']
    plt.figure()
    plt.subplot(221)        
    plt.imshow(image)        
    plt.subplot(222)        
    plt.imshow(mask)    
    plt.subplot(223)        
    plt.imshow(aimage)        
    plt.subplot(224)        
    plt.imshow(amask)        
