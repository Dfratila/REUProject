import torch
from segment_anything import sam_model_registry
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from skimage.transform import resize

def has_blank_pixels(image):
    pixels = image.getdata()
    for pixel in pixels:
        if pixel[3] == 0:  # Assuming blank pixels have an alpha value of 0
            return True
    return False

def get_non_blank_crops(np_image):
    image = Image.fromarray(np_image)
    width, height = image.size
    crops = []
    for y in range(0, height, 32):
        for x in range(0, width, 32):
            crop = image.crop((x, y, x+32, y+32))
            if not has_blank_pixels(crop):
                crops.append(np.array(crop))
    return crops #returns list of np arrays

'''
load in SAM and automatically generate masks without prompts
'''

torch.cuda.empty_cache()
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint='segmentation/sam_vit_h_4b8939.pth')
sam.to(device=DEVICE)

import cv2
from segment_anything import SamAutomaticMaskGenerator

mask_generator = SamAutomaticMaskGenerator(sam, points_per_batch=64) #default 64

image_bgr = cv2.imread('datasets_resized/60m/10272021/DJI_0659.JPG')
#image_bgr = cv2.imread('orthomosaic/results/2up_good.png')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
result = mask_generator.generate(image_rgb) #list of dicts for each mask

'''
training dataset mask generation, comment out when done generating data
'''

idx = 277
for mask in result:
    alpha = np.where(mask['segmentation'], 255, 0).astype(np.uint8)
    mask['cropped'] = np.concatenate((image_rgb, alpha[..., np.newaxis]), axis=2)
    rows = np.any(mask['segmentation'], axis=1)
    cols = np.any(mask['segmentation'], axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    mask['cropped'] = mask['cropped'][ymin:ymax+1, xmin:xmax+1]
    if (mask['cropped'].shape[0] * mask['cropped'].shape[1] < 0.02 * image_rgb.shape[0] * image_rgb.shape[1]):  #ignore masks that are less than 2% of image's total volume
        mask['cropped'] = None
    else:
        # plt.imshow(mask['cropped'])
        # plt.show()
        mask['cropped'] = resize(mask['cropped'], (256, 256))
        plt.imsave(f'segmentation/training/mask{idx}.jpg', np.ascontiguousarray(mask['cropped']))
        idx += 1
        
import supervision as sv

mask_annotator = sv.MaskAnnotator()
detections = sv.Detections.from_sam(result) #Detections object with attributes xyxy (np array, bounding boxes of masks), masks (np array, actual masks)
annotated_image = mask_annotator.annotate(image_bgr, detections)
plt.imshow(annotated_image)
plt.show()
#cv2.imwrite('segmentation/results/2up_good_resultjpg', annotated_image)