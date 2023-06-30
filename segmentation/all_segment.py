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
load classifier, predict mask class and annotate
'''
# svm_model = pickle.load(open('segmentation/saved_models/SVM_06282023.sav', 'rb'))       #CHANGE FILEPATH TO USE DIFFERENT MODELS
# categories = ['cropland', 'open-water','shrub', 'non-woody', 'wooded', 'other']
# colors = {'cropland':(255, 255, 0), 'open-water':(0,0,255), 'shrub':(255,165,0),'non-woody':(0,255,0),'wooded':(165,42,42),'other':(128,0,128)} #yellow, blue, orange, green, brown, purple
# highlighted_img = image_rgb.copy()

# model = models.resnet18(pretrained=True)
# layer = model._modules.get('avgpool')
# model.eval()

# scaler = transforms.Resize((224, 224))
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
# to_tensor = transforms.ToTensor()

# for mask in result:
#     alpha = np.where(mask['segmentation'], 255, 0).astype(np.uint8)
#     mask['cropped'] = np.concatenate((image_rgb, alpha[..., np.newaxis]), axis=2)
#     rows = np.any(mask['segmentation'], axis=1)
#     cols = np.any(mask['segmentation'], axis=0)
#     ymin, ymax = np.where(rows)[0][[0, -1]]
#     xmin, xmax = np.where(cols)[0][[0, -1]]
#     mask['cropped'] = mask['cropped'][ymin:ymax+1, xmin:xmax+1]
#     if (mask['cropped'].shape[0] * mask['cropped'].shape[1] < 0.02 * image_rgb.shape[0] * image_rgb.shape[1]):
#         continue
#     tensor = Variable(normalize(to_tensor(scaler(Image.fromarray(mask['cropped']).convert('RGB')))).unsqueeze(0))
#     embedding = torch.zeros(512)
#     def copy_data(m, i, o):
#         embedding.copy_(o.data.reshape(o.data.size(1)))
#     h = layer.register_forward_hook(copy_data)
#     model(tensor)
#     h.remove()
#     embedding = [t.numpy() for t in embedding]
#     mask['predicted'] = categories[svm_model.predict([embedding])[0]]
#     print(mask['predicted'])
#     highlighted_img[mask['segmentation']] = colors[mask['predicted']]

# plt.imshow(highlighted_img)
# plt.show()
# cv2.imwrite('segmentation/results/predicted2.jpg', highlighted_img)
