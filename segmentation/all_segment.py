import torch
from segment_anything import sam_model_registry

torch.cuda.empty_cache()
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint='segmentation/sam_vit_h_4b8939.pth')
sam.to(device=DEVICE)

import cv2
from segment_anything import SamAutomaticMaskGenerator

mask_generator = SamAutomaticMaskGenerator(sam)

image_bgr = cv2.imread('datasets_resized/60m/12032021/DJI_0010.JPG')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
result = mask_generator.generate(image_rgb)

import supervision as sv

mask_annotator = sv.MaskAnnotator()
detections = sv.Detections.from_sam(result)
annotated_image = mask_annotator.annotate(image_bgr, detections)
cv2.imwrite('segmentation/results/third_sam_result.jpg', annotated_image)
