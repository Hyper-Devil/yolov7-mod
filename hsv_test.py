#%%
import yaml
import matplotlib.pyplot as plt
#color = sns.color_palette()
import cv2
import numpy as np
%matplotlib inline

hyp_file="/home/mct/whd/yolov7-mod/data/hyp.scratch.tiny.yaml"

with open(hyp_file) as fp:
    hyp = yaml.load(fp)

def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    img_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)  
    return img_hsv

#%%
path="/home/mct/whd/yolov7-mod/datasets/VOCdevkit/images/train/0619school-0101.jpg"
img = cv2.imread(path)
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)

#%%
img_hsv = augment_hsv(img,hyp['hsv_h'],hyp['hsv_s'],hyp['hsv_v'])
img_rgb = cv2.cvtColor(img_hsv,cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)