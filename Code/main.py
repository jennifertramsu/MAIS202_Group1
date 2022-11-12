# ----------------------------------------------------------------------------------------------
# MAIS202 Group 1 - Fall 2022 Final Project
# Jennifer Tram Su, Lucy Mao
# Image Colourization using Vision Transformers (ViT)
#
# Input : Single channel representation of greyscale image
# Output : Two/three channel representation of colourized image
#
# References : 
# @Article{MaskedAutoencoders2021,
#   author  = {Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Doll{\'a}r and Ross Girshick},
#   journal = {arXiv:2111.06377},
#   title   = {Masked Autoencoders Are Scalable Vision Learners},
#   year    = {2021},
# }
# ----------------------------------------------------------------------------------------------

# File Handling
import glob

# Manipulation
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import requests

# Helpers
from visualize import show_image

# Models
from transformers import ViTMAEForPreTraining, ViTFeatureExtractor, ViTModel
import torch

# ----------------------------------------------------------------------------------------------

## Scripts / Functions Required

# 1 ) Preprocessing Code
# --> Images are converted to BW, resized to 224 x 224 pixels
# --> Trying both LAB and RGB colour spaces
# --> Data split into training and test sets

# 2 ) Training the Model
# --> Training inside loop, save loss at end of each iteration
# --> need training and loss function

# 3 ) Main Body
# --> This is where everything is put together
# --> Initializing hyperparameters, weights

# ----------------------------------------------------------------------------------------------
# Train Test Split

path = 'Datasets'
ab = path + '/ab/*'
l = path + '/l/*' 

lfiles = glob.glob(l)
abfiles = glob.glob(ab)

abs = [np.load(abfiles[i]) for i in range(len(abfiles))] # [num photos, dim1, dim2, 2]
ls = np.load(lfiles[0])

ab_train = np.append(abs[0], abs[1], axis=0) # abs[0], abs[1]
ab_test = abs[2]

l_train = ls[:20000, :, :]
l_test = ls[20000:, :, :]

# ----------------------------------------------------------------------------------------------

# Training from Scratch
# Can split greyscale into three identical channels

# ----------------------------------------------------------------------------------------------
# Building Model

feature_extractor = ViTFeatureExtractor.from_pretrained("facebook/vit-mae-base")
model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")

# ViTModel itself doesn't have unpatchify method, which is needed to visualize output
#feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
#model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

imagenet_mean = np.array(feature_extractor.image_mean)
imagenet_std = np.array(feature_extractor.image_std)

# Testing Print

#url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#image = Image.open(requests.get(url, stream=True).raw)

#inputs = feature_extractor(images=image, return_tensors="pt")

t = np.stack((l_train[0],)*3, axis=-1)

b = Image.fromarray(t)

inputs = feature_extractor(images=t, return_tensors="pt")

outputs = model(**inputs)

y = model.unpatchify(outputs.logits)
y = torch.einsum('nchw->nhwc', y).detach().cpu() # Einstein summation

plt.figure(1)
show_image(y[0], imagenet_mean, imagenet_std, title="Testing Visualization")

plt.show()