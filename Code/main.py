# ----------------------------------------------------------------------------------------------
# MAIS202 Group 1 - Fall 2022 Final Project
# Jennifer Tram Su, Lucy Mao
# Image Colourization using Vision Transformers (ViT)
#
# Input : Single channel representation of greyscale image
# Output : Two/three channel representation of colourized image
#
# ----------------------------------------------------------------------------------------------

# File Handling
import glob

# Data Manipulation
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 

# Models
#from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from transformers import AutoFeatureExtractor, ViTMAEConfig, ViTMAEModel

#extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
#model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

config = ViTMAEConfig()
model = ViTMAEModel(config)

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

plt.imshow(l_test[0], cmap='gray')
plt.show()

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
