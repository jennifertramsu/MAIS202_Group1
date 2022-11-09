# ----------------------------------------------------------------------------------------------
# MAIS202 Group 1 - Fall 2022 Final Project
# Jennifer Tram Su, Lucy Mao
# Image Colourization using Visual Transformers
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

# Models
from transformers import AutoFeatureExtractor, AutoModelForImageClassification



extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

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
