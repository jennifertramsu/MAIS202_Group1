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
from data_processing import grey2three

# Helpers
from visualize import show_image

# Models
from transformers import ViTMAEForPreTraining, ViTFeatureExtractor, Trainer, TrainingArguments
import torch
from torch.utils.data import DataLoader

# ----------------------------------------------------------------------------------------------
# Train Test Split - Update with new data

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

l_train_3 = [grey2three(i) for i in l_train]

print(l_train_3)

t = np.append()

# ----------------------------------------------------------------------------------------------
# Training Model

feature_extractor = ViTFeatureExtractor.from_pretrained("facebook/vit-mae-base")
model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy='epoch')

from sklearn.metrics import mean_absolute_error, mean_squared_error

def compute_metrics(ref, pref, sample_weight=None, multioutput="uniform average", squared=True):

    score_mse = mean_squared_error(ref, prod, sample_weight, multioutput, squared)
    score_mae = mean_absolute_error(ref, pred, sample_weight, multioutput)

    return {"mae": score_mae, 'mse': score_mse}

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = l_train_dataset,
    eval_dataset = l_test_dataset,
    compute_metrics = compute_metrics
)

#trainer.train()

# ----------------------------------------------------------------------------------------------
# Archive 

# ViTModel itself doesn't have unpatchify method, which is needed to visualize output
#feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
#model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
