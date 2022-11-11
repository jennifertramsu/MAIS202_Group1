from transformers import ViTMAEForPreTraining, ViTFeatureExtractor
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import requests

feature_extractor = ViTFeatureExtractor.from_pretrained("facebook/vit-mae-base")
model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")

imagenet_mean = np.array(feature_extractor.image_mean)
imagenet_std = np.array(feature_extractor.image_std)

def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)

y = model.unpatchify(outputs.logits)
y = torch.einsum('nchw->nhwc', y).detach().cpu()

plt.subplot(1, 2, 1)
show_image(y[0], "reconstruction")

plt.show()



# save logits to array??



