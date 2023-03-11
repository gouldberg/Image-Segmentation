
import os
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, utils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
import json


# ----------------------------------------------------------------------------------------------------------------------
# resnet18 pretrained model
# ----------------------------------------------------------------------------------------------------------------------

model = models.resnet18(pretrained=True)
print(model)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


# ----------------------------------------------------------------------------------------------------------------------
# get nn.Conv2d layers
# ----------------------------------------------------------------------------------------------------------------------

model_children = list(model.children())

print(model_children)
print(len(model_children))

print(model_children[0])


# ----------
model_weights = []
conv_layers = []

counter = 0
for i in range(len(model_children)):

    if type(model_children[i]) == nn.Conv2d:
        counter += 1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])

    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    counter += 1
                    model_weights.append(child.weight)
                    conv_layers.append(child)

print(f"Total convolution layers: {counter}")
print(conv_layers)


# ----------------------------------------------------------------------------------------------------------------------
# weight shape
# ----------------------------------------------------------------------------------------------------------------------

for weight, conv in zip(model_weights, conv_layers):
    print('--------------------------------')
    print(f"CONV: {conv} ====> SHAPE: {weight.shape}")


# ----------------------------------------------------------------------------------------------------------------------
# visualize the first conv layer filters
# ----------------------------------------------------------------------------------------------------------------------

layer_index = 0

print(conv_layers[layer_index])
print(model_weights[layer_index].shape)


# ----------
plt.figure(figsize=(20, 17))

for i, filter in enumerate(model_weights[layer_index]):
    # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
    plt.subplot(8, 8, i + 1)
    plt.imshow(filter[0, :, :].detach().cpu(), cmap='gray')
    plt.axis('off')
    # plt.savefig('../outputs/filter.png')

plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# load image --> generate feature maps
# ----------------------------------------------------------------------------------------------------------------------

dat_path = '/media/kswada/MyFiles/dataset/cub_200_2011/CUB_200_2011/images/005.Crested_Auklet'
img_file = os.path.join(dat_path, 'Crested_Auklet_0057_794932.jpg')


# ----------
image = Image.open(img_file)
plt.imshow(image)
plt.show()


# ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0., std=1.)
])

image = transform(image)
print(image.shape)

image = image.unsqueeze(0)
print(image.shape)

image = image.to(device)


# ----------
feature_maps = []
layer_names = []

for layer in conv_layers[0:]:
    image = layer(image)
    feature_maps.append(image)
    layer_names.append(str(layer))

print(len(feature_maps))
print(layer_names)

for idx, fmap in enumerate(feature_maps):
    print('--------------------------------')
    print(layer_names[idx])
    print(fmap.shape)


# ----------------------------------------------------------------------------------------------------------------------
# now convert 3D tensor to 2D, sum the same element of every channel
# ----------------------------------------------------------------------------------------------------------------------

processed = []

for fmap in feature_maps:
    fmap = fmap.squeeze(0)
    gray_img = torch.sum(fmap, 0)
    gray_img = gray_img / fmap.shape[0]
    processed.append(gray_img.data.cpu().numpy())

for fm in processed:
    print(fm.shape)


# ----------------------------------------------------------------------------------------------------------------------
# plotting feature maps
# ----------------------------------------------------------------------------------------------------------------------

fig = plt.figure(figsize=(12, 20))

for i in range(len(processed)):
    a = fig.add_subplot(5, 4, i + 1)
    plt.imshow(processed[i])
    a.axis("off")
    a.set_title(layer_names[i].split('(')[0] + f'-{i+1}', fontsize=10)

plt.show()

# plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')


########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
# again feature maps from image
# ----------------------------------------------------------------------------------------------------------------------

dat_path = '/media/kswada/MyFiles/dataset/cub_200_2011/CUB_200_2011/images/005.Crested_Auklet'
img_file = os.path.join(dat_path, 'Crested_Auklet_0057_794932.jpg')


# ----------
image = Image.open(img_file)

# ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0., std=1.)
])

image = transform(image)
print(image.shape)

image = image.unsqueeze(0)
print(image.shape)

image = image.to(device)


# ----------
# pass the image through all the layers
results = [conv_layers[0](image)]

for i in range(1, len(conv_layers)):
    # pass the result from the last layer to the next layer
    results.append(conv_layers[i](results[-1]))
# make a copy of the `results`
outputs = results


# ----------
# visualize 64 features from each layer
# (although there are more feature maps in the upper layers)

for num_layer in range(len(feature_maps)):
    plt.figure(figsize=(30, 30))
    layer_viz = feature_maps[num_layer][0, :, :, :].detach().cpu()
    layer_viz = layer_viz.data
    print(layer_viz.size())
    for i, filter in enumerate(layer_viz):
        if i == 64: # we will visualize only 8x8 blocks from each layer
            break
        plt.subplot(8, 8, i + 1)
        plt.imshow(filter, cmap='gray')
        plt.axis("off")
    # plt.savefig(f"../outputs/layer_{num_layer}.png")
    plt.show()
    # plt.close()


