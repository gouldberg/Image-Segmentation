
import os
import torch
import torch.nn as nn
import torchvision


# ----------------------------------------------------------------------------------------------------------------------
# CIFAR100 data loader
# ----------------------------------------------------------------------------------------------------------------------

cifar100_data = torchvision.datasets.CIFAR100(
    './cifar-100', train=True, download=True,
    transform=torchvision.transforms.ToTensor())

data_loader = torch.utils.data.DataLoader(cifar100_data,  batch_size=4,  shuffle=True)


iterator = iter(data_loader)


# ----------------------------------------------------------------------------------------------------------------------
# torch.nn.Conv2d
# torch.nn.MaxPool2d
# ----------------------------------------------------------------------------------------------------------------------

# with padding=1
conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
pool = torch.nn.MaxPool2d(2)
conv2 = torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)


# ----------
x, _ = next(iterator)

# (4, 3, 32, 32)
print(f'x: {x.shape}')


# ----------
x = conv1(x)
# (4, 16, 32, 32): now the channel is 16
print('after conv1:', x.shape)


# ----------
x = torch.relu(x)
x = pool(x)
# (4, 16, 16, 16): now the feature map is 32*32 --> 16*16 by 2d pooling
print('after 1st pool:', x.shape)


# ----------
x = conv2(x)
# (4, 8, 16, 16): now the channel is 8
print('after conv2:', x.shape)


# ----------
x = torch.relu(x)
x = pool(x)
# (4, 8, 8, 8): now the feature map is 16*16 --> 8*8 by 2d pooling
print('after 2nd pool:', x.shape)


# ----------------------------------------------------------------------------------------------------------------------
# torch.nn.ConvTranspose2d
# ----------------------------------------------------------------------------------------------------------------------

enc = torch.nn.Sequential(
    torch.nn.Conv2d(3, 16, 3, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(16, 8, 3, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2)
)

convt1 = torch.nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2)
convt2 = torch.nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2)


# ----------
x, _ = next(iterator)

# (4, 3, 32, 32)
print(f'x: {x.shape}')


# ----------
x = enc(x)
# (4, 8, 8, 8)
print('after encoder:', x.shape)


# ----------
x = convt1(x)

# output_size = (input_sise - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
# = (8 - 1) * 2 - 2 * 0 + 1 * (2 - 1) + 1 = 14 + 0 + 1 + 1 = 16

# (4, 16, 16, 16)
print(x.shape)


# ----------
x = torch.relu(x)
x = convt2(x)
# (4, 3, 32, 32)
print(x.shape)


# ---------
x = torch.sigmoid(x)
# (4, 3, 32, 32)
print(x.shape)


########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
# Depthwise Separable Convolution
# https://faun.pub/depthwise-separable-convolutions-in-pytorch-fd41a97327d0
# ----------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn


"""
Normal Convolution and depthwise-separable convolutions 
should output a vector with the same dimensions
input shape = 3, 28, 38 (RGB image of 28x 28)
output shape = 10, 28, 28 (10 output channels with same width and height)
"""

input = torch.rand(3, 28, 28)
print(input.shape)


# ----------
# conv2d normal
conv_layer = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, padding=1)
conv_layer_n_params = sum(p.numel() for p in conv_layer.parameters() if p.requires_grad)
conv_out = conv_layer(input)

print(f"Conv layer param numbers: {conv_layer_n_params}")
print(conv_out.shape)


# ----------
# depthwise convolution
# now add groups=3
depthwise_layer = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, groups=3)
depthwise_layer_n_params = sum(p.numel() for p in depthwise_layer.parameters() if p.requires_grad)
depthwise_out = depthwise_layer(input)

# parameters are very small (= 30)
print(f"DepthwiseConv layer param numbers: {depthwise_layer_n_params}")
print(depthwise_out.shape)


# ----------
# Depthwize-Separable Convolution:
#  - pointwise convolution (using depthwise output)
pointwise_layer = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=1)
pointwise_layer_n_params = sum(p.numel() for p in pointwise_layer.parameters() if p.requires_grad)
pointwise_out = pointwise_layer(depthwise_out)

print(f"PointwiseConv layer param numbers: {pointwise_layer_n_params}")
print(pointwise_out.shape)

print(conv_layer_n_params)
print(depthwise_layer_n_params + pointwise_layer_n_params)


