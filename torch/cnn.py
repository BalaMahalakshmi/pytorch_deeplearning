import torch 
from torch import nn 

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import requests
from pathlib import Path
# from helper_functions import accuracy_fn
from timeit import default_timer as timer 
from tqdm import tqdm


class FashinonMNISTModelv2(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape,
                    out_channels=output_shape,
                    kernel_size=3,
                    stride=1,
                    padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape,
                    out_channels=output_shape,
                    kernel_size=3,
                    stride=1,
                    padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
        )

        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*0,
                      out_features=output_shape)
        )
    def forward(self, x):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        return x


train_data = datasets.FashionMNIST(root="data", train=True, download=True, 
                                   transform=torchvision.transforms.ToTensor(), target_transform=None)
test_data = datasets.FashionMNIST(root="data", train=False, download=True, 
                                   transform=ToTensor(), target_transform=None)
class_name = train_data.classes
torch.manual_seed(42)
model_2 = FashinonMNISTModelv2(input_shape=1,
                               hidden_units=10,
                               output_shape=len(class_name)).to("cpu")

# print(model_2)
# print(model_2.state_dict())
torch.manual_seed(42)
images = torch.randn(size=(32,3,64,64))
test_image = images[0]
# print(f"image batch shape: {images.shape}")
# print(f"single image shape: {test_image.shape}")
# print(f"test image:\n {test_image}")

conv_layer = nn.Conv2d(in_channels=3,
                       out_channels=10,
                       kernel_size=(3,3),
                       stride=1,
                       padding=0)
con_output =conv_layer(test_image)
# print(con_output)
# print(test_image.unsqueeze(0).shape)

