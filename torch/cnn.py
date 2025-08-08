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
batchsize = 32
train_dataloader = DataLoader(dataset = train_data, batch_size= batchsize, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=batchsize, shuffle=True)
    
train_features_batch , train_labels_batch = next(iter(train_dataloader))
rand_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[rand_idx], train_labels_batch[rand_idx]


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

# print(test_image.shape)
# print(f"Test image original shape: {test_image.shape} ")
# print(f"Test image with unsqueezed dimension: {test_image.unsqueeze(0).shape}")
max_pool_layer = nn.MaxPool2d(kernel_size=2)
test_image_through_conv = conv_layer(test_image.unsqueeze(dim=0))
# print(f"shape after going through conv_layer(): {test_image_through_conv.shape}")
test_image_through_conv_and_max_pool = max_pool_layer(test_image_through_conv)
# print(f"shape after going through conv_layer() and max_pool_layer(): {test_image_through_conv}")

torch.manual_seed(42)
# create a random tensor
random_tensor = torch.randn(size=(1,1,2,2))
# print(f"\n Random tensor: \n{random_tensor}")
# print(f"Random tensor shape: {random_tensor.shape}")

# create a max pool layer 
max_pool_tensor = max_pool_layer(random_tensor)

# pass the random tensor through max pool layer
# print(f"\n Max pool tensor:\n {max_pool_tensor}")
# print(f"max pool tensor shape: {max_pool_tensor.shape}")
# print(random_tensor)

torch.manual_seed(42)
m0 = FashinonMNISTModelv2(input_shape=1,
                         hidden_units=10,
                         output_shape=len(class_name)).to("cpu")
plt.imshow(img.squeeze(), cmap='grey')
# plt.show()
