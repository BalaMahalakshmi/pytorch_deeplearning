import torch 
from torch import nn 

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt

# print(torch.__version__)
# print(torchvision.__version__)

train_data = datasets.FashionMNIST(root="data", train=True, download=True, 
                                   transform=torchvision.transforms.ToTensor(), target_transform=None)
test_data = datasets.FashionMNIST(root="data", train=False, download=True, 
                                   transform=ToTensor(), target_transform=None)
# print(train_data, test_data)
# print(len(test_data), len(train_data))
img, label = train_data[0]
# print(img, label)
class_name = train_data.classes
# print(class_name)
class_to_index = train_data.class_to_idx
# print(class_to_index)
# print(train_data.targets)
# print(img.shape, label)

img,label = train_data[0]
# print(f"immage label: {class_name[label]}")
# print(f"image shape: {img.shape}")
# plt.imshow(img.squeeze())
# plt.imshow(img.squeeze(), cmap = "gray")
plt.title(class_name[label])
# plt.show()
torch.manual_seed(42)
plt.close('all')
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows*cols+1):
    rand_idx = torch.randint(0, len(train_data), size=[1]).item()
    # print(rand_idx)
    img,label = train_data[rand_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_name[label])
    plt.axis(False)
    # plt.show()

