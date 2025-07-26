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

batchsize = 32
train_dataloader = DataLoader(dataset = train_data, batch_size= batchsize, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=batchsize, shuffle=True)
# print(train_dataloader, test_dataloader)

# print(f"Dataloaders: {train_dataloader, test_dataloader}")
# print(f"length of train_dataloader: {len(train_dataloader)} batches of {batchsize}")
# print(f"length of test_dataloader: {len(test_dataloader)} batches of {batchsize}")


train_features_batch , train_labels_batch = next(iter(train_dataloader))
# print(train_features_batch.shape, train_labels_batch.shape)
torch.manual_seed(42)
rand_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
# print(rand_idx)
img, label = train_features_batch[rand_idx], train_labels_batch[rand_idx]
# plt.imshow(img.squeeze(), cmap='gray')
# plt.show()
plt.title(class_name[label])
plt.axis(False)
# print(f"image size: {img.shape}")
# print(f"labels: {label}, label size: {label.shape}")


flatten_model = nn.Flatten()
x = train_features_batch[0]
# print(x)
op = flatten_model(x)
# print(f"shape before flattening: {x.shape}")
# print(f"shape after flattening: {op.shape}")

class FashionMNISTModelv0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(nn.Flatten(), 
                                         nn.Linear(in_features=input_shape,
                                                   out_features=hidden_units), 
                                          nn.Linear(in_features=hidden_units, 
                                                    out_features=output_shape))
        # print(self.layer_stack)

    def forward(self, x):
        return self.layer_stack(x)
    

torch.manual_seed(42)
m0 = FashionMNISTModelv0(input_shape=784,
                         hidden_units=10,
                         output_shape=len(class_name)).to("cpu")
# print(m0)

dummy_x = torch.rand([1,1,28,28])
# print(m0(dummy_x))
# print(m0.state_dict())


# if Path ("helper_functions.py").is_file():
#     # print("helper_functions.py already exits, skipping download")
# else:
#     # print("download helper_functions.py")
#     with open("helper_functions.py", "wb") as f:
#         f.write(requests.content)

lf = nn.CrossEntropyLoss()
optim = torch.optim.SGD(params=m0.parameters(), lr=0.1)
# print(lf, optim)

def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
    total_time = end -start
    # print(f"train time on {device}: {total_time:.3f} seconds")
    return total_time
start_time = timer()
end_time = timer()
print_train_time(start=start_time, end=end_time, device='cpu')



