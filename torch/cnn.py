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
        nn.Conv2d(in_channels=20,
                    out_channels=10,
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


device ="cuda" if torch.cuda.is_available() else "cpu"

# def train_step(model: torch.nn.Module,
#                data_loader: torch.utils.data.dataloader,
#                loss_fn: torch.nn.Module,
#                optimizer: torch.optim.Optimizer,
#                accuracy_fn,
#                device: torch.device = None):
#     train_loss, train_acc = 0,0
#     model.train()

#     for batch, (x,y) in enumerate(train_dataloader):
#         x,y = x.to(device), y.to(device)
#         pred_y = model(x)
#         loss = lf(pred_y, y)
#         train_loss += loss
#         train_acc += accuracy_fn(y_true=y, pred=pred_y.argmax(dim=1))
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     train_loss /= len(data_loader)
#     train_acc /= len(data_loader)
#     print(f"Train loss: {train_loss:.5f} | Train_acc: {train_acc}")

    
def test_step(model: torch.nn.Module,
               data_loader: torch.utils.data.dataloader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = None):
    test_loss, test_acc = 0,0
    model.eval()
    with torch.inference_mode():
        for x, y in data_loader:
            x,y = x.to(device), y.to(device)
            test_pred = model(x)
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y, pred=test_pred.argmax(dim=1))
    test_loss = test_loss/ len(data_loader)
    test_acc = test_acc / len(data_loader)
#     # print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc}%n")

lf = nn.CrossEntropyLoss()
optim = torch.optim.SGD(params=m0.parameters(), lr=0.1)
# print(lf, optim)

def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
    total_time = end -start
def accuracy_fn(y_true, pred):
    correct = (pred == y_true).sum().item()
    total = len(y_true)
    return correct/total

torch.manual_seed(42)
train_time_start_on_m0 = timer()
epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"epoch:{epoch}\n----")
    train_step(model=m0,
               data_loader=train_dataloader,
               loss_fn=lf,
               optimizer=optim,
               accuracy_fn=accuracy_fn,
               device=device)
    test_step(model=m0,
               data_loader=train_dataloader,
               loss_fn=lf,
               optimizer=optim,
               accuracy_fn=accuracy_fn,
               device=device)
    train_time_end_on_m0=timer()
    total_train_time_m0 = print_train_time(start=train_time_start_on_m0,
                                           end=train_time_end_on_m0,
                                           device=device)
m0


