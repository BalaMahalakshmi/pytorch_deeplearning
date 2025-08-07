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
# print_train_time(start=start_time, end=end_time, device='cpu')


# def accuracy_fn(y_true, pred):
#     correct = torch.eq(y_true, pred).sum().item()
#     acc = (correct/len(pred))*100
def accuracy_fn(y_true, pred):
    correct = (pred == y_true).sum().item()
    total = len(y_true)
    return correct/total


torch.manual_seed(42)
train_time_start_on_cpu = timer()
epochs = 3
for epoch in tqdm(range(epochs)):
    # print(f"epoch:{epoch}\n----")
    train_loss = 0
    for batch, (x,y) in enumerate(train_dataloader):
        m0.train()
        pred_y = m0(x)
        loss = lf(pred_y, y)
        train_loss += loss
        optim.zero_grad()
        loss.backward()
        optim.step()
        if batch % 400 == 0:
            # print(f"looked at {batch * len(x)} / {len(train_dataloader.dataset)} samples.")
          train_loss, test_acc = 0, 0
m0.eval()
with torch.inference_mode():
    for x_test, y_test in test_dataloader:
        test_pred = m0(x_test)
        loss += lf(test_pred, y_test)
        test_acc += accuracy_fn(y_true=y_test, pred=test_pred.argmax(dim=1))
        # print(accuracy_fn(y_test, test_pred.argmax(dim=1)))
        loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
        # print(f"\n Train loss:{train_loss:.4f} | loss:{loss:.4f}, Test acc:{test_acc} ")
        train_time_end_on_cpu = timer()
        total_train_time_m0 = print_train_time(start=train_time_start_on_cpu,
                                               end= train_time_end_on_cpu,
                                               device = str(next(m0.parameters())))
        next(m0.parameters()).device
        # print(next(m0.parameters()).device)

torch.manual_seed(42)
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn):
    loss,acc =0,0 
    model.eval()
    with torch.inference_mode():
        for x,y in data_loader:
            y_pred =model(x)
            loss_fn = nn.CrossEntropyLoss()
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, pred=y_pred.argmax(dim=1))
        loss /= len(data_loader)
        acc /= len(data_loader)
    return{"model_name": model.__class__.__name__,
           "model_loss": loss.item(),
           "model_acc": acc}
model_0_results = eval_model(model=m0,
                             data_loader=test_dataloader,
                             loss_fn=loss,
                             accuracy_fn=accuracy_fn)
# model_0_results



class FashionMNISTModelv1(nn.Module):
    def __init__(self,
                 input_shapes: int,
                 hidden_units: int,
                 output_shapes: int):
       super().__init__()
       self.layer_stack = nn.Sequential(
           nn.Flatten(),
           nn.Linear(in_features=input_shapes,
                     out_features=hidden_units),
           nn.ReLU(),
           nn.Linear(in_features=hidden_units,
                     out_features=output_shapes),
           nn.ReLU()

    )
    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)
    
torch.manual_seed(42)
m1 = FashionMNISTModelv1(input_shapes=784,
                         hidden_units=10,
                         output_shapes=len(class_name)).to("cpu")
# print(next(m0.parameters()).device)


device ="cuda" if torch.cuda.is_available() else "cpu"

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.dataloader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = None):
    train_loss, train_acc = 0,0
    model.train()

    for batch, (x,y) in enumerate(train_dataloader):
        x,y = x.to(device), y.to(device)
        pred_y = model(x)
        loss = lf(pred_y, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y, pred=pred_y.argmax(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    # print(f"Train loss: {train_loss:.5f} | Train_acc: {train_acc}")

    
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
    # print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc}%n")


    torch.manual_seed(42)
train_time_start_on_cpu = timer()
epochs = 3
for epoch in tqdm(range(epochs)):
    # print(f"epoch:{epoch}\n----")
    train_step(model=m1,
               data_loader=train_dataloader,
               loss_fn=lf,
               optimizer=optim,
               accuracy_fn=accuracy_fn,
               device=device)
    test_step(model=m1,
               data_loader=train_dataloader,
               loss_fn=lf,
               optimizer=optim,
               accuracy_fn=accuracy_fn,
               device=device)
    train_time_end_on_cpu=timer()
    total_train_time_m1 = print_train_time(start=train_time_start_on_cpu,
                                           end=train_time_end_on_cpu,
                                           device=device)
model_0_results


m1_results = eval_model(model=m1,
                        data_loader=test_dataloader,
                        loss_fn=lf,
                        accuracy_fn=accuracy_fn,
                        )
m1_results



