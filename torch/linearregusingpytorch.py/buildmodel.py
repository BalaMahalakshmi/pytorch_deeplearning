import torch
from torch import nn
from matplotlib import pyplot as plt


weight = 0.8
bias = 0.5
start = 0
end = 1
step = 0.02
x = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight* x + bias 

train_split = int(0.8 * len(x))
x_train , y_train = x[: train_split], y[: train_split]
x_test, y_test = x[train_split :], y[train_split :]

def plot_predictions(train_data = x_train, train_laels = y_train, test_data = x_test, test_labels= y_test, pred=None):
    plt.figure(figsize=(8,8))
    plt.scatter(train_data, train_laels, c='b', s=4, labels="training data")
    plt.scatter(test_data, test_labels, c='g', s=4, labels="testing data")
    if pred is not None:
        plt.scatter(test_data, pred, c='r', s=4, labels="predictions")


class LinearRegressionModel(nn.Module):
    def __init__(self):
        weights=0.7
        bias=0.3
        super(). __init__()
        self.weights = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
    def forward(self, x: torch.Tensor):
        print(self.weights * self.bias)


#check model parameter
a = torch.rand(1)
torch.manual_seed(42)
m = LinearRegressionModel()
# print(m)
# print(m.parameters())
# print(m.state_dict())

#making predictions

# print(x_test, y_test)

with torch.inference_mode():
    pred = m(x_test)
    # print("predictions:", pred)

with torch.no_grad():
    p2 = m(x_test)
    # print(p2)

    # plot_predictions(pred)   

#set a loss function

lf = nn.L1Loss()
print(lf)

# #set optimizer
params = m.parameters()
opt = torch.optim.SGD(params=m.parameters(), lr=0.01)
print(opt)
