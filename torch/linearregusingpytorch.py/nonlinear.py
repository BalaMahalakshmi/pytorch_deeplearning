import torch
from torch import nn
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split


weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.01
x = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight* x + bias
# print(len(x))
# print(x[:5], y[:5])

train_split = int(0.8 * len(x))
x_train , y_train = x[: train_split], y[: train_split]
x_test, y_test = x[train_split :], y[train_split :]
# print(len(x_train), len(y_train))
# print(len(x_test), len(y_test))

matplotlib.use("TkAgg")
plt.ion()
plt.close('all')

def plot_predictions(train_data = x_train, train_laels = y_train, test_data = x_test, test_labels= y_test, pred=None):
    plt.figure(figsize=(8,8))
    plt.scatter(train_data, train_laels, c='b', s=4, labels="training data")
    plt.scatter(test_data, test_labels, c='g', s=4, labels="testing data")
    plt.show()
    if pred is not None:
        plt.scatter(test_data, pred, c='r', s=4, labels="predictions")
    plot_predictions(x_train,y_train,x_test,y_test)

#recreating non linear data
n_samples = 1000
x,y = make_circles(n_samples, noise=0.03, random_state=42)
plt.scatter(x[:,0], x[:,1], c=y, cmap = plt.cm.RdBu)
# plt.show()

x = torch.from_numpy(x).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
x1,x2,y1,y2 = train_test_split(x, y, test_size=0.2, random_state=42)
# print(x1[:5], y1[:5])
device ="cuda" if torch.cuda.is_available() else "cpu"


def accuracy_fn(y_true, pred):
    correct = torch.eq(y_true, pred).sum().item()
    acc = (correct/len(pred))*100
    # print(acc)

class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()

        #create nn.layer cabable of handling the shapes of our data
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU() #non linear activation function
    def forward(self, x):
        return self.layer_3(self.relu((self.layer_2(self.layer_1(x)))))
    
m3 = CircleModelV2().to(device)
# print(m3)
lf = nn.BCEWithLogitsLoss()
optim = torch.optim.SGD(params=m3.parameters(), lr=0.1)
# print(optim)


torch.manual_seed(42)
torch.cuda.manual_seed(42)
x1,y1 = x1.to(device), y1.to(device)
x2,y2 = x2.to(device), y2.to(device)

epochs = 1000
m3.train()
y_logits = m3(x1).squeeze()
preds = torch.round(torch.sigmoid(y_logits))
# print("prediction:",preds)

    #calculate loss/accuracy
loss = lf(y_logits, y1)
# print(lf, loss)
acc = accuracy_fn(y_true=y1, pred=preds)
# print("accuracy:", acc)

#oprimizer zero grad
optim.zero_grad()

loss.backward()
optim.step()
# print(m3.state_dict())

#testing
m3.eval()
with torch.inference_mode():
    pred = torch.round(torch.sigmoid(m3(x2))).squeeze()
    # print(pred[:10], y2[:10])
    tl = m3(x2).squeeze()
    tp = torch.round(torch.sigmoid(tl))
    # print("test_loss:",tl)
    # print("test_pred:",tp)

    test_loss = lf(tl, y2)
    # print("loss:",test_loss)
    test_acc = accuracy_fn(y_true=y2, pred = tp)
    # print("accuracy:", test_acc)

# if epochs % 100 == 0:
#     print(f"epoch: {epochs} | loss:{loss.item():.4f}, accuracy:{acc:.2f}% | test loss:{test_loss.item():.4f}, test acc:{test_acc:2f}")

# from helper_functions import plot_predictions, plot_decision_boundary

# plt.figure(figsize=(10,8))
# plt.subplot(1,2,1)
# plt.title("Train")
# plot_decision_boundary(m3, x1, y1)
# plt.subplot(1,2,2)
# plt.title("Test")
# plot_decision_boundary(m3, x2, y2)
# plt.show()

#creating tensor
a = torch.arange(-10,10,1, dtype=torch.float32)
# print(a.dtype, a)
plt.plot(a)
plt.plot(torch.relu(a))
# plt.show()

def relu(x: torch.Tensor):
    return torch.maximum(torch.tensor(0), x)
# print(relu(a))

plt.plot(relu(a));
plt.show()

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))
plt.plot(torch.sigmoid(a));

