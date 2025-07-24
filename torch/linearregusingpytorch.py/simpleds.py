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
# print(y)

# print(x[:10], y[:10], len(x), len(y))
# print(len(x), len(y))


#splitting data into training and test

train_split = int(0.8 * len(x))
x_train , y_train = x[: train_split], y[: train_split]
x_test, y_test = x[train_split :], y[train_split :]
# print(len(x_train), len(y_train))
# print(len(x_test),len(y_test))

#visualize our data

def plot_predictions(train_data = x_train, train_laels = y_train, test_data = x_test, test_labels= y_test, pred=None):
    plt.figure(figsize=(8,8))
    plt.scatter(train_data, train_laels, c='b', s=4, labels="training data")
    plt.scatter(test_data, test_labels, c='g', s=4, labels="testing data")
    if pred is not None:
        plt.scatter(test_data, pred, c='r', s=4, labels="predictions")
plt.legend(prop={'size:12'})
# plt.show()
# plot_predictions()
 

