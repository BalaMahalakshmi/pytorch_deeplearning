import torch
from torch import nn
from matplotlib import pyplot as plt
import numpy as np


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
    # plt.figure(figsize=(8,8))
    # plt.scatter(train_data, train_laels, c='b', s=4, labels="training data")
    # plt.scatter(test_data, test_labels, c='g', s=4, labels="testing data")
    # if pred is not None:
    #     plt.scatter(test_data, pred, c='r', s=4, labels="predictions")
    plot_predictions(x_train,y_train,x_test,y_test)


class LinearRegressionModel(nn.Module):
    def __init__(self):
        weights=0.7
        bias=0.3
        super(). __init__()
        self.linear = torch.nn.Linear(1,1)
        self.weights = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
    def forward(self, x: torch.Tensor):
        return self.linear(x)
        return self.weights * self.bias


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
# print(lf)

# #set optimizer
params = m.parameters()
opt = torch.optim.SGD(params=m.parameters(), lr=0.01)
# print(opt)


#training loop intution 

epochs=100

epoch_count=[]
loss_values=[]
test_loss_values=[]

for epoch in range (epochs):
    m.train()
    preds = m(x_train)
    # print(preds)
    loss = lf(preds,y_train)
    # print(loss)
    opt.zero_grad()
    loss.backward()
    opt.step()
    m.eval()
    with torch.inference_mode():
        tp =m(x_test)
        # print(tp)
    # m.state_dict()
        tl = lf(tp,y_test)
if epoch % 10 == 0:
    epoch_count.append(epoch)
    loss_values.append(loss)
    test_loss_values.append(tl)

    # print(f"epoch:{epoch} | loss: {loss} | test_loss:{tl}")
    pn = m(x_test)
    # print(pn)


    # np.arry(torch.tensor(loss_values).numpy()), test_loss_values
    # plt.plot(epoch_count, np.array(torch.tensor(loss_values).numpy()), label='train loss')
    # plt.plot(epoch_count, test_loss_values, label='test loss')
    # plt.title("training and test loss curves")
    # plt.xlabel("epochs")
    # plt.ylabel("loss")
    # plt.legend()
    # plt.show()


#saving our pytorch model
# from pathlib import Path

# mp = Path("models")
# mp.mkdir(parents=True, exist_ok=True)

# #create model
# mn = "pytorch_workflow_model.path"
# msp = mp / mn
# # print(msp)

#save the model
# print(f"saving model to:{msp}")
# torch.save(m.state_dict(), f = msp)
# print(f"model saved to: {msp}")
# print( !ls -l models)

#loading a pytorch
# print(m.state_dict())
loaded_m = LinearRegressionModel()
# print(loaded_m.load_state_dict(torch.load(f=msp)))
# print(m.state_dict())

#making some predictions 
loaded_m.eval()
with torch.inference_mode():
    loaded_m_preds = loaded_m(x_test)
# print(loaded_m_preds)

# print(preds == loaded_m_preds)

#building a linear model

class LinearRegressionModelv2(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_layer = nn.Linear(in_features=1, out_features=1)
    def forward(self, x: torch.tensor):
        return self.linear_layer(x)
torch.manual_seed(42)
m1 =LinearRegressionModelv2()
print(m1, m1.state_dict())