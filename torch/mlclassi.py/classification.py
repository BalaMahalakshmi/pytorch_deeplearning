import sklearn
from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import requests
from pathlib import Path

n_sample = 1000
x,y = make_circles(n_sample, noise=0.03, random_state=42)
# print(len(x),len(y))

# print(f"first 5 samples of x:\n{x[:5]}")
# print(f"first 5 samples of y:\n{y[:5]}")
# print(y)

c = pd.DataFrame({'x1':x[:,0],'x2':x[:,1],'label':y})
# print(c.head(10))
plt.scatter(x=x[:,0],y=x[:,1],c=y,cmap=plt.cm.RdYlBu)
# plt.show()

# print(x.shape, y.shape)
# print(x)

#viewing labels and features
x_sam = x[0]
y_sam = y[0]
# print(f'values for sample f x:{x_sam} and y:{y_sam}')
# print(f"shapes for sample of x:{x_sam.shape} and y:{y_sam.shape}")

#turn data into tensors
# print(torch.__version__)

# print(type(x), x.dtype)
x = torch.from_numpy(x).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
# print(x[:5], y[:5])

#split data into training and test sets
x1,x2,y1,y2 = train_test_split(x, y, test_size=0.2, random_state=42)
# print(len(x1), len(x2), len(y1), len(y2))

device ="cuda" if torch.cuda.is_available() else "cpu"
# print(device)

#construct a model 
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()

        #create nn.layer cabable of handling the shapes of our data
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)
    def forward(self, x):
        return self.layer_2(self.layer_1(x))
m0 = CircleModelV0().to(device)
# print(m0)

#using torch.nn.sequential()

m0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)).to(device)
# print(m1)
# print(m0.state_dict())

#make predictions
untrained_pred = m0(x.to(device))
# print(f"length of predictions:{len(untrained_pred)}, shape:{untrained_pred.shape}")
# print(f"length of test samples: {len(x)}, shape:{x.shape}")
# print(f"\n First 10 predictions :\n(torch.round{untrained_pred[:10])}")
# print(f"\n first 10 labels: \n{y[:10]}")

#setup the loss function
# lf  = nn.BCELoss() #through sigmoid fun.
lf = nn.BCEWithLogitsLoss()
optim = torch.optim.SGD(params=m0.parameters(), lr=0.1)
# print(optim)

#calculate accuracy

def accuracy_fn(y_true, pred):
    correct = torch.eq(y_true, pred).sum().item()
    acc = (correct/len(pred))*100
    # print(acc)

m0.eval()
with torch.inference_mode():
    y_logits = m0(x.to(device))[:5]
# print(y_logits)

# print(y)[:5]

#use sigmoid function
y_pred_probs = torch.sigmoid(y_logits)
# print(y_pred_probs)

#find the predicted labels
preds =torch.round(y_pred_probs)
#in full
pred_labels = torch.round(torch.sigmoid(m0(x.to(device))[:5]))
#check for equality
# print(torch.eq(preds.squeeze(), pred_labels.squeeze()))
#get rid of extra dim.
# print(preds.squeeze())

#building training and test model

torch.manual_seed(42)
torch.cuda.manual_seed(42)

#set epochs
epochs=100

#put data to target device
x1,y1 = x1.to(device), y1.to(device)
x2,y2 = x2.to(device), y2.to(device)

#build training and evaluation loop
for epoch in range(epochs):
    #training 
    m0.train()

    #forward pass
    y_logits = m0(x1).squeeze()
    preds = torch.round(torch.sigmoid(y_logits))
    # print("prediction:",preds)

    #calculate loss/accuracy
    loss = lf(torch.sigmoid(y_logits), y1 )
    # print(lf, loss)
    acc = accuracy_fn(y_true=y1, pred=preds)
    # print("accuracy:", acc)

#oprimizer zero grad
optim.zero_grad()

loss.backward()
optim.step()

#testing
m0.eval()
with torch.inference_mode():
    tl = m0(x2).squeeze()
    tp = torch.round(torch.sigmoid(tl))
    # print("test_loss:",tl)
    # print("test_pred:",tp)

    test_loss = lf(tl, y2)
    # print("loss:",test_loss)
    test_acc = accuracy_fn(y_true=y2, pred = tp)
    # print("accuracy:", test_acc)

if epoch % 10 == 0:
    print(f"epoch: {epoch} | loss:{lf:.5}, accuracy:{acc:.2} % | test loss:{test_loss:.5}, test acc:{test_acc:2f} %")




if Path ("helper_functions.py").is_file():
    print("helper_functions.py already exits, skipping download")
else:
    print("download helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(requests.content)

# from helper_functions import plot_predictions, plot_decision_boundary

# plt.figure(figsize=(10,8))
# plt.subplot(1,2,1)
# plt.title("Train")
# plot_decision_boundary(m0, x1, y1)
# plt.subplot(1,2,2)
# plt.title("Test")
# plot_decision_boundary(m0, x2, y2)
# plt.show()


#create a model
class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()

        #create nn.layer cabable of handling the shapes of our data
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
    def forward(self, x):
        return self.layer_3(self.layer_2(self.layer_1(x)))
m1 = CircleModelV1().to(device)
# print(m1)
# print(m1.state_dict())

#create a loss function
lf = nn.BCEWithLogitsLoss()
optim = torch.SGD(params=m1.parameters(),lr=0.1)
print(lf, optim)

#write a training and evaluation model
torch.manual_seed(42)
torch.cuda.manual_seed(42)

#train for longer
epochs=1000
x1,y1 = x1.to(device), y1.to(device)
x2,y2 = x2.to(device), y2.to(device)

for epoch in range(epochs):
    m1.train()
    y_logits = m1(x1).squeeze()
    preds = torch.round(torch.sigmoid(y_logits))
    print("prediction:",preds)

    #calculate loss/accuracy
    loss = lf(torch.sigmoid(y_logits), y1 )
    print(lf, loss)
    acc = accuracy_fn(y_true=y1, pred=preds)
    print("accuracy:", acc)

#oprimizer zero grad
optim.zero_grad()

loss.backward()
optim.step()

#testing
m1.eval()
with torch.inference_mode():
    tl = m1(x2).squeeze()
    tp = torch.round(torch.sigmoid(tl))
    print("test_loss:",tl)
    print("test_pred:",tp)

    test_loss = lf(tl, y2)
    print("loss:",test_loss)
    test_acc = accuracy_fn(y_true=y2, pred = tp)
    print("accuracy:", test_acc)

if epoch % 100 == 0:
    print(f"epoch: {epoch} | loss:{lf:.5f}, accuracy:{acc:.2f} % | test loss:{test_loss:.5f}, test acc:{test_acc:2f} %")


