import sklearn
from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.model_selection import train_test_split

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
lf = nn.BCEWithLogitsLoss




