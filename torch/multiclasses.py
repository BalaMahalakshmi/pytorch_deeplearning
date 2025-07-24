import torch
from  matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

num_classes=4
num_features=2
random_seed = 42
#create multi class data
x_blob, y_blob = make_blobs(n_samples=1000, n_features=num_features, centers=num_classes, center_std=1.5, random_state=random_seed)

#turn data in tensors
x_blob = torch.from_numpy(x_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.float)

#split data
x_blob1, x_blob2, y_blob1, y_blob2 = train_test_split(x_blob,y_blob, test_size=0.2, random_seed=random_seed)

#plot
plt.figure(figsize=(10,7))
plt.scatter(x_blob[:,0], x_blob[:, 1], c = y_blob, cmap = plt.cm.RdYlBu)
plt.show()