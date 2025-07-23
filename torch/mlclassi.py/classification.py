import sklearn
from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt

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