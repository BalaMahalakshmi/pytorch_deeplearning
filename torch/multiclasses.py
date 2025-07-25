import torch
from  matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from torch import nn
from torchmetrics import Accuracy

num_classes=4
num_features=2
random_seed = 42
#create multi class data
x_blob, y_blob = make_blobs(n_samples=1000, n_features=num_features, centers=num_classes, cluster_std=1.5, random_state=random_seed)

#turn data in tensors
x_blob = torch.from_numpy(x_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.float)

#split data
x_blob1, x_blob2, y_blob1, y_blob2 = train_test_split(x_blob,y_blob, test_size=0.2, random_state=random_seed)

#plot
plt.figure(figsize=(10,7))
plt.scatter(x_blob[:,0], x_blob[:, 1], c = y_blob, cmap = plt.cm.RdYlBu)
# plt.show()


device ="cuda" if torch.cuda.is_available() else "cpu"

def accuracy_fn(y_true, pred):
    correct = torch.eq(y_true, pred).sum().item()
    acc = (correct/len(pred))*100
    # print(acc)


#multiclas classification model
class Blobmodel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )
    def forward(self, x):
        return self.linear_layer_stack(x)
# print(x_blob.shape, y_blob.shape[:5])

m4 = Blobmodel(input_features=2,
               output_features=4,
               hidden_units=8).to(device)
# print(m4)
# print(x_blob1.shape, y_blob1.shape[:5])
# print(torch.unique(y_blob1))

#create a loss function for multiclass classification
loss_fn = nn.CrossEntropyLoss()
# print("loss function:", loss_fn)
#create optimizer
optim = torch.optim.SGD(params=m4.parameters(), lr=0.1)
# print("optimizer:", optim)

#getting pred probabilities
# print(m4(x_blob2))
# print(next(m4.parameters()).device)

#get some raw output for model (logits)
m4.eval()
with torch.inference_mode():
    preds = m4(x_blob2.to(device))
    # print(preds[:10])

#convert our models logit outputs to pred probabilities
pred_probs = torch.softmax(preds, dim=1)
# print(preds[:5])
# print(pred_probs[:5])

# print(torch.sum(pred_probs[0]))
# print(torch.argmax(pred_probs[0]))

pred = torch.argmax(pred_probs, dim=1) #prediction labels
# print(pred)
# print(y_blob2)

#fit the multi class model to data
torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 100

x_blob1, y_blob1 = x_blob1.to(device), y_blob1.to(device)
x_blob2, y_blob2 = x_blob2.to(device), y_blob2.to(device)

for epoch in range(epochs):
    m4.train()

    logits=m4(x_blob1)
    y_pred = torch.softmax(logits, dim=1).argmax(dim=1)

    loss =loss_fn(logits, y_blob1.long())
    acc = accuracy_fn(y_true = y_blob1,
                      pred = pred)
    optim.zero_grad()
    loss.backward()
    optim.step()

    m4.eval()
    with torch.inference_mode():
        tl = m4(x_blob2)
        tp = torch.softmax(tl, dim=1).argmax(dim=1)
        te_loss = loss_fn(tl, y_blob2.long())
        # print("test loss:", te_loss)
        te_acc =accuracy_fn(y_true=y_blob2, pred=tp)
        # print("test acc:", te_acc)

if epochs % 10 == 0:
    # print(f"epoch:{epochs} | loss;{loss:.4f}, acc:{acc:.2f} % | test loss:{te_loss:.4f}, test acc:{te_acc:.2f}%")



#make predictions

 m4.eval()
 with torch.inference_mode():
     y_logits = m4(x_blob2)
# print(x_blob2[:10])

y_pred_probs =torch.softmax(y_logits, dim=1)
# print(y_pred_probs[:10])
# print(y_blob2)


from helper_functions import plot_predictions, plot_decision_boundary
plt.figure(figsize=(10,8))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(m4, x_blob1, y_blob1)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(m4, x_blob2, y_blob2)
# plt.show()

#accuracy 

tm = Accuracy()
# print(tm(y_pred, y_blob2))
