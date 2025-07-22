import numpy as np
import torch
a= np.arange(1.,10.)
t = torch.from_numpy(a)
# print(t)

#change values to a
a += 1
# print(a)

#tensor to numpy array
t = torch.ones(8)
nt = t.numpy()
# print(nt)

#pytorch reproducibility
a = torch.rand(2,4)
b = torch.rand(2,4)
# print(a == b)

#using random seed
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
c = torch.rand(4,3)
torch.manual_seed(RANDOM_SEED)
d = torch.rand(4,3)
# print(c == d)

import torch
print(torch.cuda.is_available())

