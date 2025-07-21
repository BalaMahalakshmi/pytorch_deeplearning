import torch
# print(torch.__version__)

#scalar
abi = torch.tensor(8)
# print(abi)

# print(abi.ndim)
# print(abi.item())

#vector
vector = torch.tensor([16,16])
# print(vector)
# print(vector.ndim)
# print(vector.shape)

#MATRIX

matrix = torch.tensor([[5,6],[7,8]])
# print(matrix)
# print(matrix.ndim)
# print(matrix.shape)
# print(matrix[1])

#TENSOR

ten = torch.tensor([[1,2,3],[5,6,7],[8,9,0]])
# print(ten)
# print(ten.ndim)
# print(ten.shape)

#random numbers

rt = torch.rand(3,4)
# print(rt)

#with size

rt_size = torch.rand(size=(4,2,3))
# print(rt_size)

#zeros and ones
a = torch.zeros(size=(2,4))
# print(a)

b = torch.ones(size=(3,3))
# print(b)
# print(rt.dtype)

#creating using range => o/p is 0.,1.,2.,.....

a = torch.range(0,10)
# print(a)

#creating using arange => o/p is 0,1,2,.....

b = torch.arange(0,10)
# print(b)

b = torch.arange(start=0,end=100,step=16)
# print(b)

#using like
t  = torch.zeros_like(input=b)
# print(t)

#tensor datatypes

float_32_tensor = torch.tensor([3.0,6.8,8.16], dtype=torch.int)
# print(f)

float_16_tensor = torch.tensor([3.0,6.8,8.16], dtype=None, device='cpu', requires_grad=False)
# print(float_16_tensor )

float_16_tensor = float_32_tensor.type(torch.float16)
# print(float_16_tensor * float_32_tensor )

#tensor attributes

ten = torch.rand(2,2)
# print(ten)
# print(f"dtype of tensor:{ten.dtype}")
# print(f"shape of tensor:{ten.shape}")
# print(f"run device is on:{ten.device}")

#manipulating tensors

t = torch.tensor([5,8,6])
# print(t+10)
# print(t*20)
# print(t/2)
# print(t-30)
# print(t)

