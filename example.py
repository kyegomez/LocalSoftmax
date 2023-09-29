import torch
from local_sfmx import local_softmax

tensor = torch.rand(10, 5)
result = local_softmax(tensor, 2)
print(result)