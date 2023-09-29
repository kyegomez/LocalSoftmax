import torch 
import torch.nn.functional as F

def local_softmax(tensor, num_chunks):
    #split the tensor into num chunks smaller tensor
    tensors = torch.chunk(tensor, num_chunks, dim=0)

    #apply softmax on each chunk and collect the results in a list
    results = [
        F.softmax(t, dim=0) for t in tensors
    ]

    #concat results
    concated_results = torch.cat(results, dim=0)

    return concated_results


def fast_softmax(tensor):
    shiftx = tensor - torch.max(tensor)

    exps = torch.exp(shiftx)

    return exps / torch.sum(exps)

tensor = torch.rand(10, 5)
result = fast_softmax(tensor)