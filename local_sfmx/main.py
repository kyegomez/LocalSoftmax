import timeit

import torch
import torch.nn.functional as F


def standard_softmax(tensor):
    return F.softmax(tensor, dim=0)

def local_softmax(tensor, num_chunks: int = 2):
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
    """
    LogSumExp trick for numerical stability

    tensor = torch.rand(10, 5)
    result = fast_softmax(tensor)
    print(result)
    
    """
    shiftx = tensor - torch.max(tensor)

    exps = torch.exp(shiftx)

    return exps / torch.sum(exps)



def sparse_softmax(z, k: int = 3):
    _, top_k_indices = z.topk(k, dim=0)
    omega_k = top_k_indices

    #compute sparse softmax transformation
    exp_z = torch.exp(z)
    masked_sum_exp = exp_z[omega_k].sum()
    values = torch.zeros_like(z)
    values[omega_k] = exp_z[omega_k] / masked_sum_exp

    return values


tensor = torch.randn(10, 5)
result = sparse_softmax(tensor, k=3)
print(f'result sparse softmax: {result}')


# Benchmark function
def benchmark(func, tensor, num_iterations=10000):
    timer = timeit.Timer(lambda: func(tensor))
    time_taken = timer.timeit(num_iterations)
    return time_taken

tensor = torch.randn(1000)  # Random tensor of size 1000

# Benchmarking
num_iterations = 10000

std_time = benchmark(fast_softmax, tensor, num_iterations)
fast_time = benchmark(local_softmax, tensor, num_iterations)

print(f"Standard Softmax: {std_time:.5f} seconds for {num_iterations} iterations")
print(f"Fast Softmax: {fast_time:.5f} seconds for {num_iterations} iterations")
