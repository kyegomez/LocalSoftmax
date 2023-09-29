[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# LocalSoftmax
Local Softmax parallelize the softmax computation by splitting the tensor into smaller sub-tensors and applying the softmax function on each of these smaller tensors independently. In other words, we want to compute a "local" softmax on each chunk of the tensor, instead of on the entire tensor.

# Appreciation
* Lucidrains
* Agorians



# Install
`pip install local-sftmx`


# Algorithm
function LocalSoftmax(tensor, num_chunks):
    split tensors into `num_chunks` smaller tensors
    for each smaller tensor:
        apply standard softmax
    concatenate the results
    return concatenated tensor

# License
MIT

