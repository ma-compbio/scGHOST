import torch

def to_cuda(x):
    
    if torch.cuda.is_available():
        return x.cuda()
    
    return x