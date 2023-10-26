import numpy as np
import torch

def to_cuda(x):
    
    if torch.cuda.is_available():
        return x.cuda()
    
    return x

def get_expected(M,eps=1e-8):
    E = np.zeros_like(M)
    l = len(M)

    for i in range(M.shape[0]):
        contacts = np.diag(M,i)
        expected = contacts.sum() / (l-i)
        # expected = np.median(contacts)
        x_diag,y_diag = np.diag_indices(M.shape[0]-i)
        x,y = x_diag,y_diag+i
        E[x,y] = expected

    E += E.T
    E = np.nan_to_num(E) + eps
    
    return E
    
def get_oe_matrix(M):
    E = get_expected(M)
    oe = np.nan_to_num(M / E)
    np.fill_diagonal(oe,1)
    
    return oe

def random_sample(p, size, neg=False, normed=False):

    if not normed:
        p_ = p / torch.sum(p, dim=-1, keepdim=True)
    else:
        p_ = p
    
    # rg = np.random.default_rng()

    random_num = torch.rand(p_.shape,device=p.device)
    # random_num /= torch.sum(random_num, dim=-1, keepdim=True)
    
    diff = random_num - p_

    # k = size
    sampled_weights,sampled_idx = torch.topk(diff, size, dim=-1, largest=neg)
    sampled_weights = sampled_weights[..., :size] if not neg else sampled_weights[..., -size:]
    sampled_idx = sampled_idx[..., :size] if not neg else sampled_idx[..., -size:]
    
    return sampled_weights,sampled_idx

def random_sample_sorted(p, size, top=None, neg=False, normed=False):
    
    if top is not None:
        p = p[...,-top:] if not neg else p[...,:top]
        

    if not normed:
        p_ = p / torch.sum(p, dim=-1, keepdim=True)
    else:
        p_ = p
    
    # rg = np.random.default_rng()

    random_num = torch.rand(p_.shape,device=p.device)
    # random_num /= torch.sum(random_num, dim=-1, keepdim=True)

    diff = random_num - p_

    # k = size
    sampled_weights,sampled_idx = torch.topk(diff, size, dim=-1, largest=neg)
    sampled_weights = sampled_weights[..., :size]# if not neg else sampled_weights[..., -size:]
    sampled_idx = sampled_idx[..., :size]# if not neg else sampled_idx[..., -size:]
    
    return sampled_weights,sampled_idx

def random_sample_np(p, size, normed=False):
    if not normed:
        p_ = p / np.sum(p, axis=-1, keepdims=True)
    else:
        p_ = p
    rg = np.random.default_rng()
    random_num = rg.random(p_.shape)
    random_num /= np.sum(random_num, axis=1, keepdims=True)
    diff = random_num - p_

    # k = size
    sampled_idx = np.argpartition(diff, size, axis=1)[..., :size]
    return sampled_idx