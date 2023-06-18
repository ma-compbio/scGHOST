import os
import torch
import numpy as np
import pickle
import gc
import tqdm

from tqdm import trange
from modules.preprocessing import parse_chromosomes
from torch import nn
from tqdm import trange
from torch.nn import functional as F

num_cells = 500

def to_cuda(x):
    
    if torch.cuda.is_available():
        return x.cuda()
    
    return x

class hubs(nn.Module):
    def __init__(self,N):
        super(hubs,self).__init__()
        self.N = N
        
        self.embedding = nn.Sequential(
            nn.Linear(self.N,256), nn.ReLU(),
            nn.Linear(256,128),
        )
        
        to_cuda(self)
        
    def to_one_hot(self,x):
        return F.one_hot(x,num_classes=self.N).float()
    
    def embed(self,x):
        return self.embedding(self.to_one_hot(x))
    
def embed_single_cells(pairs_labels_file,oems_file,embedding_file,cell_nums,batch_size=64):
    
    pairs_labels = pickle.load(open(pairs_labels_file,'rb'))
    all_continuous_pairs = pairs_labels['pairs']
    all_continuous_labels = pairs_labels['labels']
    
    OEMs = np.load(oems_file)
    bs = batch_size
    
    cell_nums = np.arange(len(OEMs)) if cell_nums is None else cell_nums
    
    criterion = nn.MSELoss()
    
    all_Es = []
    
    for ii,cnum in tqdm(enumerate(cell_nums),total=len(cell_nums)):
        
        model = hubs(len(OEMs[ii]))
        optimizer = torch.optim.Adam(model.parameters())
        
        cell_pairs = to_cuda(all_continuous_pairs[cnum])
        cell_labels = to_cuda(all_continuous_labels[cnum])
        
        for epoch in range(1):
            random_shuffle = np.random.permutation(len(all_continuous_pairs[cnum]))
            
            rloss = 0
            rcntl = 0
            rpcal = 0
            rsamples = 0
            
            for i in range(0,len(cell_pairs),bs):

                idx = random_shuffle[i:i+bs]
                blen = len(idx)
                
                x = model.embed(cell_pairs[idx])
                x1 = x[:,0]
                x2 = x[:,1]
                y = cell_labels[idx]
                
                sim = nn.CosineSimilarity()(x1,x2)
                
                optimizer.zero_grad()
                contact_loss = criterion(sim,y)
                loss = contact_loss

                loss.backward()
                optimizer.step()
                
                rloss += float(loss) * blen
                rcntl += float(contact_loss) * blen
                rsamples += blen
                
        E = torch.zeros(len(OEMs[cnum]),128)
        
        for i in range(0,len(OEMs[cnum]),bs):
            end = np.min([i+bs,len(OEMs[cnum])])
            e = model.embed(to_cuda(torch.arange(i,end)))
            E[i:i+len(e)] = e
        
        E = E.detach().cpu().numpy()
        all_Es.append(E)
        
        torch.cuda.empty_cache()

    all_Es = np.array(all_Es)
    np.save(embedding_file, all_Es)
    gc.collect()


def graph_embedding(runtime_args):

    chromosomes = parse_chromosomes(runtime_args['chromosomes'])

    for chrn in chromosomes:
    
        sample_path = os.path.join(runtime_args['data_directory'],'chr%d_calibrated_samples.pkl' % chrn)
        oe_path = os.path.join(runtime_args['data_directory'],'chr%d_oe.npy' % chrn)
        out_path = os.path.join(runtime_args['data_directory'],'chr%d_embeddings.npy' % chrn)

        # params: (pairs_labels_file,oems_file,embedding_file,cell_nums,batch_size=64)
        embed_single_cells(
            sample_path,
            oe_path,
            out_path,
            num_cells # specify cell_nums to limit the dataset size, not implemented modularly
        )
        
        torch.cuda.empty_cache()
