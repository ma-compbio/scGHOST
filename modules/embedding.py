import os
import torch
import numpy as np
import pickle
import gc

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
            # nn.ReLU()
            nn.Tanh(),
        )
        
        self.batch_norm = nn.BatchNorm1d(2)
        
        to_cuda(self)
        
    def to_one_hot(self,x):
        return F.one_hot(x,num_classes=self.N).float()
    
    def embed(self,x):
        return self.embedding(self.to_one_hot(x))
    
    
def embed_single_cells(pairs_labels_file,oems_file,embedding_file,num_cells,batch_size=64):
    models = []
    optimizers = []
    
    pairs_labels = pickle.load(open(pairs_labels_file,'rb'))
    all_continuous_pairs = pairs_labels['pairs']
    all_continuous_labels = pairs_labels['labels']
    
    OEMs = np.load(oems_file)
    bs = batch_size
    
    for i in range(num_cells):
        models.append(hubs(len(OEMs[i])))
        optimizers.append(torch.optim.Adam(models[i].parameters()))
        
    criterion = nn.MSELoss()
    criterion_pca = nn.MSELoss()
    criterion_clf = nn.CrossEntropyLoss() 
    
    for cnum in trange(num_cells):
        for epoch in range(1):
            random_shuffle = np.random.permutation(len(all_continuous_pairs[cnum]))
            
            rloss = 0
            rcntl = 0
            rpcal = 0
            rsamples = 0
            
            for i in range(0,len(all_continuous_pairs[cnum]),bs):

                idx = random_shuffle[i:i+bs]
                blen = len(idx)
                
                x = models[cnum].embed(all_continuous_pairs[cnum][idx])
                x1 = x[:,0]
                x2 = x[:,1]
                y = all_continuous_labels[cnum][idx]
                
                sim = nn.CosineSimilarity()(x1,x2)
                
                optimizers[cnum].zero_grad()
                contact_loss = criterion(sim,y)
                loss = contact_loss

                # print(float(loss))
                loss.backward()
                optimizers[cnum].step()
                
                rloss += float(loss) * blen
                rcntl += float(contact_loss) * blen
                rsamples += blen
                
                """print('Cell %d epoch %d: %d/%d -- loss: %.6f' % (
                    cnum + 1,epoch + 1,rsamples,len(all_continuous_pairs[cnum]),rloss / rsamples
                ),end='\r')"""
                
            # print()
            
    all_Es = []
    for cnum in range(num_cells):
        E = torch.zeros(len(OEMs[cnum]),128)
        
        bs = 32
        for i in range(0,len(OEMs[cnum]),bs):
            end = np.min([i+bs,len(OEMs[cnum])])
            e = models[cnum].embed(to_cuda(torch.arange(i,end)))
            E[i:i+len(e)] = e

        E = E.detach().cpu().numpy()
        all_Es.append(E)

    all_Es = np.array(all_Es)
    np.save(embedding_file, all_Es)
    gc.collect()


def graph_embedding(runtime_args):

    chromosomes = parse_chromosomes(runtime_args['chromosomes'])

    for chrn in chromosomes:
    
        sample_path = os.path.join(runtime_args['data_directory'],'chr%d_calibrated_samples.pkl' % chrn)
        oe_path = os.path.join(runtime_args['data_directory'],'chr%d_oe.npy' % chrn)
        out_path = os.path.join(runtime_args['data_directory'],'chr%d_embeddings' % chrn)

        embed_single_cells(
            sample_path,
            oe_path,
            out_path,
            num_cells
        )
        
        torch.cuda.empty_cache()