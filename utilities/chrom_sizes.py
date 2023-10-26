import numpy as np

def chrom_sizes(f,length=np.inf):
    data = open(f,'r')
    
    sizes = {}
    
    for line in data:
        ldata = line.split()
        
        if len(ldata[0]) > length:
            continue
            
        sizes[ldata[0]] = int(ldata[1])

    return sizes