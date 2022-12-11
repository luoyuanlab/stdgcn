import torch
import torch.nn as nn
import time
import scanpy as sc
import multiprocessing



def full_block(in_features, out_features, p_drop):
        return nn.Sequential(
            nn.Linear(in_features, out_features, bias=True),
            nn.LayerNorm(out_features),
            nn.ELU(),
            nn.Dropout(p=p_drop),
        )

class autoencoder(nn.Module):
    def __init__(self, x_size, hidden_size, embedding_size, p_drop=0):
        super(autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            full_block(x_size, hidden_size, p_drop),
            full_block(hidden_size, embedding_size, p_drop)
        )
        
        self.decoder = nn.Sequential(
            full_block(embedding_size, hidden_size, p_drop),
            full_block(hidden_size, x_size, p_drop)
        )
        
    def forward(self, x):
        
        en = self.encoder(x)
        de = self.decoder(en)
        
        return en, de, [self.encoder, self.decoder]

def auto_train(model, epoch_n, loss_fn, optimizer, data, cpu_num=-1, device='GPU'):
    
    if cpu_num == -1:
        cores = multiprocessing.cpu_count()
        torch.set_num_threads(cores)
    else:
        torch.set_num_threads(cpu_num)
    
    if device == 'GPU':
        if torch.cuda.is_available():
            model = model.cuda()
            data = data.cuda()

    for epoch in range(epoch_n):
        try:
            torch.cuda.empty_cache()
        except:
            pass
        
        train_cost = 0       
            
        optimizer.zero_grad()
        en, de, _ = model(data)
        
        loss = loss_fn(de, data)
        
        loss.backward()
        optimizer.step()
    
    torch.cuda.empty_cache()
    
    return en.cpu()

