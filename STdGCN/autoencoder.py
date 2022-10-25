import torch
import torch.nn as nn
import time
import scanpy as sc



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

def auto_train(model, epoch_n, loss_fn, optimizer, data, cpu_num=10):
    
    torch.set_num_threads(cpu_num)
    
    if torch.cuda.is_available():
        model = model.cuda()
    time_open = time.time()

    for epoch in range(epoch_n):
        train_cost = 0
        
        if torch.cuda.is_available():
            data = data.cuda()
            
        optimizer.zero_grad()
        en, de, _ = model(data)
        
        loss = loss_fn(de, data)
        
        loss.backward()
        optimizer.step()
            
    time_end = time.time() - time_open
    
    return en

