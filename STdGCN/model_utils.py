import torch
import torch.nn as nn
import torch.nn.functional as F



class JSD(nn.Module):  
    def __init__(self):
        super(JSD, self).__init__()
    
    def forward(self, p_output, q_output, get_softmax=False):
        """
        Function that measures JS divergence between target and output logits:
        """
        KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        if get_softmax:
            p_output = F.softmax(p_output)
            q_output = F.softmax(q_output)
        log_mean_output = ((p_output + q_output )/2).log()
        return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2