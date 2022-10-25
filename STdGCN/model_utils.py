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
    
    
    
class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))

    
    
class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, y_pred, y_true):
        total_loss = torch.sum(-y_true * torch.log(y_pred) - (1 - y_true) * torch.log(1 - y_pred), axis=1)
        num_of_samples = y_pred.shape[1]
        mean_loss = total_loss / num_of_samples
        mean_sample_loss = mean_loss.mean()
        return mean_sample_loss