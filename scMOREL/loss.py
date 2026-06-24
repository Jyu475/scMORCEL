
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    
    def __init__(self, temperature=0.5, base_temperature=0.07):

        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(self, features, labels):

        device = features.device
        batch_size = features.shape[0]
 
        features = F.normalize(features, p=2, dim=1)
        

        labels = labels.contiguous().view(-1, 1)  # [batch_size, 1]
        
        mask = torch.eq(labels, labels.T).float().to(device)
        similarity_matrix = torch.matmul(features, features.T)

        similarity_matrix = similarity_matrix / self.temperature
        
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        
        mask = mask * logits_mask

        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        mask_sum = mask.sum(1)  # [batch_size]
        
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        
        loss = loss.mean()
        
        return loss


def select_loss_function(loss_type, num_classes=None, feat_dim=None, device=None, **kwargs):
    if loss_type == 'contrastive':
        temperature = kwargs.get('temperature', 0.5)
        return ContrastiveLoss(temperature=temperature)
    
    else:
        raise ValueError(f"未知的损失类型: {loss_type}")


