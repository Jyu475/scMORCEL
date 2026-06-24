import torch
import torch.utils.data as Data
def Weighted_Sampling(train_label = None):

    class_sample_count = torch.tensor(
        [(train_label == t).sum() for t in torch.unique(train_label, sorted=True)])
    weight = 1. / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in train_label])
    sampler = Data.WeightedRandomSampler(samples_weight,#每个样本的权重 
                                         len(samples_weight))# 采样总数

    return sampler

