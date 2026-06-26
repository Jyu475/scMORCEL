import torch
import torch.nn as nn
import torch.nn.functional as F


class classifier(nn.Module):

    def __init__(self, input_size, class_num):

        if input_size is None or class_num is None:
            raise ValueError("Must declare number of features and number of classes")

        super(classifier, self).__init__()

        if class_num < 16:
            hidden_dims = [256, 128, 64]
        elif class_num < 64:
            hidden_dims = [512, 256, 128]
        else:
            hidden_dims = [1024, 512, 256]

        self.layer1 = nn.Linear(input_size, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(0.5)

        self.layer2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(0.5)

        self.layer3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])
        self.elu3 = nn.ELU()
        self.dropout3 = nn.Dropout(0.1)

        self.layer4 = nn.Linear(hidden_dims[2], class_num)

        self.intermediate_features = None

    def forward(self, x):

        out = self.layer1(x)
        out = self.bn1(out)
        out = self.elu1(out)
        out = self.dropout1(out)

        out = self.layer2(out)
        out = self.bn2(out)
        out = self.elu2(out)
        out = self.dropout2(out)

        out = self.layer3(out)
        out = self.bn3(out)
        out = self.elu3(out)

        self.intermediate_features = out.clone()

        out = self.dropout3(out)
        out = self.layer4(out)

        return out


class GeneAttention(nn.Module):

    def __init__(self, input_size, reduction=4):
        super(GeneAttention, self).__init__()

        hidden_dim = max(32, input_size // reduction)

        self.attention = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_size),
            nn.Sigmoid()   
        )

    def forward(self, x):
        attention_weights = self.attention(x)     
        weighted = x * attention_weights
        enhanced = x + weighted                 
        return enhanced, attention_weights

class MultiHeadGeneAttention(nn.Module):

    def __init__(self, input_size, heads=4):
        super(MultiHeadGeneAttention, self).__init__()

        self.heads = nn.ModuleList([
            GeneAttention(input_size)
            for _ in range(heads)
        ])

        self.fusion = nn.Linear(input_size * heads, input_size)

    def forward(self, x):

        head_outputs = []
        all_weights = []

        for head in self.heads:
            out, weight = head(x)
            head_outputs.append(out)
            all_weights.append(weight)

        concatenated = torch.cat(head_outputs, dim=1)

        fused = self.fusion(concatenated)

        enhanced = fused + x

        attention_weights = torch.stack(all_weights, dim=1)

        return enhanced, attention_weights

class AttentionClassifier(nn.Module):

    def __init__(self, input_size, class_num, attention_heads=4):
        super(AttentionClassifier, self).__init__()

        self.attention = MultiHeadGeneAttention(
            input_size=input_size,
            heads=attention_heads
        )

        if class_num < 16:
            hidden_dims = [256, 128, 64]
        elif class_num < 64:
            hidden_dims = [512, 256, 128]
        else:
            hidden_dims = [1024, 512, 256]

        self.layer1 = nn.Linear(input_size, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(0.5)

        self.layer2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(0.5)

        self.layer3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])
        self.elu3 = nn.ELU()
        self.dropout3 = nn.Dropout(0.3)

        self.layer4 = nn.Linear(hidden_dims[2], class_num)

        self.intermediate_features = None

    def forward(self, x, return_attention=False):

        enhanced, attention_weights = self.attention(x)

        out = self.layer1(enhanced)
        out = self.bn1(out)
        out = self.elu1(out)
        out = self.dropout1(out)

        out = self.layer2(out)
        out = self.bn2(out)
        out = self.elu2(out)
        out = self.dropout2(out)

        out = self.layer3(out)
        out = self.bn3(out)
        out = self.elu3(out)

        self.intermediate_features = out.clone()

        out = self.dropout3(out)
        out = self.layer4(out)

        if return_attention:
            return out, attention_weights
        return out
