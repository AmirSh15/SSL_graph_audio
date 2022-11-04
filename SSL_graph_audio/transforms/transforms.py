import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg

class Compose(nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def forward(self, data):
        for aug in self.transforms:
            data = aug(data)
        return data

class Denoising(nn.Module):
    def __init__(self, p=0.05, mu=0.0, sigma=1.0):
        super().__init__()
        self.p = p
        self.mu = mu
        self.sigma = sigma

    def forward(self, data):
        x = data.x
        y = data.y

        edge_idx = data.edge_index

        n, d = x.shape

        idx = torch.empty((d,), dtype=torch.float32).uniform_(0, 1) < self.p
        x = x.clone()
        x[:, idx] = x[:, idx] + torch.empty((n, idx.sum()), dtype=torch.float32).normal_(self.mu, self.sigma)

        new_data = tg.data.Data(x=x, y=y, edge_index=edge_idx)
        return new_data

class Shuffling(nn.Module):
    def __init__(self, p=0.05):
        super().__init__()
        self.p = p

    def forward(self, data):
        x = data.x
        y = data.y
        edge_idx = data.edge_index

        edge_idx = edge_idx.permute(1, 0)
        idx = torch.empty(edge_idx.size(0)).uniform_(0, 1)
        edge_idx = edge_idx[torch.where(idx >= self.p)].permute(1, 0)
        new_data = tg.data.Data(x=x, y=y, edge_index=edge_idx)
        return new_data

class Completion(nn.Module):
    def __init__(self, p=0.15):
        super().__init__()
        self.p = p

    def forward(self, data):
        x = data.x
        y = data.y
        edge_idx = data.edge_index

        n, d = x.shape
        
        idx = torch.empty((d,), dtype=torch.float32).uniform_(0, 1) < self.p
        x = x.clone()
        x[:, idx] = 0

        new_data = tg.data.Data(x=x, y=y, edge_index=edge_idx)
        return new_data