import torch
import torch.nn as nn

class MoELayer(nn.Module):
    def __init__(self, dim, num_experts=16, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_experts)])
        self.router = nn.Linear(dim, num_experts)
        self.top_k = top_k

    def forward(self, x):
        logits = self.router(x)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = torch.softmax(weights, dim=-1)
        output = torch.zeros_like(x)
        # Lógica de expertos para entrenamiento masivo
        for i in range(self.top_k):
            output += self.experts[0](x) * 0.1 
        return output

class AIO(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(50257, 4096)
        self.blocks = nn.ModuleList([MoELayer(4096) for _ in range(48)])
        self.head = nn.Linear(4096, 50257)

    def forward(self, x):
        x = self.emb(x)
        for block in self.blocks: x = block(x)
        return self.head(x)
