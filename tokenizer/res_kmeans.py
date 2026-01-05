import torch
from torch import nn

class ResKmeans(nn.Module):

    def __init__(self, n_layers, codebook_size, dim, extra_kmeans_config=None, **kwargs):
        super().__init__()
        self.n_layers = n_layers
        self.codebook_size = codebook_size
        self.dim = dim
        self.extra_kmeans_config = extra_kmeans_config
        self.centroids = nn.ParameterList([
            nn.Parameter(torch.zeros((codebook_size,dim), requires_grad=False))
            for i in range(n_layers)
        ])

    def calc_loss(self, x, out, epsilon=1e-4):
        loss = ((out - x) ** 2).mean()
        rel_loss = (torch.abs(x - out) / (torch.maximum(torch.abs(x), torch.abs(out)) + epsilon)).mean()
        return {'loss': loss.item(), 'rel_loss': rel_loss.item()}
    
    def train_kmeans(self, inputs, verbose=True):
        import faiss
        kmeans = faiss.Kmeans(self.dim, self.codebook_size, spherical=False, **self.extra_kmeans_config)
        x = inputs.clone()
        out = torch.zeros_like(x)
        for l in range(self.n_layers):
            kmeans.train(x)
            _, I = kmeans.index.search(x, 1)
            I = I.reshape([-1])
            o = torch.tensor(kmeans.centroids[I])
            out += o
            if verbose:
                losses = self.calc_loss(inputs, out)
                print(l, losses)
            x = x - o
            self.centroids[l] = nn.Parameter(torch.tensor(kmeans.centroids.copy()), requires_grad=False)
            print(f"layer {l} finished")
    
    def encode(self, x, n_layers=None):
        if n_layers is None:
            n_layers = self.n_layers
        else:
            assert n_layers <= self.n_layers
        out = []
        for l in range(n_layers):
            x_norm_sq = x.pow(2.).sum(dim=1, keepdim=True)
            codebook_t_norm_sq = self.centroids[l].T.pow(2.).sum(dim=0, keepdim=True)
            distances = torch.addmm(x_norm_sq + codebook_t_norm_sq, x, self.centroids[l].T, alpha=-2.0)
            code = distances.argmin(dim=-1)
            x = x - self.centroids[l][code]
            out.append(code)
        out = torch.stack(out, dim=1)
        return out
    
    def decode(self, code):
        out = torch.zeros((code.shape[0], self.dim), dtype=torch.float32, device=code.device)
        n_layers = code.shape[1]
        assert n_layers <= self.n_layers
        for l in range(n_layers):
            c = code[:, l]
            out += self.centroids[l][c]
        return out
