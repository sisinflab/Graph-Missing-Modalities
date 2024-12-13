import torch

class HGNNLayer(torch.nn.Module):
    def __init__(self, n_hyper_layer):
        super(HGNNLayer, self).__init__()

        self.h_layer = n_hyper_layer

    def forward(self, i_hyper, u_hyper, embeds):
        i_ret = embeds
        for _ in range(self.h_layer):
            lat = torch.mm(i_hyper.T, i_ret)
            i_ret = torch.mm(i_hyper, lat)
            u_ret = torch.mm(u_hyper, lat)
        return u_ret, i_ret
