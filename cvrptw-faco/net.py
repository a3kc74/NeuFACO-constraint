import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric.nn as gnn


class EmbNet(nn.Module):
    def __init__(self, depth=12, node_feats=7, edge_feats=6, units=32, act_fn='silu', agg_fn='mean'):
        super().__init__()
        self.depth = depth
        self.node_feats = node_feats
        self.edge_feats = edge_feats
        self.units = units
        self.act_fn = getattr(F, act_fn)
        self.agg_fn = getattr(gnn, f'global_{agg_fn}_pool')
        self.v_lin0 = nn.Linear(self.node_feats, self.units)
        self.v_lins1 = nn.ModuleList([nn.Linear(self.units, self.units) for _ in range(self.depth)])
        self.v_lins2 = nn.ModuleList([nn.Linear(self.units, self.units) for _ in range(self.depth)])
        self.v_lins3 = nn.ModuleList([nn.Linear(self.units, self.units) for _ in range(self.depth)])
        self.v_lins4 = nn.ModuleList([nn.Linear(self.units, self.units) for _ in range(self.depth)])
        self.v_bns = nn.ModuleList([gnn.BatchNorm(self.units) for _ in range(self.depth)])
        self.e_lin0 = nn.Linear(self.edge_feats, self.units)
        self.e_lins0 = nn.ModuleList([nn.Linear(self.units, self.units) for _ in range(self.depth)])
        self.e_bns = nn.ModuleList([gnn.BatchNorm(self.units) for _ in range(self.depth)])

    def reset_parameters(self):
        raise NotImplementedError

    def forward(self, x, edge_index, edge_attr):
        x = self.act_fn(self.v_lin0(x))
        w = self.act_fn(self.e_lin0(edge_attr))
        for i in range(self.depth):
            x0 = x
            w0 = w
            x1 = self.v_lins1[i](x0)
            x2 = self.v_lins2[i](x0)
            x3 = self.v_lins3[i](x0)
            x4 = self.v_lins4[i](x0)
            w1 = self.e_lins0[i](w0)
            w2 = torch.sigmoid(w0)
            x = x0 + self.act_fn(self.v_bns[i](x1 + self.agg_fn(w2 * x2[edge_index[1]], edge_index[0])))
            w = w0 + self.act_fn(self.e_bns[i](w1 + x3[edge_index[0]] + x4[edge_index[1]]))
        return w


class MLP(nn.Module):
    @property
    def device(self):
        return self._dummy.device

    def __init__(self, units_list, act_fn, final_sigmoid=False):
        super().__init__()
        self._dummy = nn.Parameter(torch.empty(0), requires_grad=False)
        self.units_list = units_list
        self.depth = len(self.units_list) - 1
        self.act_fn = getattr(F, act_fn)
        self.final_sigmoid = final_sigmoid
        self.lins = nn.ModuleList([nn.Linear(self.units_list[i], self.units_list[i + 1]) for i in range(self.depth)])

    def forward(self, x):
        for i in range(self.depth):
            x = self.lins[i](x)
            if i < self.depth - 1:
                x = self.act_fn(x)
            elif self.final_sigmoid:
                x = torch.sigmoid(x)
        return x


class ParNet(MLP):
    def __init__(self, depth=3, units=32, preds=1, act_fn='silu'):
        self.units = units
        self.preds = preds
        super().__init__([self.units] * depth + [self.preds], act_fn, final_sigmoid=False)

    def forward(self, x):
        return super().forward(x).squeeze(dim=-1)


class Net(nn.Module):
    def __init__(self, value_head=False, node_feats=7, edge_feats=6, units=32):
        super().__init__()
        self.emb_net = EmbNet(node_feats=node_feats, edge_feats=edge_feats, units=units)
        self.par_net_heu = ParNet(units=units)
        self.value_head = value_head
        self.value_net = nn.Sequential(
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, 1),
        ) if value_head else None

    def forward(self, pyg, return_value=False):
        emb = self.emb_net(pyg.x, pyg.edge_index, pyg.edge_attr)
        logits = self.par_net_heu(emb)
        if return_value:
            assert self.value_head and self.value_net is not None
            value = self.value_net(emb).mean(0)
            return logits, value
        return logits

    def freeze_gnn(self):
        for param in self.emb_net.parameters():
            param.requires_grad = False

    @staticmethod
    def reshape(pyg, vector):
        n_nodes = int(getattr(pyg, 'n_nodes', pyg.x.shape[0]))
        k_sparse = int(getattr(pyg, 'k_sparse'))
        return vector.reshape(n_nodes, k_sparse)


class Critic(nn.Module):
    def __init__(self, node_feats=7, edge_feats=6, units=32):
        super().__init__()
        self.emb_net = EmbNet(node_feats=node_feats, edge_feats=edge_feats, units=units)
        self.value_net = nn.Sequential(
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, 1),
        )

    def forward(self, pyg):
        emb = self.emb_net(pyg.x, pyg.edge_index, pyg.edge_attr)
        return self.value_net(emb).mean(0)
