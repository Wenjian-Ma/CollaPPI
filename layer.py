import torch
import torch.nn as nn
from torch_geometric.nn import GATConv,GCNConv,global_add_pool,SAGPooling,global_mean_pool
import torch.nn.functional as F

# intra rep
class IntraGraphAttention(nn.Module):
    def __init__(self, input_dim,out_dim=64,heads=2):
        super().__init__()
        self.input_dim = input_dim
        self.intra = GATConv(input_dim, out_dim,heads)
        #self.intra = GCNConv(input_dim,out_dim)

    def forward(self, p):
        input_feature, edge_index = p.x,p.edge_index
        input_feature = F.elu(input_feature)
        intra_rep = self.intra(input_feature, edge_index)
        return intra_rep


# class InterGraphAttention(nn.Module):
#     def __init__(self, input_dim,out_dim=32,heads=1):
#         super().__init__()
#         self.input_dim = input_dim
#         self.inter = GATConv((input_dim, input_dim),out_dim,heads)
#
#     def forward(self, p1_data, p2_data, b_graph):
#         edge_index = b_graph.edge_index
#         p1_input = F.elu(p1_data.x)
#         p2_input = F.elu(p2_data.x)
#         p2_rep = self.inter((p1_input, p2_input), edge_index)
#         p1_rep = self.inter((p2_input, p1_input), edge_index[[1, 0]])
#         return p1_rep, p2_rep


class MV_PPI_Block(nn.Module):
    def __init__(self, n_heads, in_features, head_out_feats):
        super().__init__()

        # self.feature_conv = GATConv(in_features, head_out_feats, n_heads)
        #self.feature_conv = GCNConv(in_features, head_out_feats)
        self.intraAtt1 = IntraGraphAttention(head_out_feats * n_heads,heads=n_heads)
        self.intraAtt2 = IntraGraphAttention(head_out_feats * n_heads, heads=n_heads)

        self.dim = head_out_feats* n_heads#//2
        self.W1_attention = nn.Linear(self.dim, self.dim)
        self.W2_attention = nn.Linear(self.dim, self.dim)
        w = nn.Parameter(torch.Tensor(self.dim,1))
        self.w = nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))

        self.readout1 = SAGPooling(n_heads * head_out_feats, min_score=-1)
        self.readout2 = SAGPooling(n_heads * head_out_feats, min_score=-1)

    def mutual_attention_func(self,splited_h):

        splited_h1 = splited_h[0]
        splited_h2 = splited_h[1]

        m1 = splited_h1.size()[0]
        m2 = splited_h2.size()[0]
        ######################################
        c1 = splited_h1.repeat(1, m2).view(m1, m2, self.dim)
        c2 = splited_h2.repeat(m1, 1).view(m1, m2, self.dim)

        d = torch.tanh(c1 + c2)#*
        alpha = torch.matmul(d, self.w).view(m1, m2)
        #######################################
        #alpha = torch.tanh(torch.mm(splited_h1,torch.t(splited_h2)))
        b1 = torch.mean(alpha, 1)
        p1 = torch.softmax(b1, 0)
        s1 = torch.matmul(torch.t(splited_h1), p1).view(1, -1)

        b2 = torch.mean(alpha, 0)
        p2 = torch.softmax(b2, 0)
        s2 = torch.matmul(torch.t(splited_h2), p2).view(1, -1)

        return s1,s2

    def mutual_attention(self, h1, h2):
        x1 = self.W1_attention(h1.x)
        x2 = self.W2_attention(h2.x)
        #########################################

        mark_h1 = list(torch.unique(h1.batch,return_counts=True)[1].cpu().tolist())
        mark_h2 = list(torch.unique(h2.batch, return_counts=True)[1].cpu().tolist())

        splited_h1 = torch.split(x1,mark_h1,dim=0)
        splited_h2 = torch.split(x2, mark_h2, dim=0)

        h1_total, h2_total = zip(*list(map(self.mutual_attention_func,list(zip(splited_h1,splited_h2)))))
        h1_total, h2_total = torch.vstack(list(h1_total)),torch.vstack(list(h2_total))

        return h1_total,h2_total

    def forward(self,p1,p2):
        p1.x = self.intraAtt1(p1)
        p2.x = self.intraAtt1(p2)

        #ablation
        # p1_x = global_mean_pool(p1.x, p1.batch)
        # p2_x = global_mean_pool(p2.x, p2.batch)
        #

        #Mutual Attention Model#
        p1_x, p2_x = self.mutual_attention(p1, p2)
        #
        return p1_x,p2_x

