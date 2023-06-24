import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv,global_mean_pool as gep,LayerNorm
import torch
from layer import MV_PPI_Block#,ConvsLayer
import torch.nn.functional as F
class nnModel(nn.Module):

    def __init__(self,feature_dim = 1280,heads=2):
        super(nnModel,self).__init__()
        # self.GCN1 = GCNConv(feature_dim, 512)
        # self.GCN1_ = GCNConv(512, 256)
        self.relu = nn.ReLU()
        self.feature_conv = GATConv(feature_dim, 64, heads)
        #self.trans_feature = nn.Linear(feature_dim,64)

        self.block1 = MV_PPI_Block(heads,feature_dim,64)
        # self.block2 = MV_PPI_Block(heads, feature_dim, 64)
        # self.block3 = MV_PPI_Block(heads, feature_dim, 64)
        # self.block4 = MV_PPI_Block(heads, feature_dim, 64)
        # self.GCN2 = GCNConv(feature_dim,512)
        # self.GCN2_ = GCNConv(512, 256)
        self.dropout = nn.Dropout(0.2)
        self.dense1 = nn.Linear(64*2*heads,512)#64
        self.dense2 = nn.Linear(512,256)
        self.dense3 = nn.Linear(256,1)
        self.ln = LayerNorm(feature_dim)
        self.sigmoid = nn.Sigmoid()

        self.dense_go1 = nn.Linear(64,1024)
        self.dense_go2 = nn.Linear(1024,100)

        self.dense_location1 = nn.Linear(64,1024)
        self.dense_location2 = nn.Linear(1024,100)
        #self.textcnn = ConvsLayer(feature_dim)
        #self.textflatten = nn.Linear(128,32)
        #self.w = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
    def forward(self, p1, p2):
        # p1_x,p1_edge_index,p1_batch = p1.x,p1.edge_index,p1.batch
        # p2_x, p2_edge_index, p2_batch = p2.x, p2.edge_index, p2.batch

        p1.x = self.ln(p1.x)
        p2.x = self.ln(p2.x)

        p1.x = self.feature_conv(p1.x, p1.edge_index)
        p2.x = self.feature_conv(p2.x, p2.edge_index)

        # p1.x = self.trans_feature(p1.x)
        # p2.x = self.trans_feature(p2.x)


        out1 = self.block1(p1,p2)
        p1_rep_layer1 = out1[0]
        p2_rep_layer1 = out1[1]
        # p1_global_graph_emb = out1[2]
        # p2_global_graph_emb = out1[3]

        # out2 = self.block2(p1,p2)
        # p1_rep_layer2 = out2[0]
        # p2_rep_layer2 = out2[1]
        #
        # out3 = self.block3(p1, p2)
        # p1_rep_layer3 = out3[0]
        # p2_rep_layer3 = out3[1]

        #######################
        # p1_cnn_rep = self.textcnn(p1_cnn)
        # p1_cnn_rep = self.relu(self.textflatten(p1_cnn_rep))
        # p2_cnn_rep = self.textcnn(p2_cnn)
        # p2_cnn_rep = self.relu(self.textflatten(p2_cnn_rep))
        # w = F.sigmoid(self.w)
        # p1_rep = torch.add((1-w)*p1_rep,w*p1_cnn_rep)
        # p2_rep = torch.add((1-w)*p2_rep,w*p2_cnn_rep)
        ########################

        ##rep = torch.cat((p1_rep,p1_cnn_rep,p2_rep,p2_cnn_rep),1)
        rep = torch.cat((p1_rep_layer1, p2_rep_layer1), 1)
        rep = self.dense1(rep)
        rep = self.relu(rep)
        rep = self.dropout(rep)
        rep = self.dense2(rep)
        rep = self.relu(rep)
        rep = self.dropout(rep)
        rep = self.dense3(rep)
        out = self.sigmoid(rep)


        # p1_x = self.GCN1(p1_x,p1_edge_index)
        # p1_x = self.relu(p1_x)
        # p1_x = self.GCN1_(p1_x, p1_edge_index)
        # p1_x = self.relu(p1_x)
        # p1_x = gep(p1_x,p1_batch)
        #
        # p2_x = self.GCN2(p2_x,p2_edge_index)
        # p2_x = self.relu(p2_x)
        # p2_x = self.GCN2_(p2_x, p2_edge_index)
        # p2_x = self.relu(p2_x)
        # p2_x = gep(p2_x,p2_batch)
        #
        # x = torch.cat((p1_x,p2_x),1)
        # x = self.dense1(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.dense2(x)
        # out = self.sigmoid(x)

        x_go1 = self.dense_go1(p1_rep_layer1)
        x_go1 = self.relu(x_go1)
        x_go1 = self.dropout(x_go1)
        x_go1 = self.dense_go2(x_go1)
        x_go1 = self.sigmoid(x_go1)

        x_go2 = self.dense_go1(p2_rep_layer1)
        x_go2 = self.relu(x_go2)
        x_go2 = self.dropout(x_go2)
        x_go2 = self.dense_go2(x_go2)
        x_go2 = self.sigmoid(x_go2)

        x_location1 = self.dense_location1(p1_rep_layer1)
        x_location1 = self.relu(x_location1)
        x_location1 = self.dropout(x_location1)
        x_location1 = self.dense_location2(x_location1)
        x_location1 = self.sigmoid(x_location1)

        x_location2 = self.dense_location1(p2_rep_layer1)
        x_location2 = self.relu(x_location2)
        x_location2 = self.dropout(x_location2)
        x_location2 = self.dense_location2(x_location2)
        x_location2 = self.sigmoid(x_location2)


        return out,x_go1,x_go2,x_location1,x_location2#,recon_adj_p1,recon_adj_p2
