import warnings
import argparse
import torch
from utils import load_dataset,load_files,PPIDataset,collate
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from torch_geometric.nn import GATConv,LayerNorm,SAGPooling
import torch.nn.functional as F
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score,confusion_matrix,roc_curve, auc


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


class MV_PPI_Block(nn.Module):
    def __init__(self, n_heads, in_features, head_out_feats):
        super().__init__()


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

        c1 = splited_h1.repeat(1, m2).view(m1, m2, self.dim)
        c2 = splited_h2.repeat(m1, 1).view(m1, m2, self.dim)

        d = torch.tanh(c1 * c2)#+
        alpha = torch.matmul(d, self.w).view(m1, m2)

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

        #Mutual Attention Model#
        p1_x, p2_x = self.mutual_attention(p1, p2)
        #
        return p1_x,p2_x



class nnModel(nn.Module):

    def __init__(self,feature_dim = 1280,heads=2):
        super(nnModel,self).__init__()
        # self.GCN1 = GCNConv(feature_dim, 512)
        # self.GCN1_ = GCNConv(512, 256)
        self.relu = nn.ReLU()
        self.feature_conv = GATConv(feature_dim, 64, heads)

        self.block1 = MV_PPI_Block(heads,feature_dim,64)

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
    def forward(self, p1, p2):

        p1.x = self.ln(p1.x)
        p2.x = self.ln(p2.x)

        p1.x = self.feature_conv(p1.x, p1.edge_index)
        p2.x = self.feature_conv(p2.x, p2.edge_index)


        out1 = self.block1(p1,p2)
        p1_rep_layer1 = out1[0]
        p2_rep_layer1 = out1[1]

        rep = torch.cat((p1_rep_layer1, p2_rep_layer1), 1)
        rep = self.dense1(rep)
        rep = self.relu(rep)
        rep = self.dropout(rep)
        rep = self.dense2(rep)
        rep = self.relu(rep)
        rep = self.dropout(rep)
        rep = self.dense3(rep)
        out = self.sigmoid(rep)


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


        return out,x_go1,x_go2,x_location1,x_location2


def cal_metric(pred,label,score,args):
    pred = torch.squeeze(pred)
    label = torch.squeeze(label)
    score = torch.squeeze(score)
    average = 'binary'
        # multi_class = 'raise'
    # np.save('./data/AUROC_Fig/' + args.dataset + '/label.npy',np.array(label))
    # np.save('./data/AUROC_Fig/' + args.dataset + '/score.npy',np.array(score))
    test_acc = accuracy_score(label, pred)#
    test_prec = precision_score(label, pred,average=average)#,average='micro'
    test_recall = recall_score(label, pred,average=average)#
    test_f1 = f1_score(label, pred,average=average)#


    fpr, tpr, threshold = roc_curve(label, score)
    test_auc = auc(fpr, tpr)

    # line_width = 1  #
    # plt.figure(figsize=(8, 5))  #
    # plt.plot(fpr, tpr, lw=line_width, label='HSIC-MKL + LP-S (AUC = %0.4f)' % test_auc, color='red')
    #test_auc = roc_auc_score(label, score)#pred
    con_matrix = confusion_matrix(label, pred)
    test_spec = con_matrix[0][0] / (con_matrix[0][0] + con_matrix[0][1])
    test_mcc = (con_matrix[0][0] * con_matrix[1][1] - con_matrix[0][1] * con_matrix[1][0]) / (((con_matrix[1][1] +
con_matrix[0][1]) * (con_matrix[1][1] +con_matrix[1][0]) * (con_matrix[0][0]+con_matrix[0][1])*(con_matrix[0][0] +con_matrix[1][0])) ** 0.5)
    return test_acc,test_prec,test_recall,test_f1,test_auc,con_matrix,test_spec,test_mcc

def test(args):
    device = torch.device('cuda:' + args.device)
    test_items = load_dataset(dataset='yeast', mark='test',args=args)
    contact_dic, lm_dic,GO_dic,location_dic = load_files(dataset='yeast')
    test_data = PPIDataset(data=test_items, contact_dic=contact_dic, lm_dic=lm_dic, GO_dic=GO_dic,location_dic=location_dic)
    dataset_test = DataLoader(test_data, shuffle=False, batch_size=args.batch_size, collate_fn=collate, drop_last=False,num_workers=6)

    model_path = './data/yeast/dictionary/models/yeast_model_mul.pt'

    params_dict = torch.load(model_path,map_location=device)

    model = nnModel(feature_dim=args.feat_dim, heads=args.heads).to(device)

    model = model.to(device)
    model.load_state_dict(params_dict)

    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_scores = torch.Tensor()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataset_test, mininterval=0.5, desc='testing', leave=False, ncols=50)):
            p1 = batch[0].to(device)
            p2 = batch[1].to(device)
            # p1_cnn = batch[2].to(device)
            # p2_cnn = batch[3].to(device)
            # BG_p1_p2 = batch[2].to(device)

            true_label = p1.label.reshape((1, -1))

            predicted_label, predicted_go_label1, predicted_go_label2, predicted_location_label1, predicted_location_label2 = model(p1, p2)

            predicted_label = predicted_label.reshape((1, -1))

            predicted_label_round = torch.round(predicted_label)
            total_preds = torch.cat((total_preds, predicted_label_round.cpu()), 1)
            total_labels = torch.cat((total_labels, true_label.cpu()), 1)
            total_scores = torch.cat((total_scores, predicted_label.cpu()), 1)

    test_acc, test_prec, test_recall, test_f1, test_auc, con_matrix, test_spec, test_mcc = cal_metric(total_preds,
                                                                                                      total_labels,total_scores,args)

    print('Acc:',
      round(float(test_acc.item()), 5), '\tPre:', round(float(test_prec.item()), 5), '\tSen:',
      round(float(test_recall.item()), 5), '\tSpe:', round(float(test_spec.item()), 5), '\tF1:',
      round(float(test_f1.item()), 5), '\tMCC:', round(float(test_mcc.item()), 5), '\tAUC:',
      round(float(test_auc.item()), 5))


if __name__ == "__main__":
    #only for Yeast dataset
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--identity', type=str, default='01', help="any/01/10/25/40")
    parser.add_argument('--device', type=str, default='1', help="cuda device.")
    parser.add_argument('--feat_dim', type=int, default=1280, help="the dimension of feature.")
    parser.add_argument('--heads', type=int, default=1, help="attention heads.")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size.")
    args = parser.parse_args()
    print(args)
    test(args)