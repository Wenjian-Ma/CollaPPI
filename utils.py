import os,torch
from torch_geometric.data import InMemoryDataset,Batch
import numpy as np
from tqdm import tqdm
from torch_geometric import data as DATA
# from prot_feat import prot_to_feature
def load_dataset(dataset='yeast',mark='train',args=None):
    data = []
    if dataset == 'yeast' or dataset == 'multi_species':
        split_mark = '\t'
    elif dataset == 'multi_class':
        split_mark = ' '
    if dataset != 'multi_species':
        with open('./data/'+dataset+'/actions/'+mark+'_cmap.actions.tsv') as f:
            for line in f:
                prot1 = line.strip().split(split_mark)[0]
                prot2 = line.strip().split(split_mark)[1]
                interaction = line.strip().split(split_mark)[2]
                data.append([prot1, prot2, interaction])
    elif dataset == 'multi_species':
        with open('./data/'+dataset+'/actions/'+mark+'-multi-'+args.identity+'.tsv') as f:
            for line in f:
                prot1 = line.strip().split(split_mark)[0]
                prot2 = line.strip().split(split_mark)[1]
                interaction = line.strip().split(split_mark)[2]
                data.append([prot1, prot2, interaction])
    return data

def load_files(dataset='yeast'):
    # if dataset == 'yeast':
    #     split_mark = '.'
    # elif dataset == 'multi_class':
    split_mark = '.npy'


    contact_path = './data/'+dataset+'/dictionary/contact_map_8.0/'
    contact_files = os.listdir(contact_path)
    contact_items = {}
    for i in tqdm(contact_files, desc='loading protein graph', leave=False):
        uid = i.split(split_mark)[0]
        contact_items[uid] = np.load(contact_path + i)

    lm_path = './data/'+dataset+'/dictionary/pretrained-emb-esm-2/'
    lm_files = os.listdir(lm_path)
    lm_items = {}
    for i in tqdm(lm_files, desc='loading feature files', leave=False):
        uid = i.split(split_mark)[0]
        lm_items[uid] = np.load(lm_path + i)

    ###########load GO-terms label
    GO_terms_path = './data/'+dataset+'/dictionary/GO_terms_label/'
    GO_files = os.listdir(GO_terms_path)
    GO_items = {}
    for i in tqdm(GO_files, desc='loading GO terms', leave=False):
        uid = i.split(split_mark)[0]
        GO_items[uid] = np.load(GO_terms_path+i)
    ###########
    ############load location label
    location_path = './data/'+dataset+'/dictionary/location_label/'
    location_files = os.listdir(location_path)
    location_items = {}
    for i in tqdm(location_files, desc='loading locations', leave=False):
        uid = i.split(split_mark)[0]
        location_items[uid] = np.load(location_path + i)
    ##############
    return contact_items,lm_items,GO_items,location_items

class PPIDataset(InMemoryDataset):
    def __init__(self, dir=None, data=None,contact_dic=None,lm_dic=None,GO_dic=None,location_dic=None,transform=None,pre_transform=None):

        super(PPIDataset, self).__init__( transform, pre_transform)
        self.dir=dir
        self.data = data
        self.contact_dic = contact_dic
        self.lm_dic = lm_dic

        self.GO_dic = GO_dic
        self.location_dic = location_dic
    def __len__(self):
        return int(len(self.data))

    def __getitem__(self, idx):
        single_item = self.data[idx]
        prot1 = single_item[0]
        prot2 = single_item[1]


        contact_map1 = self.contact_dic[prot1]
        lm_feature1 = self.lm_dic[prot1]
        contact_map2 = self.contact_dic[prot2]
        lm_feature2 = self.lm_dic[prot2]
        interaction = int(single_item[2])

        GO_label1 = self.GO_dic[prot1]
        GO_label2 = self.GO_dic[prot2]
        location_label1 = self.location_dic[prot1]
        location_label2 = self.location_dic[prot2]

        GCNProt1 = DATA.Data(x=torch.Tensor(lm_feature1),
                             edge_index=torch.LongTensor(np.argwhere(contact_map1==1).transpose(1, 0)),
                                label=torch.FloatTensor([interaction]), uid=prot1,GO_label=torch.Tensor(GO_label1),location_label=torch.Tensor(location_label1))

        GCNProt2 = DATA.Data(x=torch.Tensor(lm_feature2),
                             edge_index=torch.LongTensor(np.argwhere(contact_map2==1).transpose(1, 0)),
                             label=torch.FloatTensor([interaction]), uid=prot2,GO_label=torch.Tensor(GO_label2),location_label=torch.Tensor(location_label2))

        return GCNProt1,GCNProt2,interaction#,feature1_cnn,feature2_cnn#,Bipartite_graph


def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    #batchC = Batch.from_data_list([data[3] for data in data_list])
    # batchA_cnn = torch.vstack([data[3] for data in data_list])
    # batchB_cnn = torch.vstack([data[4] for data in data_list])
    return batchA, batchB#,batchA_cnn,batchB_cnn#,batchC

# def load_seqs_feat(dataset=None):
#     seq_feat_dic = {}
#     path = '/home/sgzhang/perl5/MV-PPI/data/'+dataset+'/dictionary/Sequence/'
#     fasta_files = os.listdir(path)
#     for i in tqdm(fasta_files, desc='loading seq features', leave=False):
#         uid = i.split('.')[0]
#         with open(path+i) as f:
#             for line in f:
#                 seq = line.strip()
#         seqs_feat = prot_to_feature(seq)
#         seq_feat_dic[uid] = seqs_feat
#     return seq_feat_dic
