import warnings
import argparse
import torch
from utils import load_dataset,PPIDataset,load_files,collate#,load_seqs_feat
from torch.utils.data import DataLoader
from trainNN import train_nn
from model import nnModel
from model_multi_class import nnModel_multi_class
def train(args):
    device = torch.device('cuda:' + args.device)
    train_items = load_dataset(dataset=args.dataset,mark='train',args=args)
    test_items = load_dataset(dataset=args.dataset,mark='test',args=args)
    contact_dic,lm_dic,GO_dic,location_dic = load_files(dataset=args.dataset)
    #seq_feat_dic = load_seqs_feat(dataset=args.dataset)
    train_data = PPIDataset(data = train_items,contact_dic = contact_dic,lm_dic=lm_dic,GO_dic=GO_dic,location_dic=location_dic)
    test_data = PPIDataset(data=test_items, contact_dic=contact_dic, lm_dic=lm_dic,GO_dic=GO_dic,location_dic=location_dic)

    dataset_train = DataLoader(train_data, shuffle=True,batch_size=args.batch_size, collate_fn=collate, drop_last=False,num_workers=6)
    dataset_test = DataLoader(test_data,  shuffle=False, batch_size=args.batch_size, collate_fn=collate, drop_last=False,num_workers=6)
    if args.dataset == 'yeast' or args.dataset == 'multi_species':
        model = nnModel(feature_dim = args.feat_dim,heads=args.heads).to(device)
    elif args.dataset == 'multi_class':
        model = nnModel_multi_class(feature_dim = args.feat_dim,heads=args.heads).to(device)


    train_nn(model=model,train_loader = dataset_train,test_loader = dataset_test,device=device,args = args)

    print()












if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='yeast', help="dataset:yeast/multi_class/multi_species")
    parser.add_argument('--identity', type=str, default='25', help="any/01/10/25/40")
    parser.add_argument('--device', type=str, default='3', help="cuda device.")
    parser.add_argument('--feat_dim', type=int, default=1280, help="the dimension of feature.")
    parser.add_argument('--heads', type=int, default=1, help="attention heads.")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size.")
    args = parser.parse_args()
    print(args)
    train(args)