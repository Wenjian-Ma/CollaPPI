import warnings
import argparse
import torch
from utils import load_dataset,load_files,PPIDataset,collate
from torch.utils.data import DataLoader
from model import nnModel
from tqdm import tqdm
from metric import cal_metric_test as cal_metric
from model_multi_class import nnModel_multi_class

def test(args):
    device = torch.device('cuda:' + args.device)
    test_items = load_dataset(dataset=args.dataset, mark='test',args=args)
    contact_dic, lm_dic,GO_dic,location_dic = load_files(dataset=args.dataset)
    test_data = PPIDataset(data=test_items, contact_dic=contact_dic, lm_dic=lm_dic, GO_dic=GO_dic,location_dic=location_dic)
    dataset_test = DataLoader(test_data, shuffle=False, batch_size=args.batch_size, collate_fn=collate, drop_last=False,num_workers=6)
    if args.dataset == 'yeast' or args.dataset == 'multi_class':
        model_path = './data/'+args.dataset+'/dictionary/models/'+args.dataset+'_model.pt'
    elif args.dataset == 'multi_species':
        model_path = './data/' + args.dataset + '/dictionary/models/' + args.dataset + '_'+args.identity+'_model.pt'
    params_dict = torch.load(model_path)
    if args.dataset == 'yeast' or args.dataset == 'multi_species':
        model = nnModel(feature_dim=args.feat_dim, heads=args.heads).to(device)
    elif args.dataset == 'multi_class':
        model = nnModel_multi_class(feature_dim=args.feat_dim, heads=args.heads).to(device)
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
            if args.dataset == 'yeast' or args.dataset == 'multi_species':
                true_label = p1.label.reshape((1, -1))
            elif args.dataset == 'multi_class':
                true_label = p1.label.reshape((-1, 1)).squeeze()
            predicted_label, predicted_go_label1, predicted_go_label2, predicted_location_label1, predicted_location_label2 = model(p1, p2)
            if args.dataset == 'yeast' or args.dataset == 'multi_species':
                predicted_label = predicted_label.reshape((1, -1))
            if args.dataset == 'yeast' or args.dataset == 'multi_species':
                predicted_label_round = torch.round(predicted_label)
                total_preds = torch.cat((total_preds, predicted_label_round.cpu()), 1)
                total_labels = torch.cat((total_labels, true_label.cpu()), 1)
                total_scores = torch.cat((total_scores, predicted_label.cpu()), 1)
            elif args.dataset == 'multi_class':
                predicted_label_round = torch.argmax(predicted_label, dim=1)
                total_preds = torch.cat((total_preds, predicted_label_round.cpu()), 0)
                total_labels = torch.cat((total_labels, true_label.cpu()), 0)
                total_scores = torch.cat((total_scores, predicted_label.cpu()), 0)
    test_acc, test_prec, test_recall, test_f1, test_auc, con_matrix, test_spec, test_mcc = cal_metric(total_preds,
                                                                                                      total_labels,total_scores,args)
    if args.dataset == 'yeast' or args.dataset == 'multi_species':
        print('Acc:',
          round(float(test_acc.item()), 5), '\tPre:', round(float(test_prec.item()), 5), '\tSen:',
          round(float(test_recall.item()), 5), '\tSpe:', round(float(test_spec.item()), 5), '\tF1:',
          round(float(test_f1.item()), 5), '\tMCC:', round(float(test_mcc.item()), 5), '\tAUC:',
          round(float(test_auc.item()), 5))
    elif args.dataset == 'multi_class':
        print('Acc:',
              round(float(test_acc.item()), 5), '\tPre:', round(float(test_prec.item()), 5), '\tF1:',
              round(float(test_f1.item()), 5))

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='yeast', help="dataset.yeast/multi_class/multi_species")
    parser.add_argument('--identity', type=str, default='01', help="any/01/10/25/40")
    parser.add_argument('--device', type=str, default='1', help="cuda device.")
    parser.add_argument('--feat_dim', type=int, default=1280, help="the dimension of feature.")
    parser.add_argument('--heads', type=int, default=1, help="attention heads.")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size.")
    args = parser.parse_args()
    print(args)
    test(args)
