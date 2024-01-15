import warnings,argparse,random,math
import torch,os
from utils import load_dataset,PPIDataset,collate
from torch.utils.data import DataLoader
from model import nnModel
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from torch import optim
from loss import AutomaticWeightedLoss
from metric import cal_metric

def train_nn(model,train_loader,test_loader,device,args):
    if args.dataset != 'yeast':
        raise Exception('Five-folds is only used for yeast dataset.')

    Epoch = 200
    awl = AutomaticWeightedLoss(3)
    optimizer = optim.Adam([{'params':model.parameters()},{'params':awl.parameters(),'weight_decay': 0}], lr=0.001)
    #optimizer = optim.Adam(model.parameters(), lr=0.001)

    bceloss = nn.BCELoss()

    celoss = nn.CrossEntropyLoss()#NLLLoss()

    best_test_f1 = -999
    best_record = []

    for e in range(Epoch):

        model.train()
        for batch_idx_train,batch in enumerate(tqdm(train_loader,mininterval=0.5,desc='Training',leave=False,ncols=50)):
            optimizer.zero_grad()
            p1 = batch[0].to(device)
            p2 = batch[1].to(device)

            true_label = p1.label.reshape((1,-1))

            p1_go_label = p1.GO_label.reshape((1,-1))
            p2_go_label = p2.GO_label.reshape((1, -1))
            p1_location_label = p1.location_label.reshape((1,-1))
            p2_location_label = p2.location_label.reshape((1, -1))
            predicted_label,predicted_go_label1,predicted_go_label2,predicted_location_label1,predicted_location_label2 = model(p1,p2)

            predicted_label = predicted_label.reshape((1,-1))
            predicted_go_label1 = predicted_go_label1.reshape((1,-1))
            predicted_go_label2 = predicted_go_label2.reshape((1,-1))
            predicted_location_label1 = predicted_location_label1.reshape((1,-1))
            predicted_location_label2 = predicted_location_label2.reshape((1, -1))


            interaction_loss = bceloss(predicted_label, true_label)

            go_loss = bceloss(predicted_go_label1, p1_go_label)+bceloss(predicted_go_label2, p2_go_label)
            location_loss = bceloss(predicted_location_label1, p1_location_label)+bceloss(predicted_location_label2, p2_location_label)

            #train_loss = interaction_loss+go_loss+location_loss
            train_loss, weight = awl(interaction_loss, go_loss, location_loss)
            #train_loss,weight = awl(interaction_loss,location_loss)
            #train_loss = interaction_loss

            train_loss.backward()
            optimizer.step()

        # scheduler.step()

        model.eval()
        total_preds = torch.Tensor()
        total_labels = torch.Tensor()
        total_scores = torch.Tensor()

        total_test_loss = torch.Tensor([0]).to(device)
        total_go_test_loss = torch.Tensor([0]).to(device)
        total_location_test_loss = torch.Tensor([0]).to(device)
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, mininterval=0.5, desc='testing', leave=False, ncols=50)):
                p1 = batch[0].to(device)
                p2 = batch[1].to(device)

                true_label = p1.label.reshape((1, -1))

                p1_go_label = p1.GO_label.reshape((1, -1))
                p2_go_label = p2.GO_label.reshape((1, -1))
                p1_location_label = p1.location_label.reshape((1, -1))
                p2_location_label = p2.location_label.reshape((1, -1))

                predicted_label,predicted_go_label1,predicted_go_label2,predicted_location_label1,predicted_location_label2 = model(p1, p2)


                predicted_label = predicted_label.reshape((1, -1))
                predicted_go_label1 = predicted_go_label1.reshape((1, -1))
                predicted_go_label2 = predicted_go_label2.reshape((1, -1))
                predicted_location_label1 = predicted_location_label1.reshape((1, -1))
                predicted_location_label2 = predicted_location_label2.reshape((1, -1))


                test_loss = bceloss(predicted_label,true_label)

                go_test_loss = bceloss(predicted_go_label1, p1_go_label) + bceloss(predicted_go_label2, p2_go_label)
                location_test_loss = bceloss(predicted_location_label1,p1_location_label)+bceloss(predicted_location_label2,p2_location_label)

                total_go_test_loss = total_go_test_loss + go_test_loss
                total_test_loss = total_test_loss + test_loss
                total_location_test_loss = total_location_test_loss + location_test_loss


                predicted_label_round = torch.round(predicted_label)
                total_preds = torch.cat((total_preds, predicted_label_round.cpu()), 1)
                total_labels = torch.cat((total_labels, true_label.cpu()), 1)
                total_scores = torch.cat((total_scores, predicted_label.cpu()), 1)


        test_acc, test_prec, test_recall, test_f1, test_auc, con_matrix,test_spec,test_mcc = cal_metric(total_preds,total_labels,total_scores,args)

        if test_f1>= best_test_f1:
            best_test_f1 = test_f1
            best_record = [float(test_acc.item()),float(test_prec.item()),float(test_recall.item()),float(test_spec.item()),float(test_f1.item()),float(test_mcc.item()),float(test_auc.item())]

        print('Epoch '+str(e+1)+':\t','\tTest loss:',total_test_loss.item()/batch_idx,'\tAcc:',round(float(test_acc.item()),5),'\tPre:',round(float(test_prec.item()),5),'\tSen:',round(float(test_recall.item()),5),'\tSpe:',round(float(test_spec.item()),5),'\tF1:',round(float(test_f1.item()),5),'\tMCC:',round(float(test_mcc.item()),5),'\tAUC:',round(float(test_auc.item()),5),'\tgo_loss:',round(float(total_go_test_loss.item()/batch_idx),5),'\tloc_loss:',round(float(total_location_test_loss.item()/batch_idx),5))

        print()
    return best_record

def load_files(dataset='yeast'):
    # if dataset == 'yeast':
    #     split_mark = '.'
    # elif dataset == 'multi_class':
    split_mark = '.npy'


    contact_path = './data/'+dataset+'/dictionary/contact_map_'+args.distance+'/'
    contact_files = os.listdir(contact_path)
    contact_items = {}
    for i in tqdm(contact_files, desc='loading protein graph', leave=False):
        uid = i.split(split_mark)[0]
        contact_items[uid] = np.load(contact_path + i)

    lm_path = './data/'+dataset+'/dictionary/pretrained-emb-'+args.lm+'/'
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


def train(args):
    device = torch.device('cuda:' + args.device)
    items = load_dataset(dataset=args.dataset, mark='train', args=args)
    contact_dic, lm_dic, GO_dic, location_dic = load_files(dataset=args.dataset)
    random.seed(7)
    random.shuffle(items)
    fold_num = int(math.floor(len(items) / 5))
    fold1,fold2,fold3,fold4,fold5 = items[:fold_num*1],items[fold_num*1:fold_num*2],items[fold_num*2:fold_num*3],items[fold_num*3:fold_num*4],items[fold_num*4:fold_num*5]
    folds = [fold1,fold2,fold3,fold4,fold5]
    all_results = []
    for i in range(5):
        print('This is the ',i+1,'th fold...\n')
        train_items = []
        valid_items = folds[i]
        folds_copy = folds.copy()
        folds_copy.remove(valid_items)
        for single_fold in folds_copy:
            train_items.extend(single_fold)


        train_data = PPIDataset(data=train_items, contact_dic=contact_dic, lm_dic=lm_dic, GO_dic=GO_dic,location_dic=location_dic)
        valid_data = PPIDataset(data=valid_items, contact_dic=contact_dic, lm_dic=lm_dic, GO_dic=GO_dic,
                               location_dic=location_dic)
        dataset_train = DataLoader(train_data, shuffle=True, batch_size=args.batch_size, collate_fn=collate,
                                   drop_last=False, num_workers=6)
        dataset_valid = DataLoader(valid_data, shuffle=False, batch_size=args.batch_size, collate_fn=collate,
                                  drop_last=False, num_workers=6)
        model = nnModel(feature_dim=args.feat_dim, heads=args.heads).to(device)

        best_record = train_nn(model=model, train_loader=dataset_train, test_loader=dataset_valid, device=device, args=args)

        all_results.append(np.array(best_record).reshape((1,-1)))
    all_results = np.vstack(all_results)
    mean_values = np.mean(all_results,axis=0)
    std_values = np.std(all_results, axis=0)
    print(mean_values)
    print(std_values)
    with open('./data/yeast/dictionary/models/5-Folds/5_fold_'+args.lm+'_'+args.distance+'.txt','a') as f:
        f.write(str(mean_values)+'\n\n'+str(std_values))








if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='yeast', help="dataset:yeast/multi_class/multi_species")
    parser.add_argument('--identity', type=str, default='25', help="any/01/10/25/40")
    parser.add_argument('--distance', type=str, default='8.0', help="distance threshold (6.0/8.0/10.0) for contact map")
    parser.add_argument('--lm', type=str, default='esm-2', help="protein language model (esm-2/ProtBert-BFD/ProtT5-XL-UniRef50)")
    parser.add_argument('--device', type=str, default='3', help="cuda device.")
    parser.add_argument('--feat_dim', type=int, default=1280, help="the dimension of feature.1280/1024")
    parser.add_argument('--heads', type=int, default=1, help="attention heads.")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size.")
    args = parser.parse_args()
    print(args)
    train(args)
