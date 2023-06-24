from torch import optim
import torch.nn as nn
from tqdm import tqdm
import torch
from metric import cal_metric
from loss import AutomaticWeightedLoss
def train_nn(model,train_loader,test_loader,device,args):
    Epoch = 200
    awl = AutomaticWeightedLoss(3)
    optimizer = optim.Adam([{'params':model.parameters()},{'params':awl.parameters(),'weight_decay': 0}], lr=0.001)
    #optimizer = optim.Adam(model.parameters(), lr=0.001)

    bceloss = nn.BCELoss()

    celoss = nn.CrossEntropyLoss()#NLLLoss()

    for e in range(Epoch):

        model.train()
        for batch_idx_train,batch in enumerate(tqdm(train_loader,mininterval=0.5,desc='Training',leave=False,ncols=50)):
            optimizer.zero_grad()
            p1 = batch[0].to(device)
            p2 = batch[1].to(device)
            if args.dataset == 'yeast' or args.dataset == 'multi_species':
                true_label = p1.label.reshape((1,-1))
            elif args.dataset == 'multi_class':
                true_label = p1.label.reshape((-1,1)).squeeze()
            p1_go_label = p1.GO_label.reshape((1,-1))
            p2_go_label = p2.GO_label.reshape((1, -1))
            p1_location_label = p1.location_label.reshape((1,-1))
            p2_location_label = p2.location_label.reshape((1, -1))
            predicted_label,predicted_go_label1,predicted_go_label2,predicted_location_label1,predicted_location_label2 = model(p1,p2)
            if args.dataset == 'yeast' or args.dataset == 'multi_species':
                predicted_label = predicted_label.reshape((1,-1))
            predicted_go_label1 = predicted_go_label1.reshape((1,-1))
            predicted_go_label2 = predicted_go_label2.reshape((1,-1))
            predicted_location_label1 = predicted_location_label1.reshape((1,-1))
            predicted_location_label2 = predicted_location_label2.reshape((1, -1))

            if args.dataset == 'yeast' or args.dataset == 'multi_species':
                interaction_loss = bceloss(predicted_label, true_label)
            elif args.dataset == 'multi_class':
                interaction_loss = celoss(predicted_label, true_label.long())
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
                if args.dataset == 'yeast' or args.dataset == 'multi_species':
                    true_label = p1.label.reshape((1, -1))
                elif args.dataset == 'multi_class':
                    true_label = p1.label.reshape((-1, 1)).squeeze()
                p1_go_label = p1.GO_label.reshape((1, -1))
                p2_go_label = p2.GO_label.reshape((1, -1))
                p1_location_label = p1.location_label.reshape((1, -1))
                p2_location_label = p2.location_label.reshape((1, -1))

                predicted_label,predicted_go_label1,predicted_go_label2,predicted_location_label1,predicted_location_label2 = model(p1, p2)

                if args.dataset == 'yeast' or args.dataset == 'multi_species':
                    predicted_label = predicted_label.reshape((1, -1))
                predicted_go_label1 = predicted_go_label1.reshape((1, -1))
                predicted_go_label2 = predicted_go_label2.reshape((1, -1))
                predicted_location_label1 = predicted_location_label1.reshape((1, -1))
                predicted_location_label2 = predicted_location_label2.reshape((1, -1))

                if args.dataset == 'yeast' or args.dataset == 'multi_species':
                    test_loss = bceloss(predicted_label,true_label)
                elif args.dataset == 'multi_class':
                    test_loss = celoss(predicted_label, true_label.long())
                go_test_loss = bceloss(predicted_go_label1, p1_go_label) + bceloss(predicted_go_label2, p2_go_label)
                location_test_loss = bceloss(predicted_location_label1,p1_location_label)+bceloss(predicted_location_label2,p2_location_label)

                total_go_test_loss = total_go_test_loss + go_test_loss
                total_test_loss = total_test_loss + test_loss
                total_location_test_loss = total_location_test_loss + location_test_loss

                if args.dataset == 'yeast' or args.dataset == 'multi_species':
                    predicted_label_round = torch.round(predicted_label)
                    total_preds = torch.cat((total_preds, predicted_label_round.cpu()), 1)
                    total_labels = torch.cat((total_labels, true_label.cpu()), 1)
                    total_scores = torch.cat((total_scores, predicted_label.cpu()), 1)
                elif args.dataset == 'multi_class':
                    predicted_label_round = torch.argmax(predicted_label,dim=1)
                    total_preds = torch.cat((total_preds, predicted_label_round.cpu()), 0)
                    total_labels = torch.cat((total_labels, true_label.cpu()), 0)
                    total_scores = torch.cat((total_scores,predicted_label.cpu()),0)

        test_acc, test_prec, test_recall, test_f1, test_auc, con_matrix,test_spec,test_mcc = cal_metric(total_preds,total_labels,total_scores,args)


        if args.dataset=='yeast' or args.dataset == 'multi_species':
            model_name = '-'.join(
                [str(e + 1), str(test_acc.item())[:7], str(test_prec.item())[:7], str(test_recall.item())[:7],
                 str(test_spec.item())[:7], str(test_f1.item())[:7], str(test_mcc.item())[:7],
                 str(test_auc.item())[:7]])
            print('Epoch '+str(e+1)+':\t','\tTest loss:',total_test_loss.item()/batch_idx,'\tAcc:',round(float(test_acc.item()),5),'\tPre:',round(float(test_prec.item()),5),'\tSen:',round(float(test_recall.item()),5),'\tSpe:',round(float(test_spec.item()),5),'\tF1:',round(float(test_f1.item()),5),'\tMCC:',round(float(test_mcc.item()),5),'\tAUC:',round(float(test_auc.item()),5),'\tgo_loss:',round(float(total_go_test_loss.item()/batch_idx),5),'\tloc_loss:',round(float(total_location_test_loss.item()/batch_idx),5))
        elif args.dataset == 'multi_class':
            model_name = '-'.join(
                [str(e + 1), str(test_acc.item())[:7], str(test_prec.item())[:7], str(test_f1.item())[:7]])
            print('Epoch ' + str(e + 1) + ':\t', '\tTest loss:', total_test_loss.item() / batch_idx, '\tAcc:',
                  round(float(test_acc.item()), 5), '\tPre:', round(float(test_prec.item()), 5), '\tF1:',
                  round(float(test_f1.item()), 5), '\tgo_loss:',
                  round(float(total_go_test_loss.item() / batch_idx), 5), '\tloc_loss:',
                  round(float(total_location_test_loss.item() / batch_idx), 5))
        print()
        # torch.save(model.state_dict(),'/home/sgzhang/perl5/MV-PPI/data/'+args.dataset+'/dictionary/models/'+args.device+'/'+model_name+'.pt')