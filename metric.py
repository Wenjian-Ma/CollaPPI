from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score,confusion_matrix,roc_curve, auc
import torch
import matplotlib.pyplot as plt
import numpy as np
def cal_metric(pred,label,score,args):
    pred = torch.squeeze(pred)
    label = torch.squeeze(label)
    score = torch.squeeze(score)
    if args.dataset == 'yeast' or args.dataset == 'multi_species':
        average = 'binary'
        # multi_class = 'raise'
    elif args.dataset == 'multi_class':
        average = 'weighted'
        test_auc=None
        con_matrix=None
        test_spec=None
        test_mcc=None
    test_acc = accuracy_score(label, pred)#
    test_prec = precision_score(label, pred,average=average)#,average='micro'
    test_recall = recall_score(label, pred,average=average)#
    test_f1 = f1_score(label, pred,average=average)#
    if args.dataset == 'yeast' or args.dataset == 'multi_species':
        test_auc = roc_auc_score(label, score)#pred
        con_matrix = confusion_matrix(label, pred)
        test_spec = con_matrix[0][0] / (con_matrix[0][0] + con_matrix[0][1])
        test_mcc = (con_matrix[0][0] * con_matrix[1][1] - con_matrix[0][1] * con_matrix[1][0]) / (((con_matrix[1][1] +
    con_matrix[0][1]) * (con_matrix[1][1] +con_matrix[1][0]) * (con_matrix[0][0]+con_matrix[0][1])*(con_matrix[0][0] +con_matrix[1][0])) ** 0.5)
    return test_acc,test_prec,test_recall,test_f1,test_auc,con_matrix,test_spec,test_mcc


def cal_metric_test(pred,label,score,args):
    pred = torch.squeeze(pred)
    label = torch.squeeze(label)
    score = torch.squeeze(score)
    if args.dataset == 'yeast' or args.dataset == 'multi_species':
        average = 'binary'
        # multi_class = 'raise'
    elif args.dataset == 'multi_class':
        average = 'weighted'
        test_auc=None
        con_matrix=None
        test_spec=None
        test_mcc=None
    np.save('/home/sgzhang/perl5/MV-PPI/data/AUROC_Fig/' + args.dataset + '/label.npy',
            np.array(label))
    np.save('/home/sgzhang/perl5/MV-PPI/data/AUROC_Fig/' + args.dataset + '/score.npy',
            np.array(score))
    test_acc = accuracy_score(label, pred)#
    test_prec = precision_score(label, pred,average=average)#,average='micro'
    test_recall = recall_score(label, pred,average=average)#
    test_f1 = f1_score(label, pred,average=average)#
    if args.dataset == 'yeast' or args.dataset == 'multi_species':

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