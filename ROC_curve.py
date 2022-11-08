# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 19:50:57 2022

@author: t1181
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def roc_curve(test_label,prob_label):
    c = zip(prob_label, test_label)
    sort_list=sorted(c,reverse=False)
    prob_label,test_label = zip(*sort_list)
    
    print('test_label',test_label)
    fpr_list=[]
    recall_list=[]
    prob_list=list(prob_label)
    print('prob_label',prob_list)
    th_range = np.arange(test_label.max(), 0, -0.01).astype('float32')
    for th in th_range:
        pre_label=list()
        for ind in range(len(test_label)):
            if prob_label[ind]>=th:
                pre_label.append(0)
            else:
                pre_label.append(1)
        cm=metrics.confusion_matrix(test_label,pre_label)
        tp = cm[0, 0]
        fn = cm[0, 1]
        fp = cm[1, 0]
        tn = cm[1, 1]
        fpr=fp.__truediv__(fp.__add__(tn)).astype('float64')
        recall=tp.__truediv__(fn.__add__(tp)).astype('float64')
        fpr_list.append(fpr)
        recall_list.append(recall)
    print("fpr list",fpr_list)
    print("recall_list",recall_list)
    return fpr_list,recall_list

def plot_roc(fpr,tpr):
    plt.figure()
    plt.plot(fpr,tpr,clip_on=False)
    plt.title("ROC curve")
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()
    print("AUC",metrics.auc(fpr,tpr))