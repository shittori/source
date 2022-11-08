'''2022/10/28 TaiyoIto
目的：2値のROCカーブを取得したい
参考：渡辺君からいただいた，ROCカーブ
'''
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
from sklearn import metrics
def main():
    im_true = cv2.imread('./ROC/TM-1300_mask_de_triming.bmp',-1)
    im_prob = np.load('./ROC/prob_acc.npy')
    im_pred = np.load('./ROC/pred_acc.npy')
    im_prob = (im_prob*255).astype('int32')
    true_list = list(im_true.flatten())
    prob_list = list(im_prob.flatten())
    # pred_list = list(im_pred.flatten())

    c = zip(prob_list,true_list)
    sort_list=sorted(c,reverse=False)
    prob_label,test_label = zip(*sort_list)

    print('test_label', test_label)
    fpr_list = []
    recall_list = []
    #listじゃないと正常に動作しない→基はtuple
    test_label = list(test_label)
    prob_list = list(prob_label)
    print('prob_label', prob_list)
    # [0~0.1,0.2,0.3というようにnumpy配列を作成]
    th_range = np.arange(max(test_label), 0, -1.0).astype('int32')
    for th in th_range:
        pre_label = list()
        for ind in range(len(test_label)):
            if prob_label[ind] >= th:
                #条件を満たしたら0，満たさなかったら(満たさない方が良い)255をappendしている
                pre_label.append(0)
            else:
                pre_label.append(255)
        cm = metrics.confusion_matrix(test_label, pre_label)
        tp = cm[0, 0]
        fn = cm[0, 1]
        fp = cm[1, 0]
        tn = cm[1, 1]
        fpr = fp.__truediv__(fp.__add__(tn)).astype('float64')
        recall = tp.__truediv__(fn.__add__(tp)).astype('float64')
        fpr_list.append(fpr)
        recall_list.append(recall)
    print("fpr list", fpr_list)
    print("recall_list", recall_list)
    plot_roc(fpr_list,recall_list)
    return fpr_list, recall_list


def plot_roc(fpr, tpr):
    plt.figure()
    plt.plot(fpr, tpr, clip_on=False)
    plt.title("ROC curve")
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()
    print("AUC", metrics.auc(fpr, tpr))
    print()

if __name__== '__main__':

    main()
