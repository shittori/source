'''2022/10/28 TaiyoIto
目的：2値のROCカーブを取得したい
参考：渡辺君からいただいた，ROCカーブ
2022/11/10 TaiyoIto
add:正解２値画像と予測画像を入力すると，pixel単位のROCカーブを算出する
'''
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
from sklearn import metrics
import pandas as pd
import csv
def main():
    im_true = cv2.imread('./ROC/data/test_true.bmp',-1)
    im_true = cv2.cvtColor(im_true,cv2.COLOR_BGR2GRAY)
    _,im_true = cv2.threshold(im_true,128,255,cv2.THRESH_BINARY)
    im_prob = cv2.imread('./ROC/data/test_prob2.bmp',-1)
    im_prob = cv2.cvtColor(im_prob, cv2.COLOR_BGR2GRAY).astype('float32')
    # _, im_prob = cv2.threshold(im_prob, 128, 255, cv2.THRESH_BINARY)

    #numpy読み込み時に使用
    # im_prob = np.load('./ROC/prob_acc.npy')
    # im_pred = np.load('./ROC/pred_acc.npy')
    #test画像使用中は，オフでお願いします

    # im_prob = (im_prob*255.0).astype('float32')

    true_list = list(im_true.flatten())
    prob_list = list(im_prob.flatten())
    # pred_list = list(im_pred.flatten())
    # df = pd.DataFrame(im_prob)
    # df.to_csv('./ROC/prob_acc.csv')
    c = zip(prob_list,true_list)
    # sort_list=sorted(c,reverse=False)
    prob_label,test_label = zip(*c)

    print('test_label', test_label)
    fpr_list = []
    recall_list = []
    #listじゃないと正常に動作しない→基はtuple
    test_label = list(test_label)
    prob_list = list(prob_label)
    # test_label = test_label[:200000]
    # prob_label = prob_label[:200000]
    print('prob_label', prob_list)
    # [0~0.1,0.2,0.3というようにnumpy配列を作成]
    th_range = np.arange(max(test_label), 0, -1.0).astype('int32')

    for th in th_range:
        pre_label = list()
        for ind in range(len(test_label)):
            check = prob_label[ind]
            check_test = test_label[ind]
            if prob_label[ind] >= th:

                #条件を満たしたら0，満たさなかったら(満たさない方が良い)255をappendしている
                pre_label.append(255)
                # pre_label.append(255)
            else:

                pre_label.append(0)
                # pre_label.append(0)
        A = np.unique(np.array(test_label))
        B = np.unique(np.array(pre_label))

        cm = metrics.confusion_matrix(test_label, pre_label,labels=[255,0])
        tp = cm[0, 0]
        fn = cm[0, 1]
        fp = cm[1, 0]
        tn = cm[1, 1]
        # fpr = fp.__truediv__(fp.__add__(tn)).astype('float64')
        fpr = (fp / (tn + fp)).astype('float64')
        # recall = tp.__truediv__(fn.__add__(tp)).astype('float64')
        recall = (tp / (tp + fn)).astype('float64')
        fpr_list.append(fpr)
        recall_list.append(recall)
        print('roop', int(th))
    with open('./ROC/recall_prelabel.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(list(np.unique(np.array(fpr_list))))
        writer.writerow(list(np.unique(np.array(recall_list))))
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
