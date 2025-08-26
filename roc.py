import numpy as np
import matplotlib.pyplot as plt
# from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
# from sklearn import cross_validation
import argparse
import os


def draw_roc(K, cnv=True):
    fig, ax = plt.subplots()
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for k in range(K):
        data=np.load(os.getcwd()+f'/save_model_1/resnet50_stad13ROC_{k}.npy')
        # data = np.load(os.getcwd() + f'/save_model_1/resnet50_coadyuann1roc_{k}.npy')
        # print(data)
        fpr, tpr = data

        # quit()
        # fpr, tpr, thresholds = roc_curve(np.array(y_true), np.array(score)[:, 1])
        roc_auc = auc(fpr, tpr)
        print(roc_auc)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.3f)' % (k, roc_auc))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, label='Mean ROC (AUC = {:.3f})'.format(mean_auc, std_auc), lw=2,
            color='Red')  # alpha=.6
    ax.legend(loc='lower right')
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.tick_params(labelsize=15)
    plt.title('ROC curve')
    # plt.show()
    fig.savefig(os.getcwd() + f'/STAD-MSI-roc.pdf')
    # fig.savefig(os.getcwd() + f'/result/coadyuan/COAD-MSI-roc.pdf')


def draw_pr(K, cnv=True):
    fig, ax = plt.subplots()
    tprs = []
    aucs = []
    fpr = []
    mean_fpr = np.linspace(0, 1, 100)
    mean_precision = 0.0
    mean_recall = np.linspace(0, 1, 100)
    mean_average_precision = []
    for k in range(K):
        y_true = np.load(os.getcwd() + f'/save_model_1/resnet50_stad13PR_{k}.npy')
        # y_true = np.load(os.getcwd() + f'/save_model_1/resnet50_coadyuann1pr_{k}.npy')
        # print(y_true)

        # 计算PR曲线
        precision, recall, thresholds = precision_recall_curve(y_true[0], y_true[1])
        average_precision = average_precision_score(y_true[0], y_true[1],
                                                    pos_label=1)  # auc_pr=trapz(precision, recall)
        mean_average_precision.append(average_precision)
        mean_precision = np.interp(mean_recall, precision, recall)
        # pr_auc = auc(recall, precision)
        # print(pr_auc)

        # 插值
        # interp_pr = np.interp(mean_recall, precision, recall)
        # interp_pr[0] = 1.0
        interp_tpr = np.interp(mean_fpr, precision, recall)
        interp_tpr[0] = 1.0
        fpr.append(recall)
        tprs.append(interp_tpr)
        aucs.append(average_precision)

        # 绘制PR曲线
        ax.plot(recall, precision, lw=1, label='PR fold %d (area = %0.3f)' % (k, average_precision))

    # 计算平均PR曲线和标准差
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 0.0
    mean_tpr_auc = np.mean(aucs, axis=0)
    std_tpr_auc = np.std(aucs)

    # 绘制平均PR曲线
    ax.plot(mean_recall, mean_tpr, label='Mean PR (AUC = {:.3f})'.format(mean_tpr_auc, std_tpr_auc), lw=2, color='Red')

    ax.legend(loc='lower left')

    # 绘制对角线
    ax.plot([1, 0], [0, 1], color='navy', linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.tick_params(labelsize=15)
    plt.title('Precision-Recall curve')

    # 保存图像
    fig.savefig(os.getcwd() + f'/STAD-MSI-pr.pdf')
    # fig.savefig(os.getcwd() + f'/result/coadyuan/COAD-MSI-pr.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script',
                                     epilog="authorized by geneis ")
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--cnv', type=bool, default=False)
    args = parser.parse_args()
    draw_roc(args.K, args.cnv)

    draw_pr(args.K, args.cnv)

    origirn_classfication_set = None

# #单个roc
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc
# import argparse
# import os
#
#
# def draw_roc():
#     # 加载数据
#     data_path = os.getcwd() + '/result/resnet50_3roc.npy'
#     fpr, tpr = np.load(data_path)
#
#     # 计算AUC
#     roc_auc = auc(fpr, tpr)
#     print(f'AUC: {roc_auc:.3f}')
#
#     # 绘制ROC曲线
#     plt.figure()
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
#     plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC Curve')
#     plt.legend(loc="lower right")
#     plt.show()
#
#     # 保存ROC曲线图
#     save_path = os.getcwd() + '/result_2/roc.pdf'
#     plt.savefig(save_path)
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Script to draw a single ROC curve.')
#     args = parser.parse_args()
#     draw_roc()

# #单个pr
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import precision_recall_curve, average_precision_score
# import argparse
# import os
# 
# 
# def draw_pr():
#     y_true = np.load(os.getcwd() + '/result/resnet50_3pr.npy')
#     precision, recall, thresholds = precision_recall_curve(y_true[0], y_true[1])
#     average_precision = average_precision_score(y_true[0], y_true[1],pos_label=1)
# 
#     # 绘制PR曲线
#     plt.figure()
#     plt.plot(recall, precision, lw=2, label='PR curve (area = %0.3f)' % average_precision)
# 
#     # 绘制对角线
#     plt.plot([1, 0], [0, 1], color='navy', linestyle='--')
#     plt.xlim([-0.05, 1.05])
#     plt.ylim([-0.05, 1.05])
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall Curve')
#     plt.legend(loc="best")
#     plt.tick_params(labelsize=15)
#     plt.legend(loc="lower left")
# 
#     # 保存图像
#     # 你可以根据需要修改文件路径和名称
#     output_path = os.getcwd() + '/result_2/pr.pdf'
#     plt.savefig(output_path)
# 
# 
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Plot a single Precision-Recall curve.',
#                                      epilog="Authorized by geneis")
#     # parser.add_argument('--file', type=str, required=True, help='Path to the .npy file containing y_true and score')
#     args = parser.parse_args()
#     draw_pr()
