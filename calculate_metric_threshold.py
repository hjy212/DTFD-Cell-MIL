import os.path

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
import seaborn as sns

def main():
    output_dir = './result/stadyuan1'
    # output_dir = './result/coadyuan'
    thresholds = [0.1, 0.15,0.16,0.17,0.18,0.19, 0.2, 0.25, 0.3, 0.4, 0.5]

    for fold in range(5):
        path = f'/home/hanjy/DTFD/STAD/dtfd_mil_cell/result/stadyuan1/stadyuan1Fold_{fold}_Test.probas.xlsx'
        # path = f'/home/hanjy/DTFD/STAD/dtfd_mil_cell/result/coadyuan/coadyuanFold_{fold}_Test.probas.xlsx'
        prob_df = pd.read_excel(path)
        print(prob_df)
        y_true = prob_df['true_label'].values
        y_pred_probs = prob_df[1]

        result = {'threshold': thresholds,
                  'accuracy': [],
                  'auc_value': [],
                  'precision': [],
                  'recall': [],
                  'fscore': []}

        for threshold in thresholds:
            y_pred = (y_pred_probs >= threshold).astype(int)

            accuracy = accuracy_score(y_true, y_pred)
            auc_value = roc_auc_score(y_true, y_pred_probs)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            fscore = f1_score(y_true, y_pred)

            result['accuracy'].append(accuracy)
            result['auc_value'].append(auc_value)
            result['precision'].append(precision)
            result['recall'].append(recall)
            result['fscore'].append(fscore)

            cm = confusion_matrix(y_true, y_pred)

            # 使用 seaborn 绘制热图
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
            plt.title(f'Confusion Matrix at Threshold {threshold}')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')

            filename = f'fold{fold}_confusion_matrix_threshold_{threshold}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath)
            plt.close()

        result_df = pd.DataFrame(result)

        result_path = os.path.join(output_dir,f'Fold{fold}_threshold_result.csv')
        result_df.to_csv(result_path)

if __name__ == '__main__':
    main()