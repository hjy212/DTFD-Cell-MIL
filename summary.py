import pandas as pd
import os

# 设置文件路径和输出文件名
file_path = '/home/hanjy/DTFD/STAD/dtfd_mil_cell/result/stadyuan1/'
output_file = '/home/hanjy/DTFD/STAD/dtfd_mil_cell/result/stadyuan1/summary.csv'
# file_path = '/home/hanjy/DTFD/STAD/dtfd_mil_cell/result/coadyuan/'
# output_file = '/home/hanjy/DTFD/STAD/dtfd_mil_cell/result/coadyuan/summary.csv'
# 初始化一个空的DataFrame来存储汇总结果
summary_df = pd.DataFrame(columns=['threshold', 'accuracy', 'auc_value', 'precision', 'recall', 'fscore'])

# 遍历所有的fold文件
for i in range(5):
    file_name = f'Fold{i}_threshold_result.csv'
    file_path_full = os.path.join(file_path, file_name)

    # 读取CSV文件
    df = pd.read_csv(file_path_full)

    if summary_df.empty:
        summary_df['threshold'] = df['threshold'].unique()
        for col in ['accuracy', 'auc_value', 'precision', 'recall', 'fscore']:
            summary_df[col] = pd.NA  # 使用pd.NA来表示缺失值（需要pandas 1.0.0及以上版本）

        summary_df.set_index('threshold', inplace=True)

    df.set_index('threshold', inplace=True)

    for col in ['accuracy', 'auc_value', 'precision', 'recall', 'fscore']:
        summary_df[col] = summary_df[col].combine_first(df[col].mean(level=0))  # level=0表示在索引级别上计算平均值

    if i == 0:
        for col in ['accuracy', 'auc_value', 'precision', 'recall', 'fscore']:
            summary_df[col] = df[col]  # 先不除以fold数量
    else:
        for col in ['accuracy', 'auc_value', 'precision', 'recall', 'fscore']:
            summary_df[col] += df[col]  # 累加值

for col in ['accuracy', 'auc_value', 'precision', 'recall', 'fscore']:
    summary_df[col] /= 5  # 计算平均值

summary_df.reset_index(inplace=True)

# 将汇总结果保存到新的CSV文件中
summary_df.to_csv(os.path.join(file_path, output_file), index=False)