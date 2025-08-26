# -*- coding: UTF-8 -*-
import os
import random
import shutil
from glob import glob  
import pandas as pd
import numpy as np
import argparse
import yaml
#from utils.common import logger
from pathlib import Path  
import time

from sklearn.model_selection import StratifiedKFold  

#with open(os.path.join(os.getcwd(), 'config/config.yml'), 'r', encoding='utf8') as fs:  
    #cfg = yaml.load(fs, Loader=yaml.FullLoader)  
#K = cfg['k']  
K=5
available_policies = {}  

save_paths ={"halves":os.getcwd()+f"/data-tmb/",
            "trisection":os.getcwd()+f"/data-tmb/"}

def allDataToTrain(X, y, divisionMethod):
    train_data = []
    for (p, label) in zip(X, y):
        for img in glob(p+"/*"):
            train_data.append((img, label))

    pdf = pd.DataFrame(train_data, columns=['img', 'label']).sort_values(by=['label'], ascending=True).reset_index(drop = True)
    # Get the smallest number of image in each category
    min_num = min(pdf['label'].value_counts())

    # Random downsampling
    index = []
    for i in range(2):
        if i == 0:
            start = 0
            end = pdf['label'].value_counts()[i]
        else:
            start = end
            end = end + pdf['label'].value_counts()[i]
        index = index + random.sample(range(start, end), min_num)

    pdf = pdf.iloc[index].reset_index(drop = True)
    # Shuffle
    pdf = pdf.reindex(np.random.permutation(pdf.index)).reset_index(drop = True)
    pdf.to_csv(save_paths[divisionMethod] + f"train.csv", index=None, header=None)

def useCrossValidation(X, y, divisionMethod):
    skf = StratifiedKFold(n_splits=K, shuffle=True)

    for fold, (train, test) in enumerate(skf.split(X, y)):
        print(fold)
        print((train, test))


        train_data = []
        test_data  = []

        train_set, train_label = pd.Series(X).iloc[train].tolist(), pd.Series(y).iloc[train].tolist()
        test_set, test_label   = pd.Series(X).iloc[test].tolist(), pd.Series(y).iloc[test].tolist()


        for (data, label) in zip(train_set, train_label):
            print(data, label)
            for img in glob(data+'/*'):
                train_data.append((img, label))

        for (data, label) in zip(test_set, test_label):
            for img in glob(data+'/*'):
                test_data.append((img, label))

        pdf = pd.DataFrame(train_data, columns=['img', 'label']).sort_values(by=['label'], ascending=True).reset_index(drop = True)

        # Get the smallest number of image in each category
        min_num = min(pdf['label'].value_counts())


        # Random downsampling
        index = []
        for i in range(2):
            if i == 0:
                start = 0
                end = pdf['label'].value_counts()[i]
                # print(end)
            else:
                start = end
                # print(end)
                end = end + pdf['label'].value_counts()[i]
                # print('s',end)
            index = index + random.sample(range(start, end), min_num)
            #print(index)

        pdf = pdf.iloc[index].reset_index(drop = True)

        pdf1 = pd.DataFrame(test_data, columns=['img', 'label']).sort_values(by=['label'], ascending=True).reset_index(
            drop=True)

        # Get the smallest number of image in each category
        min_num1 = min(pdf1['label'].value_counts())

        # Random downsampling
        index = []
        for i in range(2):
            if i == 0:
                start = 0
                end = pdf1['label'].value_counts()[i]
                # print(end)
            else:
                start = end
                # print(end)
                end = end + pdf1['label'].value_counts()[i]
                # print('s',end)
            index = index + random.sample(range(start, end), min_num1)
            # print(index)

        pdf1 = pdf1.iloc[index].reset_index(drop=True)

        # Shuffle
        pdf = pdf.reindex(np.random.permutation(pdf.index)).reset_index(drop = True)
        print(save_paths[divisionMethod] + f"train_{fold}.csv")
        pdf.to_csv(save_paths[divisionMethod] + f"train_{fold}.csv", index=None, header=None)

        # pdf1 = pd.DataFrame(test_data)
        pdf1 = pdf1.reindex(np.random.permutation(pdf1.index)).reset_index(drop=True)
        print(save_paths[divisionMethod] + f"test_{fold}.csv")
        pdf1.to_csv(save_paths[divisionMethod] + f"test_{fold}.csv", index=None, header=None)


def main(srcImg, label, divisionMethod, isCrossValidation=True, shuffle=True):
    assert os.path.exists(label), "Error: tag file does not exist"  
    assert Path(label).suffix == '.csv', "Error: The label file needs to be a csv file"  

    try:
        df = pd.read_csv(label, usecols=["long_id", "label"])
    except :
        print("Error: TCGA_ID or TMB20 column information not found in file")
    
    img_dir = glob(os.path.join(srcImg, '*'))
    xml_file_seq = [img.split('/')[-2] for img in img_dir]

    tmbh_label_seq = [getattr(row, 'long_id') for row in df.itertuples() if getattr(row, 'label') == 1]
    tmbl_label_seq = [getattr(row, 'long_id') for row in df.itertuples() if getattr(row, 'label') == 0]
    #tp53h_label_seq = [getattr(row, 'ID') for row in df.itertuples() if getattr(row, 'TP53') == 1]
    #tp53l_label_seq = [getattr(row, 'ID') for row in df.itertuples() if getattr(row, 'TP53') == 0]
    #print(len(tmbh_label_seq))
    #print(len(tmbl_label_seq))
    #print(len(tp53h_label_seq))
    #print(len(tp53l_label_seq))
    #quit()
    
    assert tmbh_label_seq != 0, "Error: Abnormal data distribution"
    assert tmbl_label_seq != 0, "Error: Abnormal data distribution"

    X  = []
    y  = []

    for tmbh in tmbh_label_seq:
        if os.path.join(srcImg, tmbh) in img_dir:
            # print(os.path.join(srcImg, tmbh))
            X.append(os.path.join(srcImg, tmbh))
            y.append(1)
    for tmbl in tmbl_label_seq:
        if os.path.join(srcImg, tmbl) in img_dir:
            X.append(os.path.join(srcImg, tmbl))
            y.append(0)

    if isCrossValidation:
        useCrossValidation(X, y, divisionMethod)  
    else:
        allDataToTrain(X, y, divisionMethod)
 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script',
                                     epilog="authorized by geneis ")
    parser.add_argument('--stained_tiles_home', type=str, default="/data/data/TCGA/colonrect/Color_Normalization/")
    parser.add_argument('--label_dir_path', type=str, default="/home/hanjy/DTFD/COAD/TMB/TCGA-COAD-TMB-LABEL.csv")
    parser.add_argument('--divisionMethod', type=str, default="halves")
    parser.add_argument("--isCrossValidation", type=bool, default=True)
    args = parser.parse_args()
    main(args.stained_tiles_home, args.label_dir_path,args.divisionMethod, args.isCrossValidation)
    
