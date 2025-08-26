import sys
import os

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch

torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import json
from torch.utils.tensorboard import SummaryWriter
import pickle
import random
from Model.Attention import Attention_Gated as Attention
from Model.Attention import Attention_with_Classifier
from utils import get_cam_1d
import torch.nn.functional as F
from Model.network import Classifier_1fc, DimReduction
import numpy as np
import collections
from custom_dset import CustomDset
# from custom_dset import CustomDset
from sklearn.metrics import confusion_matrix
from utils import eval_metric, plot_confusion_matrix
from torchvision import transforms
from models.resnet_custom import resnet50_baseline, resnet18_baseline
from collections import Counter
from sklearn.metrics import roc_curve
from collections import OrderedDict

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from calculatemetrics.CalculateMetrics import predictionsReport

parser = argparse.ArgumentParser(description='abc')
testMask_dir = ''  ## Point to the Camelyon test set mask location

parser.add_argument('--name', default='abc', type=str)
parser.add_argument('--EPOCH', default=20, type=int)
parser.add_argument('--epoch_step', default='[10]', type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--isPar', default=False, type=bool)
parser.add_argument('--log_dir', default='./debug_log_finetuning', type=str)  ## log file path
parser.add_argument('--train_show_freq', default=40, type=int)
parser.add_argument('--droprate', default='0', type=float)
parser.add_argument('--droprate_2', default='0', type=float)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--batch_size_v', default=1, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_cls', default=2, type=int)
parser.add_argument('--dataset_path', default='data/', type=str)  ## dataset_path
# parser.add_argument('--mDATA0_dir_val0', default='', type=str)      ## Validation Set
# parser.add_argument('--mDATA_dir_test0', default='', type=str)         ## Test Set
parser.add_argument('--numGroup', default=1, type=int)
parser.add_argument('--total_instance', default=4, type=int)
parser.add_argument('--numGroup_test', default=1, type=int)
parser.add_argument('--total_instance_test', default=4, type=int)
parser.add_argument('--mDim', default=512, type=int)
parser.add_argument('--in_chn', default=1024, type=int, help='in_chn for DimReduction')
parser.add_argument('--grad_clipping', default=5, type=float)
parser.add_argument('--isSaveModel', action='store_false')
parser.add_argument('--Feature_extraction_weight', default='DTFD-MIL_finetuning/', type=str)
parser.add_argument('--model_plot', default='model_plot/', type=str)
parser.add_argument('--numLayer_Res', default=0, type=int)
parser.add_argument('--temperature', default=1, type=float)
parser.add_argument('--num_MeanInference', default=1, type=int)
parser.add_argument('--K', default=5, type=int)
parser.add_argument('--cnv', default=False, type=bool)
parser.add_argument('--distill_type', default='AFS', type=str)  ## MaxMinS, MaxS, AFS
parser.add_argument('--model_results', default='./model_results', type=str)
parser.add_argument('--path_model_dict_for_feature_extraction_with_k_as_bracket', default='', type=str,
                    help='Could be None, or a path with fold number k replaced to {}')
parser.add_argument('--function_str_to_generate_model_structure', default='resnet50_baseline', type=str,
                    help='The function that returns a model, currently options: resnet50_baseline, resnet18_baseline')

torch.manual_seed(32)
torch.cuda.manual_seed(32)
np.random.seed(32)
random.seed(32)
params = parser.parse_args()


def main():
    for k in range(params.K):  # range(params.K)
        print(k)

        epoch_step = json.loads(params.epoch_step)
        writer = SummaryWriter(os.path.join(params.log_dir, 'LOG', params.name))

        classifier = Classifier_1fc(params.mDim, params.num_cls, params.droprate).to(params.device)
        attention = Attention(params.mDim).to(params.device)
        dimReduction = DimReduction(params.in_chn, params.mDim, numLayer_Res=params.numLayer_Res).to(params.device)
        attCls = Attention_with_Classifier(L=params.mDim, num_cls=params.num_cls, droprate=params.droprate_2).to(
            params.device)

        if params.isPar:
            classifier = torch.nn.DataParallel(classifier)
            attention = torch.nn.DataParallel(attention)
            dimReduction = torch.nn.DataParallel(dimReduction)
            attCls = torch.nn.DataParallel(attCls)

        ce_cri = torch.nn.CrossEntropyLoss(reduction='none').to(params.device)
        # ce_cri = Ploy1_cross_entropy(logits, labels, epsilon=8.0).to(params.device)
        if not os.path.exists(params.log_dir):
            os.makedirs(params.log_dir)
        log_dir = os.path.join(params.log_dir, f'log_{k}_zryh.txt')
        save_dir = os.path.join(params.log_dir, f'best_model_zryh{k}.pth')
        z = vars(params).copy()
        with open(log_dir, 'a') as f:
            f.write(json.dumps(z))
        log_file = open(log_dir, 'a')

        # print(bags[])

        SlideNames_train, FeatList_train, Label_train = load_data(params.cnv, f'{params.dataset_path}train_{k}.csv',
                                                                  A='train',
                                                                  save_dir=f'{params.Feature_extraction_weight}train_{k}.pkl',
                                                                  k=k)
        # print(Counter(Label_train))
        SlideNames_val, FeatList_val, Label_val = load_data(params.cnv, f'{params.dataset_path}test_{k}.csv',
                                                            A='test',
                                                            save_dir=f'{params.Feature_extraction_weight}val_{k}.pkl',
                                                            k=k)

        # print(Counter(Label_val))
        SlideNames_test, FeatList_test, Label_test = load_data(params.cnv, f'{params.dataset_path}test_{k}.csv',
                                                               A='test',
                                                               save_dir=f'{params.Feature_extraction_weight}test_{k}.pkl',
                                                               k=k)

        print_log(
            f'training slides: {len(SlideNames_train)}, validation slides: {len(SlideNames_val)}, test slides: {len(SlideNames_test)}',
            log_file)

        trainable_parameters = []
        trainable_parameters += list(classifier.parameters())
        trainable_parameters += list(attention.parameters())
        trainable_parameters += list(dimReduction.parameters())

        optimizer_adam0 = torch.optim.Adam(trainable_parameters, lr=params.lr, weight_decay=params.weight_decay)
        optimizer_adam1 = torch.optim.Adam(attCls.parameters(), lr=params.lr, weight_decay=params.weight_decay)

        scheduler0 = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam0, epoch_step, gamma=params.lr_decay_ratio)
        scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam1, epoch_step, gamma=params.lr_decay_ratio)

        best_auc = 0
        best_epoch = -1
        test_auc = 0

        for ii in range(params.EPOCH):

            for param_group in optimizer_adam1.param_groups:
                curLR = param_group['lr']
                print_log(f' current learn rate {curLR}', log_file)

            train_attention_preFeature_DTFD(classifier=classifier, dimReduction=dimReduction, attention=attention,
                                            UClassifier=attCls,
                                            mDATA_list=(SlideNames_train, FeatList_train, Label_train), ce_cri=ce_cri,
                                            optimizer0=optimizer_adam0, optimizer1=optimizer_adam1, epoch=ii,
                                            params=params, f_log=log_file, writer=writer, numGroup=params.numGroup,
                                            total_instance=params.total_instance, distill=params.distill_type)
            print_log(f'>>>>>>>>>>> Validation Epoch: {ii}', log_file)
            auc_val = test_attention_DTFD_preFeat_MultipleMean(classifier=classifier, dimReduction=dimReduction,
                                                               attention=attention,
                                                               UClassifier=attCls,
                                                               mDATA_list=(SlideNames_val, FeatList_val, Label_val),
                                                               criterion=ce_cri, epoch=ii, params=params,
                                                               f_log=log_file, writer=writer,
                                                               numGroup=params.numGroup_test,
                                                               total_instance=params.total_instance_test,
                                                               distill=params.distill_type)
            print_log(' ', log_file)
            print_log(f'>>>>>>>>>>> Test Epoch: {ii}', log_file)
            tauc, prob_1, label_1, gPred_1, gt_1, Print_Data = test_attention_DTFD_preFeat_MultipleMean(
                classifier=classifier, dimReduction=dimReduction, attention=attention,
                UClassifier=attCls, mDATA_list=(SlideNames_test, FeatList_test, Label_test), criterion=ce_cri, epoch=ii,
                params=params, f_log=log_file, writer=writer, numGroup=params.numGroup_test,
                total_instance=params.total_instance_test, distill=params.distill_type)
            print_log(' ', log_file)

            # if ii > int(params.EPOCH*0.5):
            if tauc > best_auc:
                best_auc = tauc
                best_epoch = ii
                test_auc = tauc
                y_pre = prob_1.cpu()
                y_true = label_1.cpu()
                # print(y_pre)
                # print(y_true)
                # cm = confusion_matrix(y_true, y_pre)
                # plot_confusion_matrix(cm, f'result_1/confusion_matrix_resnet50_bestmodel_coadMM_{k}.pdf',title='confusion matrix')
                fpr, tpr, threshold = roc_curve(gt_1.cpu().numpy(), gPred_1.detach().cpu().numpy(), pos_label=1)
                roc_value = [fpr, tpr]
                np.save(os.getcwd() + f'/save_model_1/resnet50_zryhroc_{k}.npy', np.array(roc_value))
                roc_value1 = [gt_1.cpu().numpy(), gPred_1.detach().cpu().numpy()]
                np.save(os.getcwd() + f'/save_model_1/resnet50_zryhpr_{k}.npy', np.array(roc_value1))
                # test=np.load(os.getcwd()+f'/save_model/roc_value_resnet18bestmodel.npy')
                # print(test)
                # Save_to_Csv(data=Print_Data, file_name=f'id_label_COAD-CELL15msi_{k}', Save_format='csv',Save_type='col',path = params.model_results)
                '''
                Previous code to calculate metrics
                cm = confusion_matrix(y_true, y_pre)
                if not os.path.isdir(params.model_results):
                    os.makedirs(params.model_results)
                plot_confusion_matrix(cm, os.path.join(params.model_results, f'confusion_matrix_resnet50_bestmodel_{k}.pdf'), title='confusion matrix')
                fpr, tpr, threshold = roc_curve(gt_1.cpu().numpy(), gPred_1.detach().cpu().numpy(), pos_label=1)
                roc_value=[fpr, tpr]
                np.save(os.path.join(params.model_results, f'resnet50_{k}_xiewenzhangyong.npy'), np.array(roc_value))
                #test=np.load(os.getcwd()+f'/save_model/roc_value_resnet18bestmodel.npy')
                #print(test)
                Save_to_Csv(data=Print_Data, file_name=f'id_label_TMB_{k}', Save_format='csv', Save_type='col',path = params.model_results)
                '''
                # New code to calculate metrics.
                predictionsReport(gPred_1.detach().cpu().numpy(), gt_1.cpu().numpy(),
                                  write_to_disk_prefix=os.path.join(params.model_results,
                                                                    'zryhFold_{}_Test'.format(k)))

            if params.isSaveModel:
                tsave_dict = {
                    'classifier': classifier.state_dict(),
                    'dim_reduction': dimReduction.state_dict(),
                    'attention': attention.state_dict(),
                    'att_classifier': attCls.state_dict()
                }
                torch.save(tsave_dict, save_dir)

            print_log(f' test auc: {test_auc}, from epoch {best_epoch}', log_file)

        scheduler0.step()
        scheduler1.step()


def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def test_attention_DTFD_preFeat_MultipleMean(mDATA_list, classifier, dimReduction, attention, UClassifier, epoch,
                                             criterion=None, params=None, f_log=None, writer=None, numGroup=3,
                                             total_instance=3, distill='MaxMinS'):
    classifier.eval()
    attention.eval()
    dimReduction.eval()
    UClassifier.eval()

    SlideNames, FeatLists, Label = mDATA_list

    # cnv
    cnv_feature = pd.read_csv('/home/hanjy/DTFD/STAD/MSI/TCGA-STAD-MSI-CELL.csv')
    print(cnv_feature.columns)
    peoples = [i for i in cnv_feature.ID]
    features = [cnv_feature[i] for i in cnv_feature.columns[1:]]
    min_max_scaler = MinMaxScaler()
    cnv_features = min_max_scaler.fit_transform(features)

    instance_per_group = total_instance // numGroup

    test_loss0 = AverageMeter()
    test_loss1 = AverageMeter()

    gPred_0 = torch.FloatTensor().to(params.device)
    gt_0 = torch.LongTensor().to(params.device)
    gPred_1 = torch.FloatTensor().to(params.device)
    gt_1 = torch.LongTensor().to(params.device)
    slide_names_test_order = []
    with torch.no_grad():

        numSlides = len(SlideNames)
        numIter = numSlides // params.batch_size_v
        tIDX = list(range(numSlides))

        for idx in range(numIter):  # 遍历每一个slide

            tidx_slide = tIDX[idx * params.batch_size_v:(idx + 1) * params.batch_size_v]
            slide_names = [SlideNames[sst] for sst in tidx_slide]
            tlabel = [Label[sst] for sst in tidx_slide]
            label_tensor = torch.LongTensor(tlabel).to(params.device)
            batch_feat = [FeatLists[sst].to(params.device) for sst in tidx_slide]  # 全部针对一个slide

            for tidx, tfeat in enumerate(batch_feat):
                tslideName = slide_names[tidx]
                tslideLabel = label_tensor[tidx].unsqueeze(0)
                midFeat = dimReduction(tfeat)  # 特征维数约简

                AA = attention(midFeat, isNorm=False).squeeze(0)  ## N

                allSlide_pred_softmax = []

                for jj in range(params.num_MeanInference):  # params.num_MeanInference=1

                    feat_index = list(range(tfeat.shape[0]))
                    random.shuffle(feat_index)
                    if tfeat.shape[0] >= 10:  # tslideLabel==1 and
                        numGroup = 5
                    else:
                        numGroup = 1
                    index_chunk_list = np.array_split(np.array(feat_index), numGroup)
                    index_chunk_list = [sst.tolist() for sst in index_chunk_list]

                    slide_d_feat = []
                    slide_sub_preds = []
                    slide_sub_labels = []

                    for tindex in index_chunk_list:
                        slide_sub_labels.append(tslideLabel)
                        idx_tensor = torch.LongTensor(tindex).to(params.device)
                        tmidFeat = midFeat.index_select(dim=0, index=idx_tensor)

                        tAA = AA.index_select(dim=0, index=idx_tensor)
                        tAA = torch.softmax(tAA, dim=0)
                        tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
                        tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  # 1 x fs#

                        X_train_minmax = [cnv_features[:, peoples.index(i[0:8])] for i in slide_names]
                        tPredict = classifier(tattFeat_tensor.to(params.device),
                                              torch.from_numpy(np.array(X_train_minmax, dtype=np.float32)).to(
                                                  params.device))
                        # tPredict = classifier(tattFeat_tensor)  ### 1 x 2
                        slide_sub_preds.append(tPredict)

                        patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
                        patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
                        patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

                        _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)

                        if distill == 'MaxMinS':
                            topk_idx_max = sort_idx[:instance_per_group].long()
                            topk_idx_min = sort_idx[-instance_per_group:].long()
                            topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
                            d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                            slide_d_feat.append(d_inst_feat)
                        elif distill == 'MaxS':
                            topk_idx_max = sort_idx[:instance_per_group].long()
                            topk_idx = topk_idx_max
                            d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                            slide_d_feat.append(d_inst_feat)
                        elif distill == 'AFS':
                            slide_d_feat.append(tattFeat_tensor)

                    slide_d_feat = torch.cat(slide_d_feat, dim=0)
                    slide_sub_preds = torch.cat(slide_sub_preds, dim=0)
                    slide_sub_labels = torch.cat(slide_sub_labels, dim=0)

                    gPred_0 = torch.cat([gPred_0, slide_sub_preds], dim=0)
                    gt_0 = torch.cat([gt_0, slide_sub_labels], dim=0)
                    loss0 = criterion(slide_sub_preds, slide_sub_labels).mean()
                    # loss0 = Ploy1_cross_entropy(slide_sub_preds, slide_sub_labels, epsilon=7.0).mean()
                    # loss0 = Poly1FocalLoss(slide_sub_preds, slide_sub_labels, epsilon=1.0).mean()
                    test_loss0.update(loss0.item(), numGroup)

                    X_train_minmax = [cnv_features[:, peoples.index(i[0:8])] for i in slide_names]
                    gSlidePred = UClassifier(slide_d_feat.to(params.device),
                                             torch.from_numpy(np.array(X_train_minmax, dtype=np.float32)).to(
                                                 params.device))
                    # gSlidePred = UClassifier(slide_d_feat)
                    allSlide_pred_softmax.append(torch.softmax(gSlidePred, dim=1))

                slide_names_test_order.append(tslideName)
                allSlide_pred_softmax = torch.cat(allSlide_pred_softmax, dim=0)
                allSlide_pred_softmax = torch.mean(allSlide_pred_softmax, dim=0).unsqueeze(0)
                gPred_1 = torch.cat([gPred_1, allSlide_pred_softmax], dim=0)
                gt_1 = torch.cat([gt_1, tslideLabel], dim=0)

                loss1 = F.nll_loss(allSlide_pred_softmax, tslideLabel)
                test_loss1.update(loss1.item(), 1)

    gPred_0 = torch.softmax(gPred_0, dim=1)
    gPred_0 = gPred_0[:, -1]
    gPred_1 = gPred_1[:, -1]

    macc_0, mprec_0, mrecal_0, mspec_0, mF1_0, auc_0, prob0, label0 = eval_metric(gPred_0, gt_0)
    macc_1, mprec_1, mrecal_1, mspec_1, mF1_1, auc_1, prob1, label1 = eval_metric(gPred_1, gt_1)

    print_log(
        f'  First-Tier acc {macc_0}, precision {mprec_0}, recall {mrecal_0}, specificity {mspec_0}, F1 {mF1_0}, AUC {auc_0}',
        f_log)
    print_log(
        f'  Second-Tier acc {macc_1}, precision {mprec_1}, recall {mrecal_1}, specificity {mspec_1}, F1 {mF1_1}, AUC {auc_1}',
        f_log)

    Print_Data = {'slide_names_test_order': slide_names_test_order, 'Prob1': prob1.cpu(), 'label': label1.cpu()}

    writer.add_scalar(f'auc_0 ', auc_0, epoch)
    writer.add_scalar(f'auc_1 ', auc_1, epoch)

    return auc_1, prob1, label1, gPred_1, gt_1, Print_Data


def Save_to_Csv(data, file_name, path, Save_format='csv', Save_type='col'):
    # data
    # 输入为一个字典，格式： { '列名称': 数据,....}
    # 列名即为CSV中数据对应的列名， 数据为一个列表

    # file_name 存储文件的名字
    # Save_format 为存储类型， 默认csv格式， 可改为 excel
    # Save_type 存储类型 默认按列存储， 否则按行存储

    # 默认存储在当前路径下

    import pandas as pd
    import numpy as np

    Name = []
    times = 0

    if Save_type == 'col':
        for name, List in data.items():
            Name.append(name)
            if times == 0:
                Data = np.array(List).reshape(-1, 1)
            else:
                Data = np.hstack((Data, np.array(List).reshape(-1, 1)))

            times += 1

        Pd_data = pd.DataFrame(columns=Name, data=Data)

    else:
        for name, List in data.items():
            Name.append(name)
            if times == 0:
                Data = np.array(List)
            else:
                Data = np.vstack((Data, np.array(List)))

            times += 1

        Pd_data = pd.DataFrame(index=Name, data=Data)

    if Save_format == 'csv':
        Pd_data.to_csv(path + file_name + '.csv', encoding='utf-8')
    else:
        Pd_data.to_excel(path + file_name + '.xls', encoding='utf-8')


def train_attention_preFeature_DTFD(mDATA_list, classifier, dimReduction, attention, UClassifier, optimizer0,
                                    optimizer1, epoch, ce_cri=None, params=None,
                                    f_log=None, writer=None, numGroup=3, total_instance=3, distill='MaxMinS'):
    SlideNames_list, mFeat_list, Label_dict = mDATA_list

    classifier.train()
    dimReduction.train()
    attention.train()
    UClassifier.train()

    instance_per_group = total_instance // numGroup

    # cnv
    cnv_feature = pd.read_csv('/home/hanjy/DTFD/STAD/MSI/TCGA-STAD-MSI-CELL.csv')
    peoples = [i for i in cnv_feature.ID]
    features = [cnv_feature[i] for i in cnv_feature.columns[1:]]
    min_max_scaler = MinMaxScaler()
    cnv_features = min_max_scaler.fit_transform(features)

    Train_Loss0 = AverageMeter()
    Train_Loss1 = AverageMeter()

    numSlides = len(SlideNames_list)
    numIter = numSlides // params.batch_size

    tIDX = list(range(numSlides))
    random.shuffle(tIDX)

    for idx in range(numIter):

        tidx_slide = tIDX[idx * params.batch_size:(idx + 1) * params.batch_size]
        tslide_name = [SlideNames_list[sst] for sst in tidx_slide]
        tlabel = [Label_dict[sst] for sst in tidx_slide]
        label_tensor = torch.LongTensor(tlabel).to(params.device)

        for tidx, (tslide, slide_idx) in enumerate(zip(tslide_name, tidx_slide)):
            tslideLabel = label_tensor[tidx].unsqueeze(0)

            slide_pseudo_feat = []
            slide_sub_preds = []
            slide_sub_labels = []

            tfeat_tensor = mFeat_list[slide_idx]
            tfeat_tensor = tfeat_tensor.to(params.device)

            feat_index = list(range(tfeat_tensor.shape[0]))  # patch数目
            random.shuffle(feat_index)
            if tfeat_tensor.shape[0] >= 10:  # tslideLabel == 1 and
                numGroup = 5
            else:
                numGroup = 1
            index_chunk_list = np.array_split(np.array(feat_index), numGroup)
            index_chunk_list = [sst.tolist() for sst in index_chunk_list]

            for tindex in index_chunk_list:
                slide_sub_labels.append(tslideLabel)
                subFeat_tensor = torch.index_select(tfeat_tensor, dim=0,
                                                    index=torch.LongTensor(tindex).to(params.device))
                # print(subFeat_tensor.size())
                tmidFeat = dimReduction(subFeat_tensor)
                tAA = attention(tmidFeat).squeeze(0)

                tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
                tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs

                X_train_minmax = [cnv_features[:, peoples.index(i[0:60])] for i in tslide_name]
                tPredict = classifier(tattFeat_tensor, torch.from_numpy(np.array(X_train_minmax, dtype=np.float32)).to(
                    params.device))  ### 1 x 2
                # tPredict = classifier(tattFeat_tensor)  ### 1 x 2
                slide_sub_preds.append(tPredict)
                # print(tattFeats.unsqueeze(0).size())
                patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n

                patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
                patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

                _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)
                topk_idx_max = sort_idx[:instance_per_group].long()
                topk_idx_min = sort_idx[-instance_per_group:].long()
                topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)

                MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)  ##########################
                max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
                af_inst_feat = tattFeat_tensor

                if distill == 'MaxMinS':
                    slide_pseudo_feat.append(MaxMin_inst_feat)
                elif distill == 'MaxS':
                    slide_pseudo_feat.append(max_inst_feat)
                elif distill == 'AFS':
                    slide_pseudo_feat.append(af_inst_feat)

            slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)  ### numGroup x fs

            ## optimization for the first tier
            slide_sub_preds = torch.cat(slide_sub_preds, dim=0)  ### numGroup x fs
            # print(slide_sub_preds)

            slide_sub_labels = torch.cat(slide_sub_labels, dim=0)  ### numGroup
            loss0 = ce_cri(slide_sub_preds, slide_sub_labels).mean()
            # loss0 = Ploy1_cross_entropy(slide_sub_preds, slide_sub_labels, epsilon=7.0).mean()
            # loss0 = Poly1FocalLoss(slide_sub_preds, slide_sub_labels, epsilon=1.0).mean()
            optimizer0.zero_grad()
            loss0.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(dimReduction.parameters(), params.grad_clipping)
            torch.nn.utils.clip_grad_norm_(attention.parameters(), params.grad_clipping)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), params.grad_clipping)
            # optimizer0.step()

            ## optimization for the second tier

            X_train_minmax = [cnv_features[:, peoples.index(i[0:60])] for i in tslide_name]
            gSlidePred = UClassifier(slide_pseudo_feat,
                                     torch.from_numpy(np.array(X_train_minmax, dtype=np.float32)).to(params.device))
            # gSlidePred = UClassifier(slide_pseudo_feat)
            gSlidePred = gSlidePred.clone()
            tslideLabel = tslideLabel.clone()
            loss1 = ce_cri(gSlidePred, tslideLabel).mean().clone()
            # loss1 = Ploy1_cross_entropy(gSlidePred, tslideLabel,epsilon=7.0).mean()
            # loss1 = Poly1FocalLoss(gSlidePred, tslideLabel,epsilon=1.0).mean()
            optimizer1.zero_grad()
            loss1.backward()
            torch.nn.utils.clip_grad_norm_(UClassifier.parameters(), params.grad_clipping)
            optimizer0.step()
            optimizer1.step()

            Train_Loss0.update(loss0.item(), numGroup)
            Train_Loss1.update(loss1.item(), 1)

        if idx % params.train_show_freq == 0:
            tstr = 'epoch: {} idx: {}'.format(epoch, idx)
            tstr += f' First Loss : {Train_Loss0.avg}, Second Loss : {Train_Loss1.avg} '
            print_log(tstr, f_log)

    writer.add_scalar(f'train_loss_0 ', Train_Loss0.avg, epoch)
    writer.add_scalar(f'train_loss_1 ', Train_Loss1.avg, epoch)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_log(tstr, f):
    # with open(dir, 'a') as f:
    f.write('\n')
    f.write(tstr)
    print(tstr)


def reOrganize_mDATA_test(mDATA):
    tumorSlides = os.listdir(testMask_dir)
    tumorSlides = [sst.split('.')[0] for sst in tumorSlides]

    SlideNames = []
    FeatList = []
    Label = []
    for slide_name in mDATA.keys():
        SlideNames.append(slide_name)

        if slide_name in tumorSlides:
            label = 1
        else:
            label = 0
        Label.append(label)

        patch_data_list = mDATA[slide_name]
        featGroup = []
        for tpatch in patch_data_list:
            tfeat = torch.from_numpy(tpatch['feature'])
            featGroup.append(tfeat.unsqueeze(0))
        featGroup = torch.cat(featGroup, dim=0)  ## numPatch x fs
        FeatList.append(featGroup)

    return SlideNames, FeatList, Label


def reOrganize_mDATA(mDATA):
    SlideNames = []
    FeatList = []
    Label = []
    for slide_name in mDATA.keys():
        SlideNames.append(slide_name)

        if slide_name.startswith('tumor'):
            label = 1
        elif slide_name.startswith('normal'):
            label = 0
        else:
            raise RuntimeError('Undefined slide type')
        Label.append(label)

        patch_data_list = mDATA[slide_name]
        featGroup = []
        for tpatch in patch_data_list:
            tfeat = torch.from_numpy(tpatch['feature'])
            featGroup.append(tfeat.unsqueeze(0))
        featGroup = torch.cat(featGroup, dim=0)  ## numPatch x fs
        FeatList.append(featGroup)

    return SlideNames, FeatList, Label


def Ploy1_cross_entropy(logits, labels, epsilon=7.0):  #
    labels_onehot = F.one_hot(labels, num_classes=2)
    pt = torch.sum(labels_onehot * torch.nn.functional.softmax(logits, dim=-1), dim=-1)  #
    CE = F.cross_entropy(logits, labels, reduction='none')
    Poly1 = CE + epsilon * (1 - pt)
    return Poly1


def Poly1FocalLoss(logits, labels, epsilon=1.0):
    p = torch.sigmoid(logits)

    # if labels are of shape [N]
    # convert to one-hot tensor of shape [N, num_classes]

    alpha = 0.25
    gamma = 2.0
    if labels.ndim == 1:
        labels = F.one_hot(labels, num_classes=2)

    # if labels are of shape [N, ...] e.g. segmentation task
    # convert to one-hot tensor of shape [N, num_classes, ...]
    else:
        labels = F.one_hot(labels.unsqueeze(1), num_classes=2).transpose(1, -1).squeeze_(-1)

    labels = labels.to(device=logits.device,
                       dtype=logits.dtype)

    ce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    pt = labels * p + (1 - labels) * (1 - p)
    FL = ce_loss * ((1 - pt) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
        FL = alpha_t * FL

    poly1 = FL + epsilon * torch.pow(1 - pt, gamma + 1)
    return poly1


def load_data(cnv, file_address, A, save_dir, k, path_model_dict_for_feature_extraction=None,
              function_str_to_generate_model_structure='resnet50_baseline'):
    print('sss')
    if A == 'train':
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(224),  # 224
                # transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    else:
        data_transforms = {
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),  # 从中心裁剪
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    # save_dir=f'/home/wangwy/HEpreictTMBcode/database6_3/'

    # if os.path.isfile(save_dir):
    #     data = load_pickle(save_dir)
    #     return data
    # else:
    if not os.path.isdir(params.Feature_extraction_weight):
        os.makedirs(params.Feature_extraction_weight)

    sets = CustomDset(file_address, data_transforms[A])
    train_loader = torch.utils.data.DataLoader(sets, batch_size=4, shuffle=False,
                                               num_workers=4, pin_memory=True)
    bags = collections.defaultdict(list)
    bag_label = []
    SlideNames = []
    FeatList = []
    device = torch.device(params.device)
    print('loading model checkpoint')
    model = resnet50_baseline(pretrained=True)
    # device = torch.device('cuda')
    # print('loading model checkpoint')
    # print('function_str_to_generate_model_structure', function_str_to_generate_model_structure)
    # model = eval(function_str_to_generate_model_structure)(pretrained=True)
    # if (path_model_dict_for_feature_extraction is not None) \
    #     and (str(path_model_dict_for_feature_extraction).lower() != 'none') \
    #     and (path_model_dict_for_feature_extraction != ''):
    #     model_dict_to_load = torch.load(path_model_dict_for_feature_extraction, map_location=device)
    #     model.load_state_dict(model_dict_to_load, strict=True)
    model = model.to(device)

    with torch.no_grad():
        for data1 in train_loader:
            inputs, labels, names_, image_name = data1
            inputs = inputs.to(device)
            features = model(inputs)
            # print(features.size())
            features = features.cpu().numpy()
            for i in range(labels.size(0)):
                people = names_[i]
                if people not in bags.keys():
                    bags[people] = list()
                    SlideNames.append(people)
                    bag_label.append(labels[i].item())
                bags[people].append(features[i])
    SlideNames = list(bags.keys())

    # FeatList = []
    for slideName in bags.keys():
        patch_data_list = bags[slideName]
        featGroup = []
        for tpatch in patch_data_list:
            tfeat = torch.from_numpy(tpatch)
            tfeat = tfeat.unsqueeze(0)
            featGroup.append(tfeat)
        featGroup = torch.cat(featGroup, dim=0)  ## numPatch x fs
        FeatList.append(featGroup)
    all_info = [SlideNames, FeatList, bag_label]
    # save_pickle(save_dir, all_info)
    return all_info


if __name__ == "__main__":
    main()
