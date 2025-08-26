import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from utils.utils import initialize_weights
import numpy as np



"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_tasks = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        
                            
        if dropout:
            self.attention_a.append(nn.Dropout(0.5))
            self.attention_b.append(nn.Dropout(0.5))
            

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        
        
        self.attention_c = nn.Linear(D, n_tasks)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        
        
        A = a.mul(b)  
        A = self.attention_c(A)  # N x n_classes

        return A, x


"""
TOAD multi-task + concat mil network w/ attention-based pooling
args:
    gate: whether to use gating in attention network
    size_args: size config of attention network
    dropout: whether to use dropout in attention network
    n_classes: number of classes
"""

class TOAD_fc_mtl_concat(nn.Module):
    def __init__(self, gate = True, size_arg = "big", dropout = False, n_classes = 2):
        super(TOAD_fc_mtl_concat, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.5))
        fc.extend([nn.Linear(size[1], size[1]), nn.ReLU()])
        if dropout:
            fc.append(nn.Dropout(0.5))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_tasks = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_tasks = 2)
        
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        
        # self.clincal = nn.Sequential(*[nn.Linear(71,71), nn.ReLU()])
        
        # 图像+..
        self.classifier = nn.Linear(size[1]+9, n_classes)                
        #self.classifier = nn.Linear(size[1]+71, n_classes)
        # self.site_classifier = nn.Linear(size[1]+1, 2)
        

        initialize_weights(self)
        print("-------------------------------",initialize_weights)        
                
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')

        else:
            self.attention_net = self.attention_net.to(device)


        self.classifier = self.classifier.to(device)
        # self.site_classifier = self.site_classifier.to(device)
        
    def forward(self, h, gender,history_other_malignancy,history_neoadjuvant_treatment,Infiltrating_Carcinoma_NOS,
    Infiltrating_Ductal_Carcinoma,Infiltrating_Lobular_Carcinoma,Medullary_Carcinoma,Metaplastic_Carcinoma,Mixed_Histology,
    Mucinous_Carcinoma,Other_specify,age_at_diagnosis_age,ajcc_tumor_pathologic_pt,ajcc_nodes_pathologic_pn,ajcc_metastasis_pathologic_pm,
    ajcc_pathologic_tumor_stage,Left,Left_Upper_Inner_Quadrant,Left_Upper_Outer_Quadrant,Left_Lower_Inner_Quadrant,Left_Lower_Outer_Quadrant,
    Right,Right_Upper_Inner_Quadrant,Right_Upper_Outer_Quadrant,Right_Lower_Inner_Quadrant,Right_Lower_Outer_Quadrant,FLT1,STK32B,RASSF7,DCK,
    MELK,EXT1,GNAZ,EBF4,MTDH,PITRM1,ESM1,IGFBP5,FGF18,SCUBE2,TGFB3,COL4A2,GPR180,MMP9,RTN4RL1,DIAPH3,CDC42BPA,CCNE2,ECT2,CENPA,LIN9,MCM6,NUSAP1,
    TSPYL5,RFC4,RECQL5,CDCA7,DTL,ALDH4A1,OXCT1,GMPS,GSTM3,SLC2A3,EGLN1,BBC3,NMU,UCHL5,AP2B1,MS4A7,RAB6B,SERF1A, 
    return_features=False,
    attention_only=False):
    
        A, h = self.attention_net(h)
          
        A = torch.transpose(A, 1, 0) 
        if attention_only:
            return A[0]

        A_raw = A 
        A = F.softmax(A, dim=1) 
        M = torch.mm(A, h)
        # print("M", M.shape)
    
        gender_tmp = gender.repeat(M.size(0),1)
        # print("gender: ",gender_tmp)
        # print("M:",M.size(1))
        
                                
        # 注释                
        """
        tmp =  torch.from_numpy( np.array([gender,history_other_malignancy,history_neoadjuvant_treatment,Infiltrating_Carcinoma_NOS,
    Infiltrating_Ductal_Carcinoma,Infiltrating_Lobular_Carcinoma,Medullary_Carcinoma,Metaplastic_Carcinoma,Mixed_Histology,
    Mucinous_Carcinoma,Other_specify,age_at_diagnosis_age,ajcc_tumor_pathologic_pt,ajcc_nodes_pathologic_pn,ajcc_metastasis_pathologic_pm,
    ajcc_pathologic_tumor_stage,Left,Left_Upper_Inner_Quadrant,Left_Upper_Outer_Quadrant,Left_Lower_Inner_Quadrant,Left_Lower_Outer_Quadrant,
    Right,Right_Upper_Inner_Quadrant,Right_Upper_Outer_Quadrant,Right_Lower_Inner_Quadrant,Right_Lower_Outer_Quadrant,FLT1,STK32B,RASSF7,DCK,
    MELK,EXT1,GNAZ,EBF4,MTDH,PITRM1,ESM1,IGFBP5,FGF18,SCUBE2,TGFB3,COL4A2,GPR180,MMP9,RTN4RL1,DIAPH3,CDC42BPA,CCNE2,ECT2,CENPA,LIN9,MCM6,NUSAP1,
    TSPYL5,RFC4,RECQL5,CDCA7,DTL,ALDH4A1,OXCT1,GMPS,GSTM3,SLC2A3,EGLN1,BBC3,NMU,UCHL5,AP2B1,MS4A7,RAB6B,SERF1A], dtype=np.float32) ).to('cuda:0')
        """
        tmp =  torch.from_numpy(np.array([ajcc_tumor_pathologic_pt,ajcc_nodes_pathologic_pn,ajcc_metastasis_pathologic_pm,
        FLT1,STK32B,RASSF7,MS4A7,RAB6B,SERF1A], dtype=np.float32) ).to('cuda:0')
                        
        # print("tmp", tmp.repeat(M.size(0), 1).shape)
                
        M = torch.cat([M, tmp.repeat(M.size(0),1)], dim=1) 
        

        logits  = self.classifier(M[0].unsqueeze(0)) 
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)

        # site_logits  = self.site_classifier(M[1].unsqueeze(0)) 
        # site_hat = torch.topk(site_logits, 1, dim = 1)[1]
        # site_prob = F.softmax(site_logits, dim = 1)

        results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        
        results_dict.update({'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'A': A_raw})

        return results_dict

