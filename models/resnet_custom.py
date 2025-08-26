# modified from Pytorch official resnet.py
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
#from torchsummary import summary
import torch.nn.functional as F
from torchvision import models
# import clip
from compact_bilinear_pooling import CountSketch, CompactBilinearPooling

available_policies = {"resnet18": models.resnet18,"resnet50": models.resnet50, "vgg16": models.vgg16, "vgg19": models.vgg19,
                      "alexnet": models.alexnet, "inception": models.inception_v3}



__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck_Baseline(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_Baseline, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet_Baseline(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet_Baseline, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1) 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

def resnet50_baseline(pretrained=False):
    """Constructs a Modified ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_Baseline(Bottleneck_Baseline, [3, 4, 6, 3])
    if pretrained:
        model = load_pretrained_weights(model, 'resnet50')
    return model

def resnet18_baseline(cnv=False, pretrained=False):
    """Constructs a Modified ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    #model = ResNet(BasicBlock, [2, 2, 2, 2])
    model = available_policies["resnet18"](pretrained=False)
    if cnv:
        model = Cnn_With_Clinical_Net(model)
        if pretrained:
            # model = load_pretrained_weights(model, 'resnet18')
            checkpoint = torch.load(
                '/home/wangwy/HEpreictTMBcode/save_model/model_best_resnet18clin_top5_TCGA.pth')
            model.load_state_dict(checkpoint['state_dict'])
    else:
        model=Net(model)
        #checkpoint = torch.load('/mnt/colon/HEpreictTMBcode/save_model/checkpoint_resnet18_bestmodel_qingyi_finetuning_100_0923_IHC_best.pth')
        #model.load_state_dict(checkpoint['state_dict'])
        #if pretrained:
            #checkpoint = torch.load('/home/wangwy/HEpreictTMBcode/save_model/checkpoint_resnet18_bestmodel_best_TCGA.pth')
            #model.load_state_dict(checkpoint['state_dict'])
            #path=os.getcwd()+f'/save_model/resnet18_{k_fold}'
            #model=torch.load('/home/wangwy/HEpreictTMBcode/save_model/resnet18_{k_fold}.pkl')
    return model

def resnet18_baseline_original(pretrained=False):
    """Constructs a Modified ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_Baseline(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        model = load_pretrained_weights(model, 'resnet18')
    return model

def resnet18_baseline_actually_might_be_transformer(cnv=False, pretrained=False):
    """Constructs a Modified ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    #model = ResNet(BasicBlock, [2, 2, 2, 2])
    model = available_policies["resnet18"](pretrained=pretrained)
    #model = clip.load("ViT-B/32", device='cuda')
    if cnv:
        model = Cnn_With_Clinical_Net(model)
        #if pretrained:
            # model = load_pretrained_weights(model, 'resnet18')
            #checkpoint = torch.load(
                #'/home/wangwy/HEpreictTMBcode/save_model/model_best_resnet18clin_top5_TCGA.pth')
            #model.load_state_dict(checkpoint['state_dict'])
    else:
        model=Net(model)
        #if pretrained:
            #checkpoint = torch.load('/home/wangwy/HEpreictTMBcode/save_model/checkpoint_resnet18_bestmodel_qingyi_finetuning_LRNoChange.pth')  #checkpoint_resnet18_bestmodel_best_TCGA
            #checkpoint = torch.load('/home/wangwy/HEpreictTMBcode/save_model/checkpoint_resnet18_bestmodel_TCGA.pth')
            #model.load_state_dict(checkpoint['state_dict'])
            

    return model

def load_pretrained_weights(model, name):
    pretrained_dict = model_zoo.load_url(model_urls[name])
    model.load_state_dict(pretrained_dict, strict=False)
    return model

class Net(nn.Module):  # 没有临床信息
    def __init__(self, model):
        super(Net, self).__init__()
        self.layer = nn.Sequential(*list(model.children()))
        self.conv = self.layer[:-1]
        self.dense = None
        if type(self.layer[-1]) == type(nn.Sequential()):
            self.feature = self.layer[-1][-1].in_features
            self.dense = self.layer[-1][:-1]
        else:
            self.feature = self.layer[-1].in_features
        self.linear = nn.Linear(self.feature, 2)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        if self.dense is not None:
            x = self.dense(x)
        x = x.view(x.size(0), -1)
        #x = self.linear(x)
        return x
        
class Net_Transformer(nn.Module):  # 没有临床信息
    def __init__(self, model):
        super(Net, self).__init__()
        layer = model.children()
        #print(layer)
        self.model = model
        self.layer1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.layer3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1)
        self.dense = None
        self.feature = 512
        self.linear = nn.Linear(self.feature, 2)
        self.transformer = Transformer(512, 1, 8, 64, mlp_dim=512, dropout=0.35)
        self.transformer1 = Transformer(512, 2, 8, 64, mlp_dim=512, dropout=0.3)
        self.transformer2 = Transformer(512, 5, 8, 64, mlp_dim=512, dropout=0.3)
        self.pool = 'mean'
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(512),
        )

        nn.init.normal_(self.layer1.weight, mean=0, std=1)
        nn.init.normal_(self.layer2.weight, mean=0, std=1)
        nn.init.normal_(self.layer3.weight, mean=0, std=1)

    def forward(self, x):
        x = self.model.encode_image(x)  # encode_image
        # x = x.view(-1, x.size(0))
        # print(x.size())
        x = x.to(torch.float32)
        x = torch.unsqueeze(x, 0)
        x = self.transformer(x)
        # print(x.size())
        # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.mlp_head(x)
        # x1 = self.transformer1(x)
        # x1 = self.mlp_head(x1)
        # x2 = self.transformer2(x)
        # x2 = self.mlp_head(x2)
        # x = x + x1 +x2
        # x = self.transformer(x)
        # x = self.mlp_head(x)
        x = x.view(x.size(1), -1)
        print(x.size())
        #x = self.linear(x)

        return x
'''
class Net(nn.Module):  # 没有临床信息
    def __init__(self, model):
        super(Net, self).__init__()
        layer = model.children()
        #print(layer)
        self.model = model
        self.layer1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.layer3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1)
        self.dense = None
        self.feature = 512
        self.linear = nn.Linear(self.feature, 2)
        nn.init.normal_(self.layer1.weight, mean=0, std=1)
        nn.init.normal_(self.layer2.weight, mean=0, std=1)
        nn.init.normal_(self.layer3.weight, mean=0, std=1)

    def forward(self, x):
        x = self.model.encode_image(x)
        x = x.reshape((512, 8, 8))
        x = x.to(torch.float32)
        x1 = self.layer1(x)
        # print(x1.size())
        x2 = self.layer2(x)
        # print(x2.size())
        x3 = self.layer3(x)
        # print(x3.size())
        x = x1 + x2 + x3 + x

        x = x.view(-1, x.size(0))
        # print(x.size())
        #if self.dense is not None:
            #x = self.dense(x)
        # print(x.size())
        #x = self.linear(x)
        return x
'''
class Cnn_With_Clinical_Net(nn.Module):
    def __init__(self, model):
        super(Cnn_With_Clinical_Net, self).__init__()
        # self.layer = nn.Sequential(*list(model.children())[:-1])
        # self.feature = list(model.children())[-1].in_features
        # self.cnn = nn.Linear(self.feature, 128)

        # CNN
        self.layer = nn.Sequential(*list(model.children()))
        self.conv = self.layer[:-1]
        self.dense = None  # 是否有密集链接：
        if type(self.layer[-1]) == type(nn.Sequential()):
            self.feature = self.layer[-1][-1].in_features
            self.dense = self.layer[-1][:-1]
        else:
            self.feature = self.layer[-1].in_features
        self.linear = nn.Linear(self.feature, 128)  # 全连接层，输出节点是128

        # clinical feature
        self.clinical = nn.Linear(5, 5)  # 临床特征的数量，原本是55的

        # concat
        self.mcb = CompactBilinearPooling(128, 5, 128).cuda()
        # self.concat = nn.Linear(128+55, 128)
        self.bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(True)
        self.classifier = nn.Linear(128, 2)  # 输入128，输出2

    def forward(self, x, clinical_features):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        if self.dense is not None:  # dense存在，则...
            x = self.dense(x)
        x = self.linear(x)
        # print(clinical_features.size())
        clinical = self.clinical(clinical_features)
        x = self.mcb(x, clinical)
        # x = torch.cat([x, clinical], dim=1)
        # x = self.concat(x)
        x = self.bn(x)
        x = self.relu(x)
        #x = self.classifier(x)
        return x

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
