from torch import nn
import torch.nn.functional as F
from torch.nn.functional import normalize
import torch
from utils import cal_weights_via_CAN


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.enc_1 = nn.Linear(input_dim, 500)
        self.enc_2 = nn.Linear(500, 500)
        self.enc_3 = nn.Linear(500, 2000)
        self.enc_4 = nn.Linear(2000, feature_dim)
    def forward(self, x):
        enc_d1 = F.relu(self.enc_1(x))
        enc_d2 = F.relu(self.enc_2(enc_d1))
        enc_d3 = F.relu(self.enc_3(enc_d2))
        enc_d4 = self.enc_4(enc_d3)
        return enc_d1, enc_d2, enc_d3, enc_d4
class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )
    def forward(self, x):
        return self.decoder(x)

class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num, device):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim),
        )
        self.class_num = class_num
        self.view = view
        self.commonfeature_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim * view, high_feature_dim),
        )
        # add share_space
        self.mk = torch.nn.Linear(feature_dim*view, 64, bias=False)
        self.mv = torch.nn.Linear(64, feature_dim*view, bias=False)
        self.softmax = torch.nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, xs):
        xrs = []
        zs = []
        hs = []
        for v in range(self.view):
            x = xs[v]
            _, _, _, z = self.encoders[v](x)
            h = normalize(self.feature_contrastive_module(z), dim=1)
            xr = self.decoders[v](z)
            hs.append(h)
            zs.append(z)
            xrs.append(xr)
        return  xrs, zs, hs

    # GCFAgg+EAMLP+graph-diy-5.14
    # def GCFAgg(self, xs):
    #     zs = []
    #     for v in range(self.view):
    #         x = xs[v]
    #         _, _, _, z = self.encoders[v](x)
    #         zs.append(z)
    #     catz = torch.cat(zs, 1)
    #     attn = self.mk(catz)
    #     attn = self.softmax(attn)
    #     out = self.mv(attn)
    #     out = self.dropout(out)
    #     commonz = normalize(self.commonfeature_contrastive_module(out + catz), dim=1)
    #     weights, raw_weights = cal_weights_via_CAN(commonz.t(), 15)
    #     return commonz, raw_weights

    # ACCMVC-TPL-before-MLP-10.17
    def GCFAgg(self, xs):
        zs = []
        for v in range(self.view):
            x = xs[v]
            _, _, _, z = self.encoders[v](x)
            zs.append(z)
        catz = torch.cat(zs, 1)
        attn = self.mk(catz)
        attn = self.softmax(attn)
        out = self.mv(attn)
        out = self.dropout(out)
        commonz = normalize(self.commonfeature_contrastive_module(out + catz), dim=1)
        weights, raw_weights = cal_weights_via_CAN(commonz.t(), 10)
        return commonz, raw_weights


    # 消融no-eamlp-diy-5.14
    # def GCFAgg(self, xs):
    #     zs = []
    #     for v in range(self.view):
    #         x = xs[v]
    #         _, _, _, z = self.encoders[v](x)
    #         zs.append(z)
    #     catz = torch.cat(zs, 1)
    #     # attn = self.mk(catz)
    #     # attn = self.softmax(attn)
    #     # out = self.mv(attn)
    #     # out = self.dropout(out)
    #     commonz = normalize(self.commonfeature_contrastive_module(catz), dim=1)
    #     weights, raw_weights = cal_weights_via_CAN(commonz.t(), 15)
    #     return commonz, raw_weights

    # 消融no-eamlp-no-grap-diy-5.14
    # def GCFAgg(self, xs):
    #     zs = []
    #     for v in range(self.view):
    #         x = xs[v]
    #         _, _, _, z = self.encoders[v](x)
    #         zs.append(z)
    #     catz = torch.cat(zs, 1)
    #     # attn = self.mk(catz)
    #     # attn = self.softmax(attn)
    #     # out = self.mv(attn)
    #     # out = self.dropout(out)
    #     commonz = normalize(self.commonfeature_contrastive_module(catz), dim=1)
    #     # weights, raw_weights = cal_weights_via_CAN(commonz.t(), 15)
    #     return commonz

    # 消融--GCFAgg+EAMLP+graph-diy-5.14--nograph--sample-level-cl
    # def GCFAgg(self, xs):
    #     zs = []
    #     for v in range(self.view):
    #         x = xs[v]
    #         _, _, _, z = self.encoders[v](x)
    #         zs.append(z)
    #     catz = torch.cat(zs, 1)
    #     # commonz = self.extract_layers(catz)
    #     attn = self.mk(catz)
    #     attn = self.softmax(attn)
    #     # attn = torch.sigmoid(attn)
    #     # attn = attn / torch.sum(attn, dim=1,keepdim=True)
    #     out = self.mv(attn)
    #     out = self.dropout(out)
    #     # commonz = normalize(out + catz)
    #     commonz = normalize(self.commonfeature_contrastive_module(out + catz), dim=1)
    #     # weights, raw_weights = cal_weights_via_CAN(commonz.t(), 15)
    #     return commonz

    #GCFAgg全部放到forward里作对比
    # def forward(self, xs):
    #     xrs = []
    #     zs = []
    #     hs = []
    #     for v in range(self.view):
    #         x = xs[v]
    #         _, _, _, z = self.encoders[v](x)
    #         h = normalize(self.feature_contrastive_module(z), dim=1)
    #         xr = self.decoders[v](z)
    #         hs.append(h)
    #         zs.append(z)
    #         xrs.append(xr)
    #     catz = torch.cat(zs, 1)
    #     attn = self.mk(catz)
    #     attn = self.softmax(attn)
    #     # attn = torch.sigmoid(attn)
    #     # attn = attn / torch.sum(attn, dim=1,keepdim=True)
    #     out = self.mv(attn)
    #     out = self.dropout(out)
    #     # commonz = normalize(out + catz)
    #     commonz = normalize(self.commonfeature_contrastive_module(out + catz), dim=1)
    #     weights, raw_weights = cal_weights_via_CAN(commonz.t(), 15)
    #     return  xrs, zs, hs, commonz, raw_weights

    # GCFAgg+EAMLP+graph-4.9
    # def GCFAgg(self, xs):
    #     zs = []
    #     for v in range(self.view):
    #         x = xs[v]
    #         _, _, _, z = self.encoders[v](x)
    #         zs.append(z)
    #     catz = torch.cat(zs, 1)
    #     commonz = self.extract_layers(catz)
    #     weights, raw_weights = cal_weights_via_CAN(commonz.t(), 15)
    #     commonz = normalize(self.commonfeature_contrastive_module(commonz), dim=1)
    #     return commonz, raw_weights

    #消融no-eamlp
    # def GCFAgg(self, xs):
    #     zs = []
    #     for v in range(self.view):
    #         x = xs[v]
    #         _, _, _, z = self.encoders[v](x)
    #         zs.append(z)
    #     #直接相加
    #     stacked_zs = torch.stack(zs, dim=0)
    #     commonz = torch.sum(stacked_zs, dim=0)
    #     weights, raw_weights = cal_weights_via_CAN(commonz.t(), 15)
    #     commonz = normalize(self.feature_contrastive_module(commonz), dim=1)
    #     return commonz, raw_weights

    # 消融， GCFAgg+EAMLP + 普通cl
    # def GCFAgg(self, xs):
    #     zs = []
    #     for v in range(self.view):
    #         x = xs[v]
    #         _, _, _, z = self.encoders[v](x)
    #         zs.append(z)
    #     catz = torch.cat(zs, 1)
    #     commonz = self.extract_layers(catz)
    #     commonz = normalize(self.commonfeature_contrastive_module(commonz), dim=1)
    #     return commonz

    # distances-graph
    # def GCFAgg(self, xs):
    #     zs = []
    #     for v in range(self.view):
    #         x = xs[v]
    #         _, _, _, z = self.encoders[v](x)
    #         zs.append(z)
    #     catz = torch.cat(zs, 1)
    #     commonz = self.extract_layers(catz)
    #     distances = distance(commonz.t(), commonz.t())
    #     softmax = torch.nn.Softmax(dim=1)
    #     # recons_w = softmax(-distances)
    #     recons_w = softmax(-distances/0.5)
    #     commonz = normalize(self.commonfeature_contrastive_module(commonz+catz), dim=1)
    #     return commonz, recons_w

    # dity多头eamlp-5.28
    # def GCFAgg(self, xs):
    #     zs = []
    #     # Loop over the view
    #     for v in range(self.view):
    #         x = xs[v]
    #         _, _, _, z = self.encoders[v](x)
    #         zs.append(z)
    #     attn = self.mk(*zs)  # 对每个视图的特征进行mk
    #     # 对列表中的每个张量进行 softmax 操作
    #     softmax_attn = []
    #     for tensor in attn: # 每个视图进行标准化
    #         softmax_tensor = self.softmax(tensor)
    #         softmax_attn.append(softmax_tensor)
    #     # attn = self.softmax(attn)  # 每个视图进行标准化
    #     out = self.mv(*attn)  # 每个视图mv
    #     # 对应位置的张量相加并组成新的列表
    #     commonz = []
    #     for tensor1, tensor2 in zip(out, zs):
    #         combined_tensor = tensor1 + tensor2
    #         commonz.append(combined_tensor)
    #     commonz = torch.cat(commonz, 1)
    #     commonz = normalize(self.commonfeature_contrastive_module(commonz), dim=1)
    #     weights, raw_weights = cal_weights_via_CAN(commonz.t(), 15)
    #     return commonz, raw_weights
    # dity多头eamlp-5.28消融nograph
    # def GCFAgg(self, xs):
    #     zs = []
    #     # Loop over the view
    #     for v in range(self.view):
    #         x = xs[v]
    #         _, _, _, z = self.encoders[v](x)
    #         zs.append(z)
    #     attn = self.mk(*zs)  # 对每个视图的特征进行mk
    #     # 对列表中的每个张量进行 softmax 操作
    #     softmax_attn = []
    #     for tensor in attn: # 每个视图进行标准化
    #         softmax_tensor = self.softmax(tensor)
    #         softmax_attn.append(softmax_tensor)
    #     # attn = self.softmax(attn)  # 每个视图进行标准化
    #     out = self.mv(*attn)  # 每个视图mv
    #     # 对应位置的张量相加并组成新的列表
    #     commonz = []
    #     for tensor1, tensor2 in zip(out, zs):
    #         combined_tensor = tensor1 + tensor2
    #         commonz.append(combined_tensor)
    #     commonz = torch.cat(commonz, 1)
    #     commonz = normalize(self.commonfeature_contrastive_module(commonz), dim=1)
    #     # weights, raw_weights = cal_weights_via_CAN(commonz.t(), 15)
    #     return commonz
    # dity多头eamlp-5.28消融no-eamlp
    # def GCFAgg(self, xs):
    #     zs = []
    #     # Loop over the view
    #     for v in range(self.view):
    #         x = xs[v]
    #         _, _, _, z = self.encoders[v](x)
    #         zs.append(z)
    #     # attn = self.mk(*zs)  # 对每个视图的特征进行mk
    #     # # 对列表中的每个张量进行 softmax 操作
    #     # softmax_attn = []
    #     # for tensor in attn: # 每个视图进行标准化
    #     #     softmax_tensor = self.softmax(tensor)
    #     #     softmax_attn.append(softmax_tensor)
    #     # # attn = self.softmax(attn)  # 每个视图进行标准化
    #     # out = self.mv(*attn)  # 每个视图mv
    #     # # 对应位置的张量相加并组成新的列表
    #     # commonz = []
    #     # for tensor1, tensor2 in zip(out, zs):
    #     #     combined_tensor = tensor1 + tensor2
    #     #     commonz.append(combined_tensor)
    #     # commonz = torch.cat(commonz, 1)
    #     commonz = torch.cat(zs, 1)
    #     commonz = normalize(self.commonfeature_contrastive_module(commonz), dim=1)
    #     weights, raw_weights = cal_weights_via_CAN(commonz.t(), 15)
    #     return commonz, raw_weights

    # dity多头eamlp-5.28消融no-eamlp-nograph
    # def GCFAgg(self, xs):
    #     zs = []
    #     # Loop over the view
    #     for v in range(self.view):
    #         x = xs[v]
    #         _, _, _, z = self.encoders[v](x)
    #         zs.append(z)
    #     commonz = torch.cat(zs, 1)
    #     commonz = normalize(self.commonfeature_contrastive_module(commonz), dim=1)
    #     # weights, raw_weights = cal_weights_via_CAN(commonz.t(), 15)
    #     return commonz