import torch.nn as nn
import torch
import numpy as np
import random
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from utils import *
from torch_geometric.nn import MessagePassing
from torch_sparse import SparseTensor, matmul
from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
class FairADG(torch.nn.Module):
    def __init__(self, train_args,logger):
        super(FairADG, self).__init__()
        self.args = train_args
        self.assigner = NeiborAssigner1(self.args.nfeat, self.args.hidden,self.args.channels).to(self.args.device)
        # model of FairSAD: encoder, masker, classifier
        self.encoder = DisGCN(nfeat=self.args.nfeat,
                                nhid=self.args.hidden,
                                nclass=self.args.nclass,
                                # 这里的channel
                                chan_num=self.args.channels,
                                layer_num=2,
                                dropout=self.args.dropout,
                                NeiborAssigner = self.assigner
                                ).to(self.args.device)
        
        
        #self.classifier_y = nn.Linear(train_args.hidden, train_args.nclass).to(self.args.device)
        self.classifier_y = nn.Linear(train_args.hidden , train_args.nclass).to(self.args.device)
        # 输出为敏感属性的分类
        self.classifier_s = nn.Linear(train_args.hidden, train_args.nclass).to(self.args.device)
        self.classifier_us = nn.Linear(train_args.hidden, 2).to(self.args.device)
        
        self.per_channel_dim = train_args.hidden 
        self.channel_cls = nn.Linear(self.per_channel_dim, train_args.channels).to(self.args.device)

        # optumizer
        self.optimizer_g = torch.optim.Adam(list(self.encoder.parameters()) + list(self.classifier_y.parameters())+list(self.classifier_s.parameters())+list(self.assigner.parameters()), lr=train_args.lr,
                                            weight_decay=train_args.weight_decay)
        self.optimizer_ass = torch.optim.Adam(list(self.assigner.parameters()), lr=train_args.lr,
                                            weight_decay=train_args.weight_decay)

        self.optimizer_c = torch.optim.Adam(list(self.channel_cls.parameters()), lr=train_args.lr,
                                            weight_decay=train_args.weight_decay)

        # loss function
        self.criterion_bce = nn.BCEWithLogitsLoss()
        self.criterion_dc = DistCor()
        self.criterion_mul_cls = nn.CrossEntropyLoss()
        self.covf = FeatCov()

        self.encoder.init_parameters()
        self.encoder.init_edge_weight()
        self.logger = logger

        for m in self.modules():
            self.weights_init(m)

    

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        h = self.encoder(x, edge_index)
        # h输出应该是每个节点都是二维的
        hs,hus = h
        # 对于hus进行分类y损失，对于hus使用和y的交叉熵损失，对hs使用和s的交叉熵损失

        output_s = self.classifier_s(hs)
        output_us = self.classifier_y(hus)
        return h, output_us,output_s

    def train_fit(self, data, epochs, **kwargs):
        # parsing parameters
        alpha = kwargs.get('alpha', None)
        beta = kwargs.get('beta', None)
        pbar = kwargs.get('pbar', None)

        best_res_val = 0.0
        save_model = 0

        # training encoder, assigner, classifier
        for epoch in range(epochs):
            self.encoder.train()
            #self.masker.train()
            self.classifier_y.train()
            self.channel_cls.train()
            self.classifier_s.train()
            self.assigner.train()
            # 训练encoder和分类器
            self.optimizer_g.zero_grad()
            # 训练通道鉴别器（对于我来说需要吗？）
            self.optimizer_c.zero_grad()
            self.optimizer_ass.zero_grad()

            h,hus,hs = self.encoder(data.features, data.edge_index)
            
            output_us = self.classifier_y(hus)
        #     # 尝试对分类头进行平滑
        #     us_for_s = self.classifier_us(hus)
        #    # us_for_s = F.relu(us_for_s)
        #     us_for_s = F.log_softmax(us_for_s, dim=-1)
            output_s = self.classifier_s(hs)

     

            # downstream tasks loss
            loss_cls_train_us = self.criterion_bce(output_us[data.idx_train],
                                                data.labels[data.idx_train].unsqueeze(1).float())
            # 需要有一个对于s的平滑损失

            # loss for s
            loss_cls_train_s = self.criterion_bce(output_s[data.idx_train],data.sens[data.idx_train].unsqueeze(1).float())

            # channel identification loss
            loss_chan_train = 0
            for i in range(self.args.channels):
                chan_output = self.channel_cls(h[:, i*self.per_channel_dim:(i+1)*self.per_channel_dim])
                chan_tar = torch.ones(chan_output.shape[0], dtype=int)*i
                chan_tar = chan_tar.to(self.args.device)
                loss_chan_train += self.criterion_mul_cls(chan_output, chan_tar)

            # distance correlation loss

            # 去相关损失
            loss_disen_train = 0
            len_per_channel = int(h.shape[1] / self.args.channels)
            for i in range(self.args.channels):
                for j in range(i + 1, self.args.channels):
                    loss_disen_train += self.criterion_dc(
                        h[data.idx_train, i * len_per_channel:(i + 1) * len_per_channel],
                        h[data.idx_train, j * len_per_channel:(j + 1) * len_per_channel])
            # 平滑损失
           # uniform_target = torch.ones_like(us_for_s, dtype=torch.float).to(self.args.device) / self.args.nclass
            # print(us_for_s)
            # print(uniform_target.shape)
            # F.kl_div
           # kl_loss = F.kl_div(us_for_s[data.idx_train], uniform_target[data.idx_train].float())
            kl_loss = self.covf(hus[data.idx_train],data.sens[data.idx_train])
            
           
            

            loss_train = loss_cls_train_us + alpha * (loss_disen_train+loss_chan_train)+ +beta*loss_cls_train_s + 0.3*kl_loss

            loss_train.backward()
            self.optimizer_g.step()
            self.optimizer_c.step()
            self.optimizer_ass.step()

            # evaluating encoder, assigner, classifier
            self.encoder.eval()
            # self.assigner.eval()
            self.classifier_y.eval()

            if epoch % 10 == 0:
                self.assigner.eval()
                #self.masker.eval()
                #edge_weight = self.encoder.assigner(torch.cat([data.features[data.edge_index.storage._col, :],
                                                               #data.features[data.edge_index.storage._row, :]], dim=1).detach())
                # 这里会在encoder中生成edge_weight
                # 但是我是否应该对edge_index进行处理，删除概率比较小的？
                h = self.encoder.emb_test(data.features, data.edge_index,self.assigner)
               
                y_output_val = self.classifier_y(h)
                y_output_val = y_output_val.detach()
                y_pred_val = (y_output_val.squeeze() > 0).type_as(data.sens)
                acc_val = accuracy_score(data.labels[data.idx_val].cpu(), y_pred_val[data.idx_val].cpu())
                roc_val = roc_auc_score(data.labels[data.idx_val].cpu(), y_output_val[data.idx_val].cpu())
                f1_val = f1_score(data.labels[data.idx_val].cpu(), y_pred_val[data.idx_val].cpu())
                parity, equality = fair_metric(y_pred_val[data.idx_val].cpu().numpy(),
                                               data.labels[data.idx_val].cpu().numpy(),
                                               data.sens[data.idx_val].cpu().numpy())

                res_val = acc_val + roc_val - parity - equality
                self.logger.info(f'Epoch {epoch}: Val Accuracy: {acc_val:.4f}, Val ROC AUC: {roc_val:.4f}, Val F1: {f1_val:.4f}, Val Parity: {parity:.4f}, Val Equality: {equality:.4f}')

                if res_val > best_res_val:
                    best_res_val = res_val
                    """
                        evaluation
                    """
                    acc_test = accuracy_score(data.labels[data.idx_test].cpu(), y_pred_val[data.idx_test].cpu())
                    roc_test = roc_auc_score(data.labels[data.idx_test].cpu(), y_output_val[data.idx_test].cpu())
                    f1_test = f1_score(data.labels[data.idx_test].cpu(), y_pred_val[data.idx_test].cpu())
                    parity_test, equality_test = fair_metric(y_pred_val[data.idx_test].cpu().numpy(),
                                                   data.labels[data.idx_test].cpu().numpy(),
                                                   data.sens[data.idx_test].cpu().numpy())
                    save_model = epoch
                
            if pbar is not None:
                pbar.set_postfix({'train_total_loss': "{:.2f}".format(loss_train.item()),
                                  'cls_loss_us':  "{:.2f}".format(loss_cls_train_us.item()),
                                  'cls_loss_s': "{:.2f}".format(loss_cls_train_s.item()),
                                  'disen_loss': "{:.2f}".format(loss_disen_train.item()),
                                    'kl': "{:.2f}".format(kl_loss.item()),
                                #   'mask_loss': "{:.2f}".format(loss_mask_train.i tem()),
                                  'val_loss': "{:.2f}".format(res_val), 'save model': save_model})
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        self.logger.info(f'ROC AUC: {roc_test:.4f}, Test F1: {f1_test:.4f}, Test Parity: {parity_test:.4f}, Test Equality: {equality_test:.4f}')


        return roc_test, f1_test, acc_test, parity_test, equality_test

class DisenLayer(MessagePassing):
    def __init__(self, in_dim, out_dim, channels, reduce=False):
        super(DisenLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.channels = channels
        self.per_channel_dim = self.out_dim 
        self.reduce = reduce

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for i in range(channels):
            if reduce:
                self.lin_layers.append(nn.Linear(in_features=in_dim, out_features=self.per_channel_dim))
                self.conv_layers.append(Linear(in_channels=self.per_channel_dim, out_channels=self.per_channel_dim, bias=False,
                                               weight_initializer='glorot'))
            else:
                self.conv_layers.append(Linear(in_channels=self.in_dim, out_channels=self.per_channel_dim, bias=False,
                                               weight_initializer='glorot'))
        self.bias_list = nn.ParameterList(
            nn.Parameter(torch.empty(size=(1, self.per_channel_dim), dtype=torch.float), requires_grad=True) for i in
            range(self.channels))

    def get_reddim_k(self, x):
        z_feats = []
        for k in range(self.channels):
            z_feat = self.lin_layers[k](x)
            z_feats.append(z_feat)
        return z_feats

    def get_k_feature(self, x):
        z_feats = []
        for k in range(self.channels):
            z_feats.append(x)
        return z_feats

    def forward(self, x, edge_index, edge_weight):
        assert self.channels == edge_weight.shape[1], "axis dimension in direction 1 need to be equal to channels number"
        # print(self.channels)
        if self.reduce:
            z_feats = self.get_reddim_k(x)
        else:
            z_feats = self.get_k_feature(x)
        c_feats = []
        for k, layer in enumerate(self.conv_layers):
            # print("k",k)
            c_temp = layer(z_feats[k])
            edge_index_copy = edge_index.clone()
            if not edge_index_copy.has_value():
                edge_index_copy = edge_index_copy.fill_value(1., dtype=None)
            edge_index_copy.storage.set_value_(edge_index_copy.storage.value() * edge_weight[:, k])
            
            out = self.propagate(edge_index_copy, x=c_temp)
            if self.bias_list is not None:
                out = out + self.bias_list[k]
            c_feats.append(F.normalize(out, p=2, dim=1))
        output = torch.cat(c_feats, dim=1)
        # print(len(c_feats))
        outputs = torch.cat(c_feats[:self.channels//2])
        outputus = torch.cat(c_feats[self.channels//2:])
        return output,outputs,outputus

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

class DisGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, chan_num, layer_num, dropout=0.5, NeiborAssigner=None):
        super(DisGCN, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.dropout_rate = dropout
        self.chan_num = chan_num
        self.layer_num = layer_num
        self.edge_weight = None
        self.assigner = NeiborAssigner

        
        self.disenlayers = nn.ModuleList()
        for i in range(layer_num-1):
            if i == 0:
                self.disenlayers.append(DisenLayer(nfeat, nhid, chan_num))
            else:
                self.disenlayers.append(DisenLayer(nhid, nhid, chan_num))
        self.dropout = nn.Dropout(dropout)

        self.init_parameters()

    def init_parameters(self):
        for i, item in enumerate(self.parameters()):
            torch.nn.init.normal_(item, mean=0, std=1)
    
    def init_edge_weight(self):
        for m in self.assigner.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        assert isinstance(edge_index, SparseTensor), "Expected input is sparse tensor"
        feats_pair = torch.cat([x[edge_index.storage._col, :], x[edge_index.storage._row, :]], dim=1)
        # edge_weight = self.assigner(feats_pair)
        edge_weight = self.assigner(x.detach(),edge_index)
        
        for layer in self.disenlayers:
            x,xs,xus = layer(x, edge_index, edge_weight)
            x = self.dropout(x)
            xs = self.dropout(xs)
            xus = self.dropout(xus)
        # 需要输出分为hs和hus
        return x,xs,xus
    def emb_test(self, x, edge_index,assigner):
        assert isinstance(edge_index, SparseTensor), "Expected input is sparse tensor"
        feats_pair = torch.cat([x[edge_index.storage._col, :], x[edge_index.storage._row, :]], dim=1)
        # edge_weight = self.assigner(feats_pair)
        edge_weight = assigner(x.detach(),edge_index)
        # print(edge_weight)
        
        
        for layer in self.disenlayers:
            x,xs,xus = layer(x, edge_index, edge_weight)
            xus = self.dropout(xus)
        return xus

class NeiborAssigner(nn.Module):
    def __init__(self, nfeats, nhidden,channels):
        super(NeiborAssigner, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=2 * nfeats, out_features=channels),
            nn.Linear(in_features=channels, out_features=channels)
        )
        

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, features_pair):
        alpha_score = self.layers(features_pair)
        # print(alpha_score.shape)
        alpha_score = torch.softmax(alpha_score, dim=1)
        print(alpha_score[0])
        return alpha_score
    
class NeighborAttention(nn.Module):
    def __init__(self, nfeats, channels, hidden_dim):
        super(NeighborAttention, self).__init__()
        self.nfeats = nfeats
        self.channels = channels
        self.hidden_dim = hidden_dim

        # Linear layers for MLP
        self.mlp = nn.Sequential(
            nn.Linear(nfeats * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, channels)
        )

    def forward(self, x, edge_index):
        row, col = edge_index

        # Concatenate the features of the source and target nodes
        edge_h = torch.cat([x[row], x[col]], dim=1)

        # Pass concatenated features through MLP
        attention_scores = self.mlp(edge_h)

        # Normalize attention scores for each node's neighbors
        attention_scores = torch.softmax(attention_scores, row, num_nodes=x.size(0))

        return attention_scores

# class channel_masker(nn.Module):
#     def __init__(self, hid_num):
#         super(channel_masker, self).__init__()
#         self.hid_num = hid_num
#         self.weights = nn.Parameter(torch.distributions.Uniform(0, 1).sample((hid_num, 2)))

#     def forward(self, x):
#         mask = F.gumbel_softmax(self.weights, tau=1, hard=False)[:, 0]
#         x = x * mask
#         return x
    
class NeiborAssigner1(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(NeiborAssigner1, self).__init__()
        self.body = GCN_Body(nfeat,nhid,dropout)
        self.fc = nn.Linear(nhid*2,nclass)

    def forward(self, x,edge_index):
        x = self.body(x,edge_index)

        feats_pair = torch.cat([x[edge_index.storage._col, :], x[edge_index.storage._row, :]], dim=1)
        x = self.fc(feats_pair)
        #print(x.shape)
        alpha_score = torch.softmax(x, dim=1)
        #print(alpha_score[0])
        return alpha_score
    
# def GCN(nn.Module):
class GCN_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN_Body, self).__init__()

        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x,edge_index):
        x = self.gc1(x, edge_index)
        x = self.dropout(x)
        
        # x = self.dropout(x)
        return x    
