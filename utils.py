import math
import torch
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import scipy.sparse as sp
from scipy.spatial import distance_matrix
import pandas as pd
import torch.nn.functional as F

def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh * max_sim)[0]
        import random
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])
    idx_map = np.array(idx_map)
    return idx_map


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2 * (features - min_values).div(max_values - min_values) - 1


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()

@torch.no_grad()
def compute_attribute_vectors_avg_diff(embed: torch.Tensor, sens: torch.Tensor, sample=False,
                                       norm=False) -> torch.Tensor:
    pos_mask = (sens == 1).long()
    neg_mask = 1 - pos_mask
    def compute_angle(v1, v2):
        # Ensure the vectors are normalized to prevent numerical issues
        v1_norm = v1 / torch.norm(v1)
        v2_norm = v2 / torch.norm(v2)
        cos_sim = torch.dot(v1_norm, v2_norm)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)  # Clamp values to the valid range for acos
        angle = torch.acos(cos_sim) * (180.0 / torch.pi)  # Convert from radians to degrees
        return angle.item()  # Convert Tensor to float for readability
    if sample:
        neg_sample_num = min(pos_mask.sum(), neg_mask.sum()).item()
        np.random.seed(0)
        negative_indices = torch.nonzero(neg_mask).squeeze(1).detach().cpu().numpy()
        random_sample_negative_indices = np.random.choice(negative_indices, neg_sample_num, replace=False)
        neg_mask = torch.zeros_like(neg_mask)
        neg_mask[random_sample_negative_indices] = 1

    cnt_pos = pos_mask.count_nonzero().item()
    cnt_neg = neg_mask.count_nonzero().item()
    z_pos_per_attribute = torch.sum(embed * pos_mask.unsqueeze(1), 0)
    z_neg_per_attribute = torch.sum(embed * neg_mask.unsqueeze(1), 0)
    attr_vector = ((z_pos_per_attribute / cnt_pos) - (z_neg_per_attribute / cnt_neg))
    angle = compute_angle(z_pos_per_attribute, z_neg_per_attribute)
    print(f'Angle between z_pos_per_attribute and z_neg_per_attribute: {angle} degrees')
    l2_norm = math.sqrt((attr_vector * attr_vector).sum())
    if norm is True:
        attr_vector /= l2_norm
    return attr_vector

# fairmixup的代码实现能够实现梯度计算尝试对比？
def sample_batch_sen_idx(X, A, y, batch_size, s):    
    # print(f'popu={np.where(A.cpu().numpy()==s)[0]}')
    batch_idx = np.random.choice(np.where(A.cpu().numpy()==s)[0], size=batch_size, replace=False).tolist()
    batch_x = X[batch_idx]
    batch_y = y[batch_idx]
    # batch_x = torch.tensor(batch_x).cuda().float()
    # batch_y = torch.tensor(batch_y).cuda().float()

    return batch_x, batch_y

def fair_mixup(all_logit, labels, sens, model, alpha = 2):
    idx_s0 = np.where(sens.cpu().numpy()==0)[0]
    idx_s1 = np.where(sens.cpu().numpy()==1)[0]

    # print(f'idx_s0={len(idx_s0)}')
    # print(f'idx_s1={len(idx_s1)}')
    batch_size = min(len(idx_s0), len(idx_s1))

    batch_logit_0, batch_y_0 = sample_batch_sen_idx(all_logit, sens, labels, batch_size, 0)
    batch_logit_1, batch_y_1 = sample_batch_sen_idx(all_logit, sens, labels, batch_size, 1)

    
    gamma = beta(alpha, alpha)

    batch_logit_mix = batch_logit_0 * gamma + batch_logit_1 * (1 - gamma)
    batch_logit_mix = batch_logit_mix.requires_grad_(True)

    output = F.softmax(batch_logit_mix, dim=1)

    # gradient regularization
    gradx = torch.autograd.grad(output.sum(), batch_logit_mix, create_graph=True)[0]

    batch_logit_d = batch_logit_1 - batch_logit_0
    grad_inn = (gradx * batch_logit_d).sum(1)
    E_grad = grad_inn.mean(0)
    loss_reg = torch.abs(E_grad)

    return loss_reg

@torch.no_grad()
def compute_attribute_vectors_avg_diff_y(embed: torch.Tensor, sens: torch.Tensor, labels: torch.Tensor, 
                                       sample=False, norm=False) -> dict:
    unique_labels = labels.unique()
    attr_vectors_diff = {}
    def compute_angle(v1, v2):
        # Ensure the vectors are normalized to prevent numerical issues
        v1_norm = v1 / torch.norm(v1)
        v2_norm = v2 / torch.norm(v2)
        cos_sim = torch.dot(v1_norm, v2_norm)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)  # Clamp values to the valid range for acos
        angle = torch.acos(cos_sim) * (180.0 / torch.pi)  # Convert from radians to degrees
        return angle.item()  # Convert Tensor to float for readability

    for label in unique_labels:
        print(f'label={label}')
        label_mask = (labels == label).long()
        
        pos_mask = (sens == 1).long()
        neg_mask = (sens == 0).long()
        
        pos_combined_mask = label_mask * pos_mask
        neg_combined_mask = label_mask * neg_mask

        if sample:
            sample_num = min(pos_combined_mask.sum(), neg_combined_mask.sum()).item()
            np.random.seed(0)
            negative_indices = torch.nonzero(neg_combined_mask).squeeze(1).detach().cpu().numpy()
            random_sample_negative_indices = np.random.choice(negative_indices, sample_num, replace=False)
            sampled_neg_mask = torch.zeros_like(neg_combined_mask)
            sampled_neg_mask[random_sample_negative_indices] = 1
            neg_combined_mask = sampled_neg_mask

        cnt_pos = pos_combined_mask.count_nonzero().item()
        cnt_neg = neg_combined_mask.count_nonzero().item()
        if cnt_pos == 0 or cnt_neg == 0:
            continue

        z_pos_per_attribute = torch.sum(embed * pos_combined_mask.unsqueeze(1), 0)
        z_neg_per_attribute = torch.sum(embed * neg_combined_mask.unsqueeze(1), 0)

        attr_vector_diff = (z_pos_per_attribute / cnt_pos) - (z_neg_per_attribute / cnt_neg)
        
        
        angle = compute_angle(z_pos_per_attribute, z_neg_per_attribute)
        print(f'Angle between z_pos_per_attribute and z_neg_per_attribute: {angle} degrees')
        if norm:
            l2_norm = math.sqrt((attr_vector_diff * attr_vector_diff).sum())
            attr_vector_diff /= l2_norm
            

        attr_vectors_diff[label.item()] = attr_vector_diff

    return attr_vectors_diff
@torch.no_grad()
def calculate_angles(attr_vectors_diff: dict):
    angles = {}
    for label, vector in attr_vectors_diff.items():
        for other_label, other_vector in attr_vectors_diff.items():
            if label != other_label:
                cosine_similarity = F.cosine_similarity(vector.unsqueeze(0), other_vector.unsqueeze(0))
                cosine_similarity = torch.clamp(cosine_similarity, -1.0, 1.0)
                angle = torch.acos(cosine_similarity).item()
                angles[(label, other_label)] = angle
    return angles
class Results:
    def __init__(self, seed_num, model_num, args):
        super(Results, self).__init__()

        self.seed_num = seed_num
        self.model_num = model_num
        self.dataset = args.dataset
        self.model = args.model
        self.auc, self.f1, self.acc, self.parity, self.equality = np.zeros(shape=(self.seed_num, self.model_num)), \
                                                                  np.zeros(shape=(self.seed_num, self.model_num)), \
                                                                  np.zeros(shape=(self.seed_num, self.model_num)), \
                                                                  np.zeros(shape=(self.seed_num, self.model_num)), \
                                                                  np.zeros(shape=(self.seed_num, self.model_num))
        

    def report_results(self):
        for i in range(self.model_num):
            print(f"============" + f"{self.dataset}" + "+" + f"{self.model}" + "============")
            print(f"AUCROC: {np.around(np.mean(self.auc[:, i]) * 100, 2)} ± {np.around(np.std(self.auc[:, i]) * 100, 2)}")
            print(f'F1-score: {np.around(np.mean(self.f1[:, i]) * 100, 2)} ± {np.around(np.std(self.f1[:, i]) * 100, 2)}')
            print(f'ACC: {np.around(np.mean(self.acc[:, i]) * 100, 2)} ± {np.around(np.std(self.acc[:, i]) * 100, 2)}')
            print(f'Parity: {np.around(np.mean(self.parity[:, i]) * 100, 3)} ± {np.around(np.std(self.parity[:, i]) * 100, 3)}')
            print(f'Equality: {np.around(np.mean(self.equality[:, i]) * 100, 3)} ± {np.around(np.std(self.equality[:, i]) * 100, 3)}')
            print("=================END=================")



    def save_results(self, args):
        for i in range(self.model_num):
            with open(f"{args.dataset}_tao{args.tem}_model{i}.txt", 'a') as f:
                f.write(f"copy={args.copy}, τ={args.tem}, γ={args.alpha}, lr_w={args.lr_w} , sample={args.rs},perturb_epsilon={args.per},epoch={args.epochs},avgy={args.avgy},hiiden={args.hidden} ,channel={args.channels},lr={args.lr},adv={args.adv}\n")
                f.write(f"AUCROC: {np.around(np.mean(self.auc[:, i]) * 100, 2)} ± {np.around(np.std(self.auc[:, i]) * 100, 2)}\n")
                f.write(f'F1-score: {np.around(np.mean(self.f1[:, i]) * 100, 2)} ± {np.around(np.std(self.f1[:, i]) * 100, 2)}\n')
                f.write(f'ACC: {np.around(np.mean(self.acc[:, i]) * 100, 2)} ± {np.around(np.std(self.acc[:, i]) * 100, 2)}\n')
                f.write(f'Parity: {np.around(np.mean(self.parity[:, i]) * 100, 2)} ± {np.around(np.std(self.parity[:, i]) * 100, 2)}\n')
                f.write(f'Equality: {np.around(np.mean(self.equality[:, i]) * 100, 2)} ± {np.around(np.std(self.equality[:, i]) * 100, 2)}\n')
            f.close()

# 我需要的应该是这个，计算特征和敏感特征之间的相关损失
class FeatCov(_Loss):
    def __init__(self):
        super(FeatCov, self).__init__()

    def forward(self, features, sens):
        cov = 0
        for k in range(features.shape[1]):
            cov += torch.abs(torch.mean((sens - torch.mean(sens)) * (features[:, k] - torch.mean(features[:, k]))))
        return cov


class DistCor(_Loss):
    def __init__(self):
        super(DistCor, self).__init__()

    def Distance_Correlation(self, channel1, channel2):
        assert channel1.shape[1] == channel2.shape[1], "The dim of two tensors need to be equal"
        dim_num = channel1.shape[1]
        correlation_r = 0
        for i in range(dim_num):
            latent = channel1[:, i]
            latent = latent.unsqueeze(1)
            control = channel2[:, i]
            control = control.unsqueeze(1)

            matrix_a = torch.sqrt(torch.sum(torch.square(latent.unsqueeze(0) - latent.unsqueeze(1)), dim = -1) + 1e-12)
            matrix_b = torch.sqrt(torch.sum(torch.square(control.unsqueeze(0) - control.unsqueeze(1)), dim = -1) + 1e-12)

            matrix_A = matrix_a - torch.mean(matrix_a, dim = 0, keepdims= True) - torch.mean(matrix_a, dim = 1, keepdims= True) + torch.mean(matrix_a)
            matrix_B = matrix_b - torch.mean(matrix_b, dim = 0, keepdims= True) - torch.mean(matrix_b, dim = 1, keepdims= True) + torch.mean(matrix_b)

            Gamma_XY = torch.sum(matrix_A * matrix_B)/ (matrix_A.shape[0] * matrix_A.shape[1])
            Gamma_XX = torch.sum(matrix_A * matrix_A)/ (matrix_A.shape[0] * matrix_A.shape[1])
            Gamma_YY = torch.sum(matrix_B * matrix_B)/ (matrix_A.shape[0] * matrix_A.shape[1])

            correlation_r += Gamma_XY/torch.sqrt(Gamma_XX * Gamma_YY + 1e-9)
        return correlation_r

    def forward(self, channel1, channel2):
        dc_loss = self.Distance_Correlation(channel1, channel2)
        return dc_loss