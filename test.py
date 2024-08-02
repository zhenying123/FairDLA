import torch
from torch import nn
import matplotlib.pyplot as plt

# 数据
samples = [2, 5, 6, 10, 15]
aucroc = [88.66, 88.57, 88.46, 88.59, 88.35]
f1_score = [78.0, 77.76, 77.28, 78.24, 77.99]
acc = [83.77, 83.43, 82.92, 83.98, 83.77]
parity = [0.63, 0.41, 0.44, 0.96, 1.22]
equality = [1.33, 0.72, 0.69, 0.76, 0.94]

# 创建画布和双 y 轴
fig, ax1 = plt.subplots(figsize=(12, 8))

# 左 y 轴
ax1.plot(samples, parity, marker='o', color='b', label='Parity')
ax1.plot(samples, equality, marker='s', color='g', label='Equality')
ax1.set_xlabel('Sample')
ax1.set_ylabel('DP/EO', color='k')
ax1.tick_params(axis='y', labelcolor='k')
ax1.legend(loc='upper left')


# 右 y 轴
ax2 = ax1.twinx()
ax2.plot(samples, aucroc, marker='^', color='r', label='AUCROC')
ax2.plot(samples, f1_score, marker='v', color='m', label='F1-score')
ax2.plot(samples, acc, marker='D', color='c', label='ACC')
ax2.set_ylabel('AUCROC/F1-score/ACC', color='k')
ax2.tick_params(axis='y', labelcolor='k')
ax2.legend(loc='upper right')

# 调整横轴步数
ax1.set_xticks(range(min(samples), max(samples) + 1))

# 标题
plt.title('Metrics vs Sample with perturb_epsilon=0.1')

# 显示图像
plt.show()


# 保存图像
plt.savefig('/home/yzhen/code/fair/FairSAD_copy/plot_bail_0.1.png')

# 数据
samples = [2, 5, 6, 10, 15]
aucroc = [88.66, 88.57, 88.46, 88.59, 88.35]
f1_score = [78.0, 77.76, 77.28, 78.24, 77.99]
acc = [83.77, 83.43, 82.92, 83.98, 83.77]
parity = [0.63, 0.41, 0.44, 0.96, 1.22]
equality = [1.33, 0.72, 0.69, 0.76, 0.94]

# 创建画布和双 y 轴
fig, ax1 = plt.subplots(figsize=(12, 8))

# 左 y 轴
ax1.plot(samples, parity, marker='o', color='b', label='Parity')
ax1.plot(samples, equality, marker='s', color='g', label='Equality')
ax1.set_xlabel('Sample')
ax1.set_ylabel('DP/EO', color='k')
ax1.tick_params(axis='y', labelcolor='k')
ax1.legend(loc='upper left')


# 右 y 轴
ax2 = ax1.twinx()
ax2.plot(samples, aucroc, marker='^', color='r', label='AUCROC')
ax2.plot(samples, f1_score, marker='v', color='m', label='F1-score')
ax2.plot(samples, acc, marker='D', color='c', label='ACC')
ax2.set_ylabel('AUCROC/F1-score/ACC', color='k')
ax2.tick_params(axis='y', labelcolor='k')
ax2.legend(loc='upper right')

# 调整横轴步数
ax1.set_xticks(range(min(samples), max(samples) + 1))

# 标题
plt.title('Metrics vs Sample with perturb_epsilon=0.1')

# 显示图像
plt.show()


# 保存图像
plt.savefig('/home/yzhen/code/fair/FairSAD_copy/plot_bail_0.1.png')
# 显示图像



# class BinaryFocalLoss(nn.Module):
#     """
#     参考 https://github.com/lonePatient/TorchBlocks
#     """

#     def __init__(self, gamma=2.0, alpha=0.25, epsilon=1.e-9):
#         super(BinaryFocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.epsilon = epsilon

#     def forward(self, input, target):
#         """
#         Args:
#             input: model's output, shape of [batch_size, num_cls]
#             target: ground truth labels, shape of [batch_size]
#         Returns:
#             shape of [batch_size]
#         """
#         multi_hot_key = target
#         logits = input
#         # 如果模型没有做sigmoid的话，这里需要加上
#         # logits = torch.sigmoid(logits)
#         zero_hot_key = 1 - multi_hot_key
#         loss = -self.alpha * multi_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
#         loss += -(1 - self.alpha) * zero_hot_key * torch.pow(logits, self.gamma) * (1 - logits + self.epsilon).log()
#         return loss.mean()


# if __name__ == '__main__':
#     m = nn.Sigmoid()
#     loss = BinaryFocalLoss()
#     input = torch.randn(3, requires_grad=True)
#     target = torch.empty(3).random_(2)
#     print(target)
#     output = loss(m(input), target)
#     print("loss:", output)
#     output.backward()

# import torch
# import torch.nn.functional as F
# import numpy as np


# import numpy as np

# import numpy as np

# # Data
# import numpy as np

# # Data
# import numpy as np

# # Define the new data
# import torch
# import torch.nn.functional as F
# import numpy as np

# import torch
# import torch.nn.functional as F
# import numpy as np

# def compute_angle(v1, v2):
#     cos_theta = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
#     theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
#     return theta

# def slerp(v0, v1, t):
#     """Spherical linear interpolation."""
#     dot = torch.dot(v0 / torch.norm(v0), v1 / torch.norm(v1))
#     dot = np.clip(dot.item(), -1.0, 1.0)
    
#     theta = np.arccos(dot) * t
#     relative_vec = (v1 - v0 * dot)
#     relative_vec = relative_vec / torch.norm(relative_vec)
#     return ((v0 * np.cos(theta)) + (relative_vec * np.sin(theta)))

# # 创建两个16维向量
# a = torch.randn(16)
# b = torch.randn(16)

# # 计算夹角
# angle = compute_angle(a, b)

# # 使用Slerp进行旋转
# b_rotated = slerp(a, b, 0.5)

# # 向量加法
# c_add = a + b_rotated

# # 向量减法
# c_sub = a - b_rotated

# print(f"Original Vectors:\na: {a}\nb: {b}\n")
# print(f"Rotated Vector b:\nb_rotated: {b_rotated}\n")
# print(f"Vector Addition:\nc_add: {c_add}\n")
# print(f"Vector Subtraction:\nc_sub: {c_sub}\n")




# acc = np.array([76.31999850273132, 71.20000123977661,69.46666836738586, 77.26666927337646, 77.16000080108643])
# parity = np.array([0.583257809498372, 1.3489022176411791, 4.129301635555393, 1.2636861867989069, 1.0777835478721576])
# equality = np.array([1.1122102156636782, 0.9224160469926801, 3.503743369680745, 1.2119862317291563, 0.7240930587326555])
# epoch = np.array([502, 502, 502, 502, 502])
# aucroc = np.array([0.603620877918239, 0.5965835756653177, 0.6037905335266416, 0.5743649343735162, 0.592628247101536])
# f1 = np.array([0.8621118012422361, 0.8189741870600067, 0.8034672159285959, 0.8684312061115828, 0.8686248945471278])


# # Calculate means
# means = {
#     'acc': np.mean(acc),
#     'parity': np.mean(parity),
#     'equality': np.mean(equality),
#     'epoch': np.mean(epoch),
#     'aucroc': np.mean(aucroc),
#     'f1': np.mean(f1)
    
# }

# # Calculate variances
# variances = {
#     'acc': np.var(acc, ddof=1),
#     'parity': np.var(parity, ddof=1),
#     'equality': np.var(equality, ddof=1),
#     'epoch': np.var(epoch, ddof=1),
#     'aucroc': np.var(aucroc, ddof=1),
#     'f1': np.var(f1, ddof=1)
# }

# # Combine means and variances as mean ± standard deviation
# results = {
#     'acc': f"{means['acc']:.2f} ± {np.sqrt(variances['acc']):.2f}",
#     'parity': f"{means['parity']:.2f} ± {np.sqrt(variances['parity']):.2f}",
#     'equality': f"{means['equality']:.2f} ± {np.sqrt(variances['equality']):.2f}",
#     'epoch': f"{means['epoch']:.2f} ± {np.sqrt(variances['epoch']):.2f}",
#     'aucroc': f"{means['aucroc']:.4f} ± {np.sqrt(variances['aucroc']):.4f}",
#     'f1': f"{means['f1']:.4f} ± {np.sqrt(variances['f1']):.4f}"
    
# }

# print(results)


# # 模拟一个简单的编码器和分类器
# class SimpleEncoder(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(SimpleEncoder, self).__init__()
#         self.fc = torch.nn.Linear(input_dim, hidden_dim)

#     def forward(self, x):
#         return self.fc(x)

# class SimpleClassifier(torch.nn.Module):
#     def __init__(self, input_dim):
#         super(SimpleClassifier, self).__init__()
#         self.fc = torch.nn.Linear(input_dim, 1)

#     def forward(self, x):
#         return self.fc(x)

# # 模拟损失函数
# def criterion_bce(output, targets):
#     return F.binary_cross_entropy_with_logits(output, targets)

# # 模拟数据生成函数
# def generate_mock_data(num_samples, feature_dim):
#     features = torch.randn(num_samples, feature_dim)
#     labels = torch.randint(0, 2, (num_samples,))
#     sens_attr = torch.randint(0, 2, (num_samples,))
#     return features, labels, sens_attr

# # 增强数据函数
# @torch.no_grad()
# def augment_data(embed: torch.Tensor, y: torch.Tensor, sens_attr_vector: torch.Tensor, random_attack_num_samples, perturb_epsilon):
#     y_repeated = y.repeat_interleave(random_attack_num_samples)
#     noisy_latents = embed.repeat_interleave(random_attack_num_samples, dim=0).clone().detach()
#     coeffs = (2 * torch.rand(noisy_latents.shape[0], 1, device=noisy_latents.device) - 1) * perturb_epsilon
#     noisy_latents += sens_attr_vector * coeffs
#     return noisy_latents, y_repeated

# # 测试函数
# def test_weighted_loss():
#     random_attack_num_samples = 5
#     perturb_epsilon = 0.1
#     input_dim = 10
#     hidden_dim = 16
#     num_samples = 20

#     # 初始化模拟数据和模型
#     features, labels, sens_attr = generate_mock_data(num_samples, input_dim)
    
#     encoder = SimpleEncoder(input_dim, hidden_dim)
#     classifier = SimpleClassifier(hidden_dim)
    
#     # 计算sens_avg的隐藏表示
#     h_features = encoder(features)
#     sens_avg = torch.mean(h_features[sens_attr == 1], dim=0) - torch.mean(h_features[sens_attr == 0], dim=0)
    
#     optimizer_g = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=0.001)
    
#     # 模拟训练过程
#     encoder.train()
#     classifier.train()
#     optimizer_g.zero_grad()

#     h = encoder(features)

#     noisy_embeds, y_repeated = augment_data(h, labels, sens_avg, random_attack_num_samples, perturb_epsilon)
#     print(noisy_embeds.shape)
#     print(y_repeated.shape)

#     train_emb = torch.cat([h, noisy_embeds])
#     print(train_emb.shape)
#     y_targets = torch.cat([labels, y_repeated])

#     output = classifier(train_emb)

#     original_samples = h.repeat_interleave(random_attack_num_samples, dim=0)
#     distances = torch.norm(noisy_embeds - original_samples, p=2, dim=1)
#     normalized_distances = (distances - distances.min()) / (distances.max() - distances.min())
#     weights = torch.cat([torch.ones(h.size(0), device=distances.device), normalized_distances])
#     print(weights)

#     loss_cls_train = (weights * criterion_bce(output, y_targets.unsqueeze(1).float())).mean()

#     loss_cls_train.backward()
#     optimizer_g.step()

#     print("Weighted loss:", loss_cls_train.item())

# # 运行测试
# test_weighted_loss()









# def sample_batch_sen_idx(X, A, y, batch_size, s):
#     batch_idx = np.random.choice(np.where(A.cpu().numpy() == s)[0], size=batch_size, replace=False).tolist()
#     batch_x = X[batch_idx]
#     batch_y = y[batch_idx]
#     return batch_x, batch_y

# def fair_mixup(all_logit, labels, sens, model, alpha=2):
#     idx_s0 = np.where(sens.cpu().numpy() == 0)[0]
#     idx_s1 = np.where(sens.cpu().numpy() == 1)[0]

#     batch_size = min(len(idx_s0), len(idx_s1))

#     batch_logit_0, batch_y_0 = sample_batch_sen_idx(all_logit, sens, labels, batch_size, 0)
#     batch_logit_1, batch_y_1 = sample_batch_sen_idx(all_logit, sens, labels, batch_size, 1)
#     print(batch_logit_0.shape)

#     gamma = Beta(alpha, alpha).sample()
    
#     batch_logit_mix = batch_logit_0 * gamma + batch_logit_1 * (1 - gamma)
#     print(batch_logit_mix.shape)
#     batch_logit_mix = batch_logit_mix.requires_grad_(True)

#     output = F.softmax(batch_logit_mix, dim=1)
#     print(output.shape)

#     gradx = torch.autograd.grad(output.sum(), batch_logit_mix, create_graph=True)[0]

#     batch_logit_d = batch_logit_1 - batch_logit_0
#     grad_inn = (gradx * batch_logit_d).sum(1)
#     E_grad = grad_inn.mean(0)
#     loss_reg = torch.abs(E_grad)

#     return loss_reg

# # 创建一个简单的测试用例
# def test_fair_mixup():
#     torch.manual_seed(42)  # 设置随机种子以确保可重复性

#     # 创建示例数据
#     num_samples = 10
#     num_features = 5
#     num_classes = 2
    
#     all_logit = torch.randn(num_samples, num_features, requires_grad=True)
#     print(all_logit)
#     labels = torch.randint(0, num_classes, (num_samples,))
#     sens = torch.randint(0, 2, (num_samples,))
#     print(sens.shape)

#     class SimpleModel(torch.nn.Module):
#         def forward(self, x):
#             return x

#     model = SimpleModel()

#     # 调用 fair_mixup 并输出结果
#     loss_reg = fair_mixup(all_logit, labels, sens, model)
#     print("Regularization Loss:", loss_reg.item())

# # 运行测试用例
# test_fair_mixup()







# class DataAugmentor:
#     def __init__(self, random_attack_num_samples=10, perturb_epsilon=0.1):
#         self.random_attack_num_samples = random_attack_num_samples
#         self.perturb_epsilon = perturb_epsilon

#     @torch.no_grad()
#     def augment_data(self, embed: torch.Tensor, y: torch.Tensor, sens_attr_vectors: dict):
#         assert y.dim() == 1 and sens_attr_vectors is not None

#         y_repeated = y.repeat_interleave(self.random_attack_num_samples)
#         assert embed.dim() == 2 and embed.size(0) == y.size(0)

#         noisy_latents = embed.repeat_interleave(self.random_attack_num_samples, dim=0).clone().detach()
#         coeffs = (2 * torch.rand(noisy_latents.shape[0], 1, device=noisy_latents.device) - 1) * self.perturb_epsilon

#         sens_attr_vector_tensor = torch.zeros_like(noisy_latents)
#         for label in sens_attr_vectors.keys():
#             mask = (y_repeated == label)
#             sens_attr_vector_tensor[mask] = sens_attr_vectors[label]

#         noisy_latents += sens_attr_vector_tensor * coeffs

#         return noisy_latents, y_repeated

# def test_augment_data():
#     # Create a DataAugmentor instance
#     augmentor = DataAugmentor(random_attack_num_samples=5, perturb_epsilon=0.1)

#     # Example embeddings
#     embed = torch.randn(10, 5)  # 10 samples, 5-dimensional embeddings

#     # Example labels
#     y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

#     # Example sensitive attribute vectors
#     sens_attr_vectors = {
#         0: torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
#         1: torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1])
#     }

#     # Call augment_data method
#     noisy_latents, y_repeated = augmentor.augment_data(embed, y, sens_attr_vectors)

#     # Print the results
#     print("Original Embeddings:\n", embed.shape)
#     print("Noisy Latents:\n", noisy_latents.shape)
#     print("y:\n", y.shape)
#     print("Repeated Labels:\n", y_repeated.shape)

# # Run the test function
# test_augment_data()

# @torch.no_grad()
# def compute_attribute_vectors_avg_diff(embed: torch.Tensor, sens: torch.Tensor, labels: torch.Tensor, 
#                                        sample=False, norm=False) -> dict:
#     unique_labels = labels.unique()
#     attr_vectors_diff = {}

#     for label in unique_labels:
#         label_mask = (labels == label).long()
        
#         pos_mask = (sens == 1).long()
#         neg_mask = (sens == 0).long()
        
#         pos_combined_mask = label_mask * pos_mask
#         neg_combined_mask = label_mask * neg_mask

#         if sample:
#             sample_num = min(pos_combined_mask.sum(), neg_combined_mask.sum()).item()
#             np.random.seed(0)
#             negative_indices = torch.nonzero(neg_combined_mask).squeeze(1).detach().cpu().numpy()
#             random_sample_negative_indices = np.random.choice(negative_indices, sample_num, replace=False)
#             sampled_neg_mask = torch.zeros_like(neg_combined_mask)
#             sampled_neg_mask[random_sample_negative_indices] = 1
#             neg_combined_mask = sampled_neg_mask

#         cnt_pos = pos_combined_mask.count_nonzero().item()
#         cnt_neg = neg_combined_mask.count_nonzero().item()
#         if cnt_pos == 0 or cnt_neg == 0:
#             continue

#         z_pos_per_attribute = torch.sum(embed * pos_combined_mask.unsqueeze(1), 0)
#         z_neg_per_attribute = torch.sum(embed * neg_combined_mask.unsqueeze(1), 0)
#         attr_vector_diff = (z_pos_per_attribute / cnt_pos) - (z_neg_per_attribute / cnt_neg)
        

#         if norm:
#             l2_norm = math.sqrt((attr_vector_diff * attr_vector_diff).sum())
#             attr_vector_diff /= l2_norm

#         attr_vectors_diff[label.item()] = attr_vector_diff

#     return attr_vectors_diff




# # 示例数据
# embed = torch.randn(100, 50)  # 100 samples, 50-dim embedding
# sens = torch.randint(0, 2, (100,))  # 0 or 1 sensitive attribute
# labels = torch.randint(0, 3, (100,))  # 3 classes

# # 计算每类y的不同s的平均值列表
# attribute_vectors = compute_attribute_vectors_avg_diff(embed, sens, labels, sample=True, norm=True)
# print(attribute_vectors)

