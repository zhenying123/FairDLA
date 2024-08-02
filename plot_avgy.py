import numpy as np
import matplotlib.pyplot as plt


# Credit
# copy=0, τ=0.5, γ=0.5, lr_w=1 , sample=15,perturb_epsilon=0.4,epoch=600,avgy=False,hiiden=16 ,channel=2,lr=0.001,adv=1
# AUCROC: 66.22 ± 1.66
# F1-score: 87.23 ± 0.64
# ACC: 78.0 ± 0.59
# Parity: 1.44 ± 1.48
# Equality: 0.9 ± 1.12
# copy=0, τ=0.5, γ=0.5, lr_w=1 , sample=15,perturb_epsilon=0.4,epoch=600,avgy=True,hiiden=16 ,channel=2,lr=0.001,adv=1
# AUCROC: 67.57 ± 0.97
# F1-score: 87.68 ± 0.07
# ACC: 78.35 ± 0.12
# Parity: 0.62 ± 0.66
# Equality: 0.35 ± 0.22

# pokec-n
# copy=0, τ=0.5, γ=0.05, lr_w=1 , sample=3,perturb_epsilon=0.7,epoch=600,avgy=False,hiiden=16 ,channel=2,lr=0.001,adv=0
# AUCROC: 72.65 ± 0.13
# F1-score: 62.04 ± 0.51
# ACC: 67.07 ± 0.09
# Parity: 1.88 ± 0.4
# Equality: 0.76 ± 0.71
# copy=0, τ=0.5, γ=0.05, lr_w=1 , sample=3,perturb_epsilon=0.7,epoch=600,avgy=True,hiiden=16 ,channel=2,lr=0.001,adv=0
# AUCROC: 72.73 ± 0.15
# F1-score: 62.54 ± 0.37
# ACC: 67.45 ± 0.25
# Parity: 1.16 ± 0.48
# bail
# Equality: 0.59 ± 0.41,copy=0, τ=0.5, γ=0.001, lr_w=1 , sample=2,perturb_epsilon=0.6,epoch=600,avgy=True,hiiden=16 ,channel=2,lr=0.001,adv=0
# AUCROC: 88.63 ± 0.0
# F1-score: 78.23 ± 0.0
# ACC: 83.94 ± 0.0
# Parity: 0.54 ± 0.0
# Equality: 1.08 ± 0.0
# copy=0, τ=0.5, γ=0.001, lr_w=1 , sample=2,perturb_epsilon=0.6,epoch=600,avgy=False,hiiden=16 ,channel=2,lr=0.001,adv=0
# AUCROC: 88.11 ± 0.0
# F1-score: 77.43 ± 0.0
# ACC: 83.51 ± 0.0
# Parity: 0.7 ± 0.0
# Equality: 1.23 ± 0.0绘制这三个数据集上，关于是否设置avgy为true的柱状图
# 数据整理
# 数据整理
import matplotlib.pyplot as plt
import numpy as np

# 设置整体字体属性
# plt.rcParams.update({'font.size': 18, 'font.family': 'sans-serif'})

# datasets = ['Bail', 'Credit', 'Pokec-n']
# metrics = ['AUC ↑', 'F1 ↑', 'ACC ↑', '$\\Delta_{DP}$ ↓', '$\\Delta_{EO}$ ↓']
# accuracy_metrics = metrics[:3]
# fairness_metrics = metrics[3:]
# avgy_settings = ['avgy=False', 'avgy=True']

# # Bail 数据集
# bail_avgy_false = [88.11, 77.43, 83.51, 0.7, 1.23]
# bail_avgy_false_std = [0.12, 0.53, 0.77, 0.74, 0.48]
# bail_avgy_true = [88.63, 78.23, 83.94, 0.54, 1.08]
# bail_avgy_true_std = [0.22, 0.34, 0.63, 0.32, 0.42]

# # Credit 数据集
# credit_avgy_false = [66.22, 87.23, 78.0, 1.44, 0.9]
# credit_avgy_false_std = [1.66, 0.64, 0.59, 0.48, 0.82]
# credit_avgy_true = [67.57, 87.68, 78.35, 0.62, 0.35]
# credit_avgy_true_std = [0.97, 0.07, 0.12, 0.46, 0.22]

# # Pokec-n 数据集
# pokecn_1_avgy_false = [72.65, 62.04, 67.07, 1.88, 0.76]
# pokecn_1_avgy_false_std = [0.13, 0.51, 0.09, 0.4, 0.71]
# pokecn_1_avgy_true = [72.73, 62.54, 67.45, 1.16, 0.59]
# pokecn_1_avgy_true_std = [0.15, 0.37, 0.25, 0.48, 0.41]

import matplotlib.pyplot as plt
import numpy as np

# 设置整体字体属性为 sans-serif
plt.rcParams.update({'font.size': 25, 'font.family': 'serif'})

datasets = [ 'Credit', 'Pokec-n']
metrics = ['AUC', 'F1', '$\\Delta_{DP}$ ↓', '$\\Delta_{EO}$ ↓']
accuracy_metrics = metrics[:2]
fairness_metrics = metrics[2:]
avgy_settings = ['avgy=False', 'avgy=True']

# Bail 数据集
bail_avgy_false = [88.11, 77.43, 0.7, 1.23]
bail_avgy_false_std = [0.12, 0.53, 0.74, 0.48]
bail_avgy_true = [88.63, 78.23, 0.54, 1.08]
bail_avgy_true_std = [0.22, 0.34, 0.32, 0.42]

# Credit 数据集
credit_avgy_false = [66.22, 87.23, 1.44, 0.9]
credit_avgy_false_std = [1.66, 0.64, 0.48, 0.82]
credit_avgy_true = [67.57, 87.68, 0.62, 0.35]
credit_avgy_true_std = [0.97, 0.07, 0.46, 0.22]

# Pokec-n 数据集
pokecn_1_avgy_false = [72.65, 62.04, 1.88, 0.76]
pokecn_1_avgy_false_std = [0.13, 0.51, 0.4, 0.71]
pokecn_1_avgy_true = [72.73, 62.54, 1.16, 0.59]
pokecn_1_avgy_true_std = [0.15, 0.37, 0.48, 0.41]

# 整合数据
data = {
    # 'Bail': {
    #     'avgy=False': (bail_avgy_false, bail_avgy_false_std),
    #     'avgy=True': (bail_avgy_true, bail_avgy_true_std)
    # },
    'Credit': {
        'avgy=False': (credit_avgy_false, credit_avgy_false_std),
        'avgy=True': (credit_avgy_true, credit_avgy_true_std)
    },
    'Pokec-n': {
        'avgy=False': (pokecn_1_avgy_false, pokecn_1_avgy_false_std),
        'avgy=True': (pokecn_1_avgy_true, pokecn_1_avgy_true_std)
    }
}

# 创建画布
fig, axs = plt.subplots(1, 2, figsize=(16.5, 6.5))

# 颜色定义
colors = ['lightpink', 'skyblue']

# 绘制每个数据集的柱状图
for i, dataset in enumerate(datasets):
    ax1 = axs[i]
    ax2 = ax1.twinx()
    for j, avgy in enumerate(avgy_settings):
        means, stds = data[dataset][avgy]
        ax1.bar(np.arange(len(accuracy_metrics)) * 0.85 + j*0.3, means[:2], width=0.3, color=colors[j], align='center', edgecolor='black', yerr=stds[:2], capsize=3, label=avgy if i == 0 else "")
        ax2.bar(np.arange(len(fairness_metrics)) * 0.85 + j*0.3 + len(accuracy_metrics) * 0.85, means[2:], width=0.3, color=colors[j], align='center', edgecolor='black', yerr=stds[2:], capsize=3, label=avgy if i == 0 else "")
    
    ax1.set_title(dataset, fontsize=30)
    ax1.set_xticks(np.arange(len(metrics)) * 0.85 + 0.3)
    ax1.set_xticklabels(metrics, fontsize=25)
    if i==0:
        ax1.set_ylabel('Utility Metrics', fontsize=22)
        ax2.set_ylabel('Fairness Metrics', fontsize=22)
    
    if i == 0:
        ax1.legend(fontsize=20, loc='upper right', ncol=4)

    # 调整y轴范围
    ax1.set_ylim([min(min(means[:2]) for means, _ in data[dataset].values()) * 0.95, max(max(means[:2]) for means, _ in data[dataset].values()) * 1.05])
    ax2.set_ylim([min(min(means[2:]) for means, _ in data[dataset].values()) * 0.55, max(max(means[2:]) for means, _ in data[dataset].values()) * 1.45])


plt.tight_layout(rect=[0, 0, 1, 1,])
plt.subplots_adjust(wspace=0.35)
plt.show()




plt.savefig('/home/yzhen/code/fair/FairSAD_copy/comparison_avgy_cp.png', dpi=500)
plt.show()
# import numpy as np
# import matplotlib.pyplot as plt

# # 数据整理
# datasets = ['Bail', 'Credit', 'Pokec-n']
# metrics = ['AUCROC', 'F1-score', 'Parity']
# accuracy_metrics = metrics[:2]
# fairness_metrics = metrics[2:]
# avgy_settings = ['avgy=False', 'avgy=True']

# # Bail 数据集
# bail_avgy_false = [88.11, 77.43, 0.4]
# bail_avgy_true = [88.63, 78.23, 0.54]

# # Credit 数据集
# credit_avgy_false = [66.22, 87.23, 1.44]
# credit_avgy_true = [67.57, 87.68, 0.62]

# # pokec-n 数据集
# pokecn_1_avgy_false = [72.65, 62.04, 1.88]
# pokecn_1_avgy_true = [72.73, 62.54, 1.16]

# # 整合数据
# data = {
#     'Bail': {
#         'avgy=False': bail_avgy_false,
#         'avgy=True': bail_avgy_true
#     },
#     'Credit': {
#         'avgy=False': credit_avgy_false,
#         'avgy=True': credit_avgy_true
#     },
#     'Pokec-n': {
#         'avgy=False': pokecn_1_avgy_false,
#         'avgy=True': pokecn_1_avgy_true
#     }
# }

# # 创建画布
# fig, axs = plt.subplots(1, 2, figsize=(24, 8))

# # 颜色定义
# colors = ['#FF69B4', '#6495ED']
# bar_width = 0.2

# # 绘制准确性指标
# for i, (dataset, results) in enumerate(data.items()):
#     for j, avgy in enumerate(avgy_settings):
#         axs[0].bar(np.arange(len(accuracy_metrics)) + i*bar_width*2 + j*bar_width, results[avgy][:2], 
#                    width=bar_width, color=colors[j], align='center', 
#                    edgecolor='black', linewidth=1, 
#                    label=f"{dataset} {avgy}" if j == 0 else "")

# axs[0].set_title('Accuracy Metrics Comparison (AUC and F1-score)', fontsize=16)
# axs[0].set_xticks(np.arange(len(accuracy_metrics)) + bar_width)
# axs[0].set_xticklabels(accuracy_metrics, fontsize=14)
# axs[0].set_ylabel('Accuracy Metrics', fontsize=14)
# axs[0].legend(fontsize=12)
# axs[0].tick_params(axis='y', labelsize=14)

# # 绘制公平性指标
# for i, (dataset, results) in enumerate(data.items()):
#     for j, avgy in enumerate(avgy_settings):
#         axs[1].bar(np.arange(len(fairness_metrics)) + i*bar_width*2 + j*bar_width, results[avgy][2:], 
#                    width=bar_width, color=colors[j], align='center', 
#                    edgecolor='black', linewidth=1, 
#                    label=f"{dataset} {avgy}" if j == 0 else "")

# axs[1].set_title('Fairness Metrics Comparison (Parity)', fontsize=16)
# axs[1].set_xticks(np.arange(len(fairness_metrics)) + bar_width)
# axs[1].set_xticklabels(fairness_metrics, fontsize=14)
# axs[1].set_ylabel('Fairness Metrics', fontsize=14)
# axs[1].legend(fontsize=12)
# axs[1].tick_params(axis='y', labelsize=14)

# # 设置整体图的标题和标签
# fig.suptitle('Comparison of Metrics with avgy=True and avgy=False', fontsize=18)
# plt.tight_layout(rect=[0, 0, 1, 0.96])

# plt.savefig('/home/yzhen/code/fair/FairSAD_copy/comparison_avgy1.png', dpi=500)
# plt.show()

