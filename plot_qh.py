import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
plt.rcParams.update({'font.size': 20, 'font.family': 'serif'})
datasets = ["German", "Bail", "Credit", "Pokec-z", "Pokec-n"]

methods = ["GCN", "NIEFTY", "FairGNN", "FairVGNN", "FuGNN", "FairSAD", "Ours"]

acc_values = {
    "GCN": [71.20, 83.98, 74.58, 69.36, 67.51],
    "NIEFTY": [70.4, 77.68, 74.54, 64.47, 65.57],
    "FairGNN": [69.32, 82.94, 73.41, 67.89, 65.27],
    "FairVGNN": [69.84, 85.43, 77.89, 68.11, 66.10],
    "FuGNN": [70.32, 90.86, 76.28, 68.30, 66.60],
    "FairSAD": [69.91, 83.84, 77.7, 69.32, 67.21],
    "Ours": [70.08, 84.16, 78.33, 69.16, 67.34]
}
f1_values = {
    "GCN": [80.01, 78.91, 84.59, 70.20, 65.23],
    "NIEFTY": [81.12, 69.23, 83.51, 68.56, 60.21],
    "FairGNN": [81.44, 77.50, 81.84, 68.82, 63.22],
    "FairVGNN": [81.51, 79.69, 87.55, 68.34, 62.35],
    "FuGNN": [82.51, 87.90, 85.43, 68.28, 62.14],
    "FairSAD": [82.30, 78.22, 87.02, 69.03, 62.62],
    "Ours": [82.61, 79.31, 87.64, 70.62, 62.35]
}

eo_values = {
    "GCN": [28.78, 4.87, 9.61, 4.95, 11.23],
    "NIEFTY": [2.44, 2.67, 6.47, 2.86, 7.27],
    "FairGNN": [3.40, 4.65, 3.97, 2.15, 3.11],
    "FairVGNN": [3.05, 4.95, 1.31, 2.12, 4.74],
    "FuGNN": [0.34, 1.99, 1.02, 1.37, 1.17],
    "FairSAD": [0.05, 1.79, 1.02, 1.40, 2.95],
    "Ours": [0.01, 0.98, 0.52, 1.06, 0.67]
}
dp_values = {
    "GCN": [36.67, 7.43, 11.47, 4.17, 7.24],
    "NIEFTY": [2.54, 3.57, 8.56, 3.51, 5.66],
    "FairGNN": [3.49, 6.72, 5.41, 2.79, 3.31],
    "FairVGNN": [2.53, 6.65, 2.8, 1.89, 3.22],
    "FuGNN": [0.51, 5.76, 1.27, 0.88, 1.28],
    "FairSAD": [0.25, 1.82, 2.01, 0.97, 1.99],
    "Ours": [0.21, 0.68, 0.76, 1.14, 1.0]
}
# 每个数据集的 AUC 和 ΔEo 值
# data = {
#     "German": (acc_values, eo_values),
#     "Bail": (acc_values, eo_values),
#     "Credit": (acc_values, eo_values),
#     "Pokec-z": (acc_values, eo_values),
#     "Pokec-n": (acc_values, eo_values)
# }
# data = {
#     "German": (acc_values, dp_values),
#     "Bail": (acc_values, dp_values),
#     "Credit": (acc_values, dp_values),
#     "Pokec-z": (acc_values, dp_values),
#     "Pokec-n": (acc_values, dp_values)
# }

fig, axes = plt.subplots(1, 5, figsize=(25.5, 5.5), sharey=False)

# 使用不同的颜色和标记为每个方法绘制点
markers = ['o', 'v', 's', 'p', '*', 'X', 'D']
colors = ['b', 'g', 'k', 'c', 'm', 'y', 'r']
sizes = [140, 140, 140, 140, 140, 140, 150]
# 收集所有图例元素
handles = []
labels = []

for i, dataset in enumerate(datasets):
    ax = axes[i]
    
    # 获取当前数据集的 F1 和 ΔEo 数据
    f1_values_list = [f1_values[method][i] for method in methods]
    eo_values_list = [eo_values[method][i] for method in methods]
    
    # 设置横轴和纵轴的范围
    f1_min = min(f1_values_list) - 1
    f1_max = max(f1_values_list) + 1
    eo_min = min(eo_values_list) - 1
    eo_max = max(eo_values_list) + 1
    
    ax.set_xlim(f1_min, f1_max)
    ax.set_ylim(eo_min, eo_max)
    
    # 绘制散点图
    # 绘制散点图
    for j, method in enumerate(methods):
        sc = ax.scatter(f1_values[method][i], eo_values[method][i], 
                        label=method, marker=markers[j], color=colors[j], s=sizes[j])
        # 仅在第一个子图中添加图例元素
        if i == 0:
            handles.append(sc)
            labels.append(method)

    # 添加子图标题和标签
    ax.set_title(dataset, fontsize=26)
    ax.set_xlabel("ACC(%)", fontsize=28)
    if i == 0:  # 仅在第一个子图上显示 y 轴标签
        
        ax.set_ylabel("$\\Delta_{EO}$ (↓)", fontsize=28)
        

        
    ax.invert_yaxis()
# 添加浅色网格线
    ax.grid(True, linestyle='-', color='grey')

    # # 控制网格线密度
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))  # 主网格线间隔
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))  # 次网格线间隔
    # ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    # ax.grid(which='minor', linestyle=':', linewidth='0.5', color='lightgrey')  # 设置次网格线样式
# 在整体图上方添加图例
fig.legend(handles=handles[:7], labels=labels[:7], loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=len(methods), fontsize=25)

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.9])  # 为图例留出更多空间   

# 调整布局

plt.show()

plt.savefig('/home/yzhen/code/fair/FairSAD_copy/xin-acc_vs_eo_tradeoff.png', dpi=400)


fig, axes = plt.subplots(1, 5, figsize=(25.5, 5.5), sharey=False)

handles = []
labels = []

# 使用不同的颜色和标记为每个方法绘制点
markers = ['o', 'v', 's', 'p', '*', 'X', 'D']
colors = ['b', 'g', 'k', 'c', 'm', 'y', 'r']
sizes = [140, 140, 140, 140, 140, 140, 150]
for i, dataset in enumerate(datasets):
    ax = axes[i]
    
    # 获取当前数据集的 F1 和 ΔEo 数据
    f1_values_list = [f1_values[method][i] for method in methods]
    dp_values_list = [dp_values[method][i] for method in methods]
    
    # 设置横轴和纵轴的范围
    f1_min = min(f1_values_list) - 1
    f1_max = max(f1_values_list) + 1
    dp_min = min(dp_values_list) - 1
    dp_max = max(dp_values_list) + 1
    
    ax.set_xlim(f1_min, f1_max)
    ax.set_ylim(dp_min, dp_max)
    
    # 绘制散点图
    # 绘制散点图
    for j, method in enumerate(methods):
        sc = ax.scatter(f1_values[method][i], dp_values[method][i], 
                        label=method, marker=markers[j], color=colors[j], s=sizes[j])
        # 仅在第一个子图中添加图例元素
        if i == 0:
            handles.append(sc)
            labels.append(method)

    # 添加子图标题和标签
    ax.set_title(dataset, fontsize=26)
    ax.set_xlabel("ACC(%)", fontsize=28)
    if i == 0:  # 仅在第一个子图上显示 y 轴标签
        ax.set_ylabel("$\\Delta_{DP}$ (↓)", fontsize=28)

        # ax.legend(fontsize=13)
    ax.invert_yaxis()
    ax.grid(True, linestyle='-', color='grey')

# 在整体图上方添加图例
fig.legend(handles=handles[:7], labels=labels[:7], loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=len(methods), fontsize=25)

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.9])  # 为图例留出更多空间   


plt.show()

plt.savefig('/home/yzhen/code/fair/FairSAD_copy/xin-acc_vs_dp_tradeoff1.png', dpi=400)