import matplotlib.pyplot as plt
import numpy as np

# 数据集名称
datasets = ['Bail', 'Pokec_z', 'Pokec_n']

# 不同FairOOD指标
metrics = ['FairOOD\\FD', 'FairOOD\\FEnv', 'FairOOD']

# 对应的AUC均值和标准差数据
dp_means = [
    [2.02, 2.19, 0.68],
    [2.99 ,2.51 ,1.14 ],
    [5.07 ,2.02 ,1.04 ]
]

dp_stds = [
    [0.14	,0.16,0.12],
    [1.29, 0.16, 0.35],
    [0.46, 1.06, 0.46]
]

# 柱形图颜色设置
colors = ['#87CEEB', '#2E8B57', '#FF8C00']

# 绘制柱形图
bar_width = 0.2
index = np.arange(len(datasets)) * 0.9  # 调整每组柱状图的位置  # 调整每组柱状图的位置

fig, ax = plt.subplots(figsize=(6, 4))

for i in range(len(metrics)):
    bars = ax.bar(index + i * bar_width, [auc[i] for auc in dp_means], bar_width, 
                  yerr=[std[i] for std in dp_stds], capsize=5, label=metrics[i], color=colors[i])

# 添加误差线标签
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.05, round(yval, 2), ha='center', va='bottom')

# ax.set_xlabel('Datasets')
ax.set_ylabel('DP(%)')
# ax.set_title('AUC Scores by Dataset and FairOOD Metrics')
ax.set_xticks(index + bar_width * 0.8)
ax.set_xticklabels(datasets)
ax.legend()

# 设置y轴范围
ax.set_ylim(0, 6)

plt.tight_layout()

# 保存图片
plt.savefig('/home/yzhen/code/fair/FairSAD_copy/plot_xaiorong_dp.png')

plt.show()

