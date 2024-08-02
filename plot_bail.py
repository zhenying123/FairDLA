import numpy as np
import matplotlib.pyplot as plt

# 假设font1已定义
font1 = {'family': 'serif',  'size': 25}
samples = [1, 2, 3, 5, 6, 8, 10, 15]

# AUCROC values
aucroc = [88.48, 88.66, 88.52, 88.57, 88.46, 88.46, 88.59, 88.35]


# F1-score values
f1_score = [77.71, 78.0, 78.25, 77.76, 77.28, 78.11, 78.24, 77.99]


# ACC values
acc = [83.56, 83.77, 84.09, 83.43, 82.92, 83.78, 83.98, 83.77]


# Parity values
parity = [0.58, 0.63, 1.07, 0.41, 0.44, 0.87, 0.96, 1.22]

# Equality values
equality = [1.57, 1.33, 0.8, 0.72, 0.69, 0.86, 0.76, 0.94]

# 将数据转换为numpy数组
samples = np.array(samples)
aucroc = np.array(aucroc)
f1_score = np.array(f1_score)
acc = np.array(acc)
parity = np.array(parity)
equality = np.array(equality)

# 创建包含两个子图的画布
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(18.5, 7))

color1 = '#c82423'  # Deep Pink
color2 = '#2878b5'  # Sky Blue # Sky Blue

# 左 y 轴 - 准确性指标
ax1.plot(samples, aucroc, marker='o', linestyle='-', color=color1, label='AUC', linewidth=2.5,markersize=10)
ax1.plot(samples, f1_score, marker='x', linestyle='--', color=color1, label='F1', linewidth=2.5,markersize=10)
ax1.plot(samples, acc, marker='s', linestyle='-.', color=color1, label='ACC', linewidth=2.5,markersize=10)
ax1.set_xlabel('Number K of Environments', fontsize=30, fontdict=font1)
ax1.set_ylabel('AUC/ F1/ ACC (%)', fontsize=25, fontdict=font1)
ax1.tick_params(axis='y', labelsize=16)
ax1.tick_params(axis='x', labelsize=16)

# 右 y 轴 - 公平性指标
ax2 = ax1.twinx()
ax2.plot(samples, parity, marker='o', linestyle='-', color=color2, label='$\\Delta_{DP}$', linewidth=2.5,markersize=10)
ax2.plot(samples, equality, marker='x', linestyle='--', color=color2, label='$\\Delta_{EO}$', linewidth=2.5,markersize=10)
ax2.set_ylabel('$\\Delta_{DP}/\\Delta_{EO}$', fontsize=25, fontdict=font1)
ax2.tick_params(axis='y', labelsize=16)

# 调整横轴步数
ax1.set_xticks(range(min(samples), max(samples) + 1))

perturb_epsilon = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]
aucroc = [88.57, 88.47, 88.47, 88.54, 88.47, 88.48, 88.48, 88.48, 88.51, 88.49]
f1 = [77.76, 77.69, 78.08, 77.87, 77.92, 77.77, 77.43, 77.72, 77.76, 77.6]
acc = [83.43, 83.53, 83.92, 83.6, 83.69, 83.51, 83.16, 83.43, 83.53, 83.37]
parity = [0.41, 0.25, 0.74, 0.65, 0.33, 0.44, 0.2, 0.44, 0.24, 0.3]
equality = [0.72, 0.57, 0.68, 0.86, 0.72, 0.91, 0.67, 0.73, 1.01, 0.91]

# 设置perturb_epsilon相关数据
perturb_epsilon = np.array(perturb_epsilon)
aucroc = np.array(aucroc)
f1 = np.array(f1)
acc = np.array(acc)
parity = np.array(parity)
equality = np.array(equality)

# 增加图大小
color1 = '#c82423'  # Deep Pink
color2 = '#2878b5'  # Sky Blue

# 绘制准确性指标
ax3.set_xlabel('Control coefficient $\\alpha$', fontsize=30, fontdict=font1)
ax3.set_ylabel('AUC/ F1/ ACC (%)', fontsize=25, fontdict=font1)
ax3.plot(perturb_epsilon, aucroc, marker='o', label='AUC', color=color1, linestyle='-', linewidth=2.5,markersize=10)
ax3.plot(perturb_epsilon, f1, marker='x', label='F1', color=color1, linestyle='--', linewidth=2.5,markersize=10)
ax3.plot(perturb_epsilon, acc, marker='s', label='ACC', color=color1, linestyle='-.', linewidth=2.5,markersize=10)
ax3.tick_params(axis='y', labelsize=16)
ax3.tick_params(axis='x', labelsize=16)

# 创建一个双轴来绘制公平性指标
ax4 = ax3.twinx()
ax4.set_ylabel('$\\Delta_{DP}/\\Delta_{EO}$', fontsize=25, fontdict=font1)
ax4.plot(perturb_epsilon, parity, marker='o', label='$\\ \Delta_{DP}', color=color2, linestyle='-', linewidth=2.5,markersize=10)
ax4.plot(perturb_epsilon, equality, marker='x', label='$\\  \Delta_{EO}$', color=color2, linestyle='--', linewidth=2.5,markersize=10)
ax4.tick_params(axis='y', labelsize=16)

# 添加整体图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()


fig.legend(lines1 + lines2 , labels1 + labels2 , loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=5, fontsize=28)

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.88])
plt.subplots_adjust(wspace=0.35)
plt.savefig('/home/yzhen/code/fair/FairSAD_copy/combined_plot_bail.png', dpi=400)
plt.show()
