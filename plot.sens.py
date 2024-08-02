import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 数据
perturb_epsilon = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2])
aucroc = np.array([73.02, 72.79, 72.34, 72.47, 72.42, 72.72, 72.45, 72.49])
aucroc_std = np.array([0.19, 0.19, 0.12, 0.22, 0.27, 0.16, 0.26, 0.29])
parity = np.array([1.99, 1.8, 1.36, 1.52, 1.2, 1.23, 1.34, 1.16])
parity_std = np.array([0.14, 0.14, 0.39, 0.7, 0.15, 0.31, 0.2, 0.55])

# 字体属性
font1 = {'weight': 'normal', 'size': 23}

# 创建图形和 3D 轴
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 绘制 3D 图
ax.plot(perturb_epsilon, aucroc, parity, marker='o', label='AUC vs DP', color='#FF69B4', linestyle='-', linewidth=2)

# 设置坐标轴标签
ax.set_xlabel('Control coefficient $\\alpha$', fontsize=25, fontdict=font1)
ax.set_ylabel('AUCROC (%)', fontsize=20, fontdict=font1)
ax.set_zlabel('DP (Parity)', fontsize=20, fontdict=font1)

# 设置刻度标签大小
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.tick_params(axis='z', labelsize=16)

# 添加图例
ax.legend(loc='upper left', fontsize='large')

# 显示图形
plt.tight_layout()
plt.savefig('/home/yzhen/code/fair/FairSAD_copy/plot_pokecn_3d.png', dpi=400)
plt.show()
