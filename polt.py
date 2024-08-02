import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
font1 = {
'weight' : 'normal',
'size'   : 23,}
font_list = [x.name for x in font_manager.fontManager.ttflist]
print(font_list)

# Data
'''绘制bail相同sample下随per变化的折线图'''
# import numpy as np
# import matplotlib.pyplot as plt

# # Data

perturb_epsilon = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2]

aucroc = [73.02, 72.79, 72.34, 72.47, 72.42, 72.72, 72.45, 72.49]
aucroc_std = [0.19, 0.19, 0.12, 0.22, 0.27, 0.16, 0.26, 0.29]

f1 = [62.32, 61.62, 62.49, 62.35, 61.87, 62.51, 62.23, 61.92]
f1_std = [0.61, 0.61, 0.71, 0.66, 0.56, 0.56, 0.53, 0.63]

acc = [67.2, 67.2, 67.06, 67.04, 67.01, 67.45, 67.03, 67.17]
acc_std = [0.29, 0.29, 0.22, 0.27, 0.14, 0.16, 0.16, 0.22]

parity = [1.99, 1.8, 1.36, 1.52, 1.2, 1.23, 1.34, 1.16]
parity_std = [0.14, 0.14, 0.39, 0.7, 0.15, 0.31, 0.2, 0.55]

equality = [1.54, 1.24, 0.62, 0.68, 0.52, 0.54, 0.4, 0.87]
equality_std = [0.37, 0.37, 0.58, 0.53, 0.35, 0.29, 0.24, 0.3]


# # Convert data to numpy arrays for easier handling
perturb_epsilon = np.array(perturb_epsilon)
aucroc = np.array(aucroc)
f1 = np.array(f1)
acc = np.array(acc)
parity = np.array(parity)
equality = np.array(equality)

# Increase figure size
fig, ax1 = plt.subplots(figsize=(8 ,6))

color1 = '#FF69B4'  # Deep Pink
color2 = '#6495ED'  # Sky Blue

# Plotting accuracy metrics
ax1.set_xlabel('Control coefficient $\\alpha $',fontsize=25,fontdict=font1)
ax1.set_ylabel('AUC/F1/ACC (%)',fontsize=20,fontdict=font1)
ax1.plot(perturb_epsilon, aucroc, marker='o', label='AUC', color=color1, linestyle='-',linewidth=2)
ax1.plot(perturb_epsilon, f1, marker='x', label='F1', color=color1, linestyle='--',linewidth=2)
ax1.plot(perturb_epsilon, acc, marker='s', label='ACC', color=color1, linestyle='-.',linewidth=2)
ax1.tick_params(axis='y',labelsize = 16)
ax1.tick_params(axis='x', labelsize=16)
# Creating a twin axis to plot fairness metrics
ax2 = ax1.twinx()
ax2.set_ylabel('$\\ \Delta_{DP}/ \Delta_{EO}$', fontsize=20,fontdict=font1)
ax2.plot(perturb_epsilon, parity, marker='o', label='DP (Parity)', color=color2, linestyle='-',linewidth=2)
ax2.plot(perturb_epsilon, equality, marker='x', label='EO (Equality)', color=color2, linestyle='--',linewidth=2)
ax2.tick_params(axis='y',labelsize = 16)

# Adding separate legends for each axis
# For ax1
lines1, labels1 = ax1.get_legend_handles_labels()
ax1.legend(lines1, labels1, loc='upper left', fontsize='large')

# For ax2
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines2, labels2, loc='upper right', fontsize='large')

# Adjust layout to make room for the legends
fig.tight_layout()
plt.savefig('/home/yzhen/code/fair/FairSAD_copy/plot_pokecn_K5.png',dpi=400)
plt.show()


perturb_epsilon = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]
aucroc = [88.57, 88.47, 88.47, 88.54, 88.47, 88.48, 88.48, 88.48, 88.51, 88.49]
f1 = [77.76, 77.69, 78.08, 77.87, 77.92, 77.77, 77.43, 77.72, 77.76, 77.6]
acc = [83.43, 83.53, 83.92, 83.6, 83.69, 83.51, 83.16, 83.43, 83.53, 83.37]
parity = [0.41, 0.25, 0.74, 0.65, 0.33, 0.44, 0.2, 0.44, 0.24, 0.3]
equality = [0.72, 0.57, 0.68, 0.86, 0.72, 0.91, 0.67, 0.73, 1.01, 0.91]
perturb_epsilon = np.array(perturb_epsilon)
aucroc = np.array(aucroc)
f1 = np.array(f1)
acc = np.array(acc)
parity = np.array(parity)
equality = np.array(equality)

# Increase figure size
fig, ax1 = plt.subplots(figsize=(8 ,6))

color1 = '#FF69B4'  # Deep Pink
color2 = '#6495ED'  # Sky Blue

# Plotting accuracy metrics
ax1.set_xlabel('Control coefficient $\\alpha $',fontsize=25,fontdict=font1)
ax1.set_ylabel('AUC/F1/ACC (%)',fontsize=20,fontdict=font1)
ax1.plot(perturb_epsilon, aucroc, marker='o', label='AUC', color=color1, linestyle='-',linewidth=2)
ax1.plot(perturb_epsilon, f1, marker='x', label='F1', color=color1, linestyle='--',linewidth=2)
ax1.plot(perturb_epsilon, acc, marker='s', label='ACC', color=color1, linestyle='-.',linewidth=2)
ax1.tick_params(axis='y',labelsize = 16)
ax1.tick_params(axis='x', labelsize=16)
# Creating a twin axis to plot fairness metrics
ax2 = ax1.twinx()
ax2.set_ylabel('$\\ \Delta_{DP}/ \Delta_{EO}$', fontsize=20,fontdict=font1)
ax2.plot(perturb_epsilon, parity, marker='o', label='DP (Parity)', color=color2, linestyle='-',linewidth=2)
ax2.plot(perturb_epsilon, equality, marker='x', label='EO (Equality)', color=color2, linestyle='--',linewidth=2)
ax2.tick_params(axis='y',labelsize = 16)

# Adding separate legends for each axis
# For ax1
lines1, labels1 = ax1.get_legend_handles_labels()
ax1.legend(lines1, labels1, loc='upper left', fontsize='large')

# For ax2
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines2, labels2, loc='upper right', fontsize='large')

# Adjust layout to make room for the legends
fig.tight_layout()
plt.savefig('/home/yzhen/code/fair/FairSAD_copy/plot_bail_K5.png',dpi=400)
plt.show()


import matplotlib.pyplot as plt
'''绘制bail相同per下随sample变化的折线图'''

# samples = [2,  5, 6, 10,12,15]
# aucroc = [72.53, 88.57, 88.46, 88.59, 88.35]
# f1_score = [78.0, 77.76, 77.28, 78.24, 77.99]
# acc = [83.77, 83.43, 82.92, 83.98, 83.77]
# parity = [0.63, 0.41, 0.44, 0.96, 1.22]
# equality = [1.33, 0.72, 0.69, 0.76, 0.94]

# # 创建画布和双 y 轴
# fig, ax1 = plt.subplots(figsize=(8, 6))
# color1 = '#FF1493'  # Deep Pink
# color2 = '#6495ED'  # Sky Blue
# # 左 y 轴 - 准确性指标
# ax1.plot(samples, aucroc, marker='^', linestyle='-', color=color1, label='AUCROC')
# ax1.plot(samples, f1_score, marker='v', linestyle='--', color=color1, label='F1-score')
# ax1.plot(samples, acc, marker='D', linestyle='-.', color=color1, label='ACC')
# ax1.set_xlabel('Number K of Environments')
# ax1.set_ylabel('Accuracy Metrics (AUCROC/F1-score/ACC)', color=color1)
# ax1.tick_params(axis='y', labelcolor=color1)
# ax1.legend(loc='upper left')

# # 右 y 轴 - 公平性指标
# ax2 = ax1.twinx()
# ax2.plot(samples, parity, marker='o', linestyle='-', color=color2, label='DP (Parity)')
# ax2.plot(samples, equality, marker='s', linestyle='--', color=color2, label='EO (Equality)')
# ax2.set_ylabel('Fairness Metrics (DP/EO)', color=color2)
# ax2.tick_params(axis='y', labelcolor=color2)
# ax2.legend(loc='upper right')

# # 调整横轴步数
# ax1.set_xticks(range(min(samples), max(samples) + 1))

# plt.savefig('/home/yzhen/code/fair/FairSAD_copy/plot_bail_0.1.png')
# plt.show()
# 新数据
'''pokec_n相同sample下随per变化的折线图'''
# samples = [1, 2, 3, 5, 6, 8, 10, 15]

# # AUCROC values
# aucroc = [88.48, 88.66, 88.52, 88.57, 88.46, 88.46, 88.59, 88.35]


# # F1-score values
# f1_score = [77.71, 78.0, 78.25, 77.76, 77.28, 78.11, 78.24, 77.99]


# # ACC values
# acc = [83.56, 83.77, 84.09, 83.43, 82.92, 83.78, 83.98, 83.77]


# # Parity values
# parity = [0.58, 0.63, 1.07, 0.41, 0.44, 0.87, 0.96, 1.22]

# # Equality values
# equality = [1.57, 1.33, 0.8, 0.72, 0.69, 0.86, 0.76, 0.94]

equality = [1.57, 1.33, 0.8, 0.72, 0.69, 0.86, 0.76, 0.94]
samples = [1, 3, 5, 7, 10, 12, 15]
aucroc = [72.5, 72.44, 72.72, 72.46, 71.7, 72.37, 71.25]
f1_score = [62.47, 61.92, 62.51, 62.03, 60.42, 59.39, 57.81]
acc = [67.14, 66.93, 67.45, 67.44, 66.77, 67.16, 67.42]
parity = [2.2, 1.5, 1.23, 1.58, 2.13, 2.4, 3.59]
equality = [1.43, 1.16, 0.54, 1.19, 2.17, 2.4, 3.62]

samples = np.array(samples)
aucroc = np.array(aucroc)
f1_score = np.array(f1_score)
acc = np.array(acc)
parity = np.array(parity)
equality = np.array(equality)
# 创建画布和双 y 轴
fig, ax1 = plt.subplots(figsize=(8 ,6))
color1 = '#FF69B4'  # Deep Pink
color2 = '#6495ED'  # Sky Blue
# 左 y 轴 - 准确性指标
ax1.plot(samples, aucroc, marker='^', linestyle='-', color=color1, label='AUC',linewidth=2)
ax1.plot(samples, f1_score, marker='v', linestyle='--', color=color1, label='F1',linewidth=2)
ax1.plot(samples, acc, marker='D', linestyle='-.', color=color1, label='ACC',linewidth=2)
ax1.set_xlabel('Number K of Environments',fontsize=25,fontdict=font1)
ax1.set_ylabel('AUC/F1/ACC (%)',fontsize=20,fontdict=font1)
ax1.tick_params(axis='y', labelsize = 16)
ax1.legend(loc='upper left')
ax1.tick_params(axis='x', labelsize=16)
# 右 y 轴 - 公平性指标
ax2 = ax1.twinx()
ax2.plot(samples, parity, marker='o', linestyle='-', color=color2, label='DP (Parity)',linewidth=2)
ax2.plot(samples, equality, marker='s', linestyle='--', color=color2, label='EO (Equality)',linewidth=2)
ax2.set_ylabel('$\\ \Delta_{DP}/ \Delta_{EO}$ ',fontsize=20,fontdict=font1)
ax2.tick_params(axis='y', labelsize = 16)
ax2.legend(loc='upper right')

# 调整横轴步数
ax1.set_xticks(range(min(samples), max(samples) + 1))


plt.savefig('/home/yzhen/code/fair/FairSAD_copy/plot_POKECN_0.1_t.png',bbox_inches='tight',dpi=400)



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
samples = np.array(samples)
aucroc = np.array(aucroc)
f1_score = np.array(f1_score)
acc = np.array(acc)
parity = np.array(parity)
equality = np.array(equality)
# 创建画布和双 y 轴
fig, ax1 = plt.subplots(figsize=(8 ,6))
color1 = '#FF69B4'  # Deep Pink
color2 = '#6495ED'  # Sky Blue
# 左 y 轴 - 准确性指标
ax1.plot(samples, aucroc, marker='^', linestyle='-', color=color1, label='AUC',linewidth=2)
ax1.plot(samples, f1_score, marker='v', linestyle='--', color=color1, label='F1',linewidth=2)
ax1.plot(samples, acc, marker='D', linestyle='-.', color=color1, label='ACC',linewidth=2)
ax1.set_xlabel('Number K of Environments',fontsize=25,fontdict=font1)
ax1.set_ylabel('AUC/F1/ACC (%)',fontsize=20,fontdict=font1)
ax1.tick_params(axis='y', labelsize = 16)
ax1.legend(loc='upper left')
ax1.tick_params(axis='x', labelsize=16)
# 右 y 轴 - 公平性指标
ax2 = ax1.twinx()
ax2.plot(samples, parity, marker='o', linestyle='-', color=color2, label='DP (Parity)',linewidth=2)
ax2.plot(samples, equality, marker='s', linestyle='--', color=color2, label='EO (Equality)',linewidth=2)
ax2.set_ylabel('$\\ \Delta_{DP}/ \Delta_{EO}$ ',fontsize=20,fontdict=font1)
ax2.tick_params(axis='y', labelsize = 16)
ax2.legend(loc='upper right')

# 调整横轴步数
ax1.set_xticks(range(min(samples), max(samples) + 1))


plt.savefig('/home/yzhen/code/fair/FairSAD_copy/plot_bail_0.1_t.png',bbox_inches='tight',dpi=400)
