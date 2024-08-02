import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.rcParams.update({'font.size': 20, 'font.family': 'serif'})

# def plot_embeddings_from_npy(embedding_path, labels_path, sens_path, title='Node Embeddings', output_filename='plot_from_npy.png'):
#     # 读取保存的节点表示、标签和敏感属性
#     embeddings_2d = np.load(embedding_path)
#     labels = np.load(labels_path)
#     sens = np.load(sens_path)

#     # 创建 DataFrame 用于绘图
#     df = pd.DataFrame({
#         'X': embeddings_2d[:, 0],
#         'Y': embeddings_2d[:, 1],
#         'Label': labels,
#         'Sens': sens
#     })

#     # 定义形状映射
#     shapes = {0: 'o', 1: 's'}  # 假设预测值 y 只有 0 和 1 两类，分别用圆形和方形表示

#     colors = {0: 'pink', 1: 'lightskyblue'}
#     plt.figure(figsize=(10, 8))

#     # 遍历每个标签和敏感属性，绘制散点图
#     for label in df['Label'].unique():
#         for sens in df['Sens'].unique():
#             subset = df[(df['Label'] == label) & (df['Sens'] == sens)]
#             plt.scatter(subset['X'], subset['Y'], label=f'Label {label}, Sens {sens}',
#                         c=[colors[sens]], marker=shapes[label], edgecolor='k', s=100)

#     plt.title(title)
#     plt.xlabel('Component 1')
#     plt.ylabel('Component 2')

#     # 将图例放置在整个图片的上方
   

#     # 保存绘图
#     plt.savefig(output_filename)

# # 示例调用方法
# # 读取并绘制从npy文件中读取的数据
# plot_embeddings_from_npy(
#     embedding_path='/home/yzhen/code/fair/FairVGNN/embeddings_bail.npy',
#     labels_path='/home/yzhen/code/fair/FairVGNN/labels_bail.npy',
#     sens_path='/home/yzhen/code/fair/FairVGNN/sens_bail.npy',
#     title='Node Embeddings',
#     output_filename='/home/yzhen/code/fair/FairVGNN/plot_from_npy.png'
# )
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_embeddings_from_npy(ax, embedding_path, labels_path, sens_path, x_label='FairGNN'):
    # 读取保存的节点表示、标签和敏感属性
    embeddings_2d = np.load(embedding_path)
    labels = np.load(labels_path)
    sens = np.load(sens_path)

    # 创建 DataFrame 用于绘图
    df = pd.DataFrame({
        'X': embeddings_2d[:, 0],
        'Y': embeddings_2d[:, 1],
        'Label': labels,
        'Sens': sens
    })

    # 定义形状映射
    shapes = {0: 'o', 1: 's'}  # 假设预测值 y 只有 0 和 1 两类，分别用圆形和方形表示

    colors = {0: 'lightskyblue', 1: 'pink'}
    
    # 遍历每个标签和敏感属性，绘制散点图
    for label in df['Label'].unique():
        for sens in df['Sens'].unique():
            subset = df[(df['Label'] == label) & (df['Sens'] == sens)]
            ax.scatter(subset['X'], subset['Y'], label=f'Label {label}, Sens {sens}',
                       c=[colors[sens]], marker=shapes[label], edgecolor='k', s=100)

    
    ax.set_xlabel(x_label, fontsize=28,labelpad=10)
    # ax.set_ylabel('Component 2',fontsize=20)
    # ax.set_title(title, fontsize=25, pad=30,loc='center', y=-0.28)

# 创建一个包含三个子图的图形窗口
fig, axs = plt.subplots(1, 3, figsize=(30, 8))

# 定义每个路径下的 .npy 文件路径
paths = [
    ('/home/yzhen/code/fair/FairVGNN/embeddings_bail.npy', '/home/yzhen/code/fair/FairVGNN/labels_bail.npy', '/home/yzhen/code/fair/FairVGNN/sens_bail.npy','(1) FairGNN' ),
    ('/home/yzhen/code/fair/FairSAD/embeddings_bail.npy', '/home/yzhen/code/fair/FairSAD/labels_bail.npy', '/home/yzhen/code/fair/FairSAD/sens_bail.npy','(2) FairSAD' ),
    ('/home/yzhen/code/fair/FairSAD_copy/embeddings_bail.npy', '/home/yzhen/code/fair/FairSAD_copy/labels_bail.npy', '/home/yzhen/code/fair/FairSAD_copy/sens_bail.npy','(3) Ours')
]

# 绘制每个子图
for i, (embedding_path, labels_path, sens_path,x_label ) in enumerate(paths):
    plot_embeddings_from_npy(axs[i], embedding_path, labels_path, sens_path,x_label)

# 在整体图上添加图例
handles, labels = axs[0].get_legend_handles_labels()

fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=28, bbox_to_anchor=(0.5, 1.),markerscale=2)

# 调整子图之间的间

# 调整子图之间的间距，并向下移动子图
plt.subplots_adjust(wspace=0.2, top=0.87)


# 保存最终拼接好的大图
plt.savefig('/home/yzhen/code/fair/FairSAD_copy/combined_plot_keshihua.png', dpi=600)
plt.show()

 
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# # 加载三张 PNG 图像
# img1 = mpimg.imread('/home/yzhen/code/fair/FairVGNN/plot_from_npy.png')
# img2 = mpimg.imread('/home/yzhen/code/fair/FairSAD/plot_from_npy.png')
# img3 = mpimg.imread('/home/yzhen/code/fair/FairSAD_copy/plot_from_npy.png')

# # 创建一个包含三行的图形窗口
# fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# # 将每张图像放置在相应的子图中
# axs[0].imshow(img1)
# axs[0].axis('off')  # 关闭坐标轴显示
# axs[0].set_title("Plot 1")

# axs[1].imshow(img2)
# axs[1].axis('off')  # 关闭坐标轴显示
# axs[1].set_title("Plot 2")

# axs[2].imshow(img3)
# axs[2].axis('off')  # 关闭坐标轴显示
# axs[2].set_title("Plot 3")

# # 调整子图之间的距离
# fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.05, hspace=0.4)

# # 添加整体图例
# handles, labels = axs[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', ncol=3)

# # 保存最终拼接好的大图
# plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局，避免图例重叠
# plt.savefig('/home/yzhen/code/fair/FairSAD_copy/combined_plot_keshihua.png', dpi=400)
# plt.show()
