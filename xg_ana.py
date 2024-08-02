import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# 数据加载
# data = pd.read_csv('/home/yzhen/code/fair/FairSAD_copy/datasets/bail/bail.csv')

# # 查看数据的基本信息
# print(data.head())

# # 敏感属性和下游任务标签的分布
# white_counts = data['WHITE'].value_counts()
# recid_counts = data['RECID'].value_counts()

# print("WHITE counts:\n", white_counts)
# print("RECID counts:\n", recid_counts)

# # 敏感属性与下游任务标签的交叉表
# contingency_table = pd.crosstab(data['WHITE'], data['RECID'])
# print("Contingency Table:\n", contingency_table)

# # 计算卡方检验的p值
# chi2, p, dof, expected = chi2_contingency(contingency_table)
# print(f"Chi-squared Test: chi2 = {chi2}, p = {p}")

# # 如果p值小于0.05，说明敏感属性和下游任务标签之间有显著关联
# if p < 0.05:
#     print("敏感属性WHITE与下游任务RECID之间存在显著关联，可能存在预测捷径。")
# else:
#     print("敏感属性WHITE与下游任务RECID之间不存在显著关联。")

# # 可视化
# plt.figure(figsize=(10, 6))

# # 绘制交叉表的热力图
# sns.heatmap(contingency_table, annot=True, cmap='Blues', fmt='d')
# plt.title('Contingency Table between WHITE and RECID')
# plt.xlabel('RECID')
# plt.ylabel('WHITE')
# plt.show()
# plt.savefig('/home/yzhen/code/fair/FairSAD_copy/xg_ana_bail_reli.png')

# # 条形图显示敏感属性与下游任务标签的关系
# sns.barplot(x='WHITE', y='RECID', data=data,  errorbar=None)
# plt.title('Relationship between WHITE and RECID')
# plt.xlabel('WHITE')
# plt.ylabel('RECID Rate')
# plt.show()
# plt.savefig('/home/yzhen/code/fair/FairSAD_copy/xg_ana_bail_tiaoxing.png')


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# 数据加载
data = pd.read_csv('/home/yzhen/code/fair/FairSAD_copy/datasets/pokec/region_job.csv')

# 查看数据的基本信息
print(data.head())

# 敏感属性和下游任务标签的分布
region_counts = data['region'].value_counts()
I_am_working_in_field_counts = data['I_am_working_in_field'].value_counts()

print("region counts:\n", region_counts)
print("I_am_working_in_field counts:\n", I_am_working_in_field_counts)

# 敏感属性与下游任务标签的交叉表
contingency_table = pd.crosstab(data['region'], data['I_am_working_in_field'])
print("Contingency Table:\n", contingency_table)

# 计算卡方检验的p值
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-squared Test: chi2 = {chi2}, p = {p}")

# 如果p值小于0.05，说明敏感属性和下游任务标签之间有显著关联
if p < 0.05:
    print("敏感属性region与下游任务I_am_working_in_field之间存在显著关联，可能存在预测捷径。")
else:
    print("敏感属性region与下游任务I_am_working_in_field之间不存在显著关联。")

# 可视化
plt.figure(figsize=(10, 6))

# 绘制交叉表的热力图
sns.heatmap(contingency_table, annot=True, cmap='Blues', fmt='d')
plt.title('Contingency Table between region and I_am_working_in_field')
plt.xlabel('I_am_working_in_field')
plt.ylabel('region')
plt.show()
plt.savefig('/home/yzhen/code/fair/FairSAD_copy/xg_ana_pokecz_reli.png')
# 条形图显示敏感属性与下游任务标签的关系
sns.barplot(x='region', y='I_am_working_in_field', data=data, errorbar=None)
plt.title('Relationship between region and I_am_working_in_field')
plt.xlabel('region')
plt.ylabel('I_am_working_in_field Rate')
plt.show()
plt.savefig('/home/yzhen/code/fair/FairSAD_copy/xg_ana_pokecz_tiaoxing.png')
# 绘制敏感属性与下游任务标签的交叉表
sns.countplot(x='region', hue='I_am_working_in_field', data=data)
plt.title('Count of I_am_working_in_field by region')
plt.xlabel('region')
plt.ylabel('Count')
plt.show()

# 按照不同的性别计算GoodCustomer的比例
region_good_customer_rate = data.groupby('region')['I_am_working_in_field'].mean()
print("I_am_working_in_field rate by region:\n", region_good_customer_rate)
