import scipy.io as sio

# 读取 .mat 文件
data = sio.loadmat('/mnt/sdb1/leijh/EnergySnake1/18_0crf-image-labeling-master-vertebrae/original_version/A46_simp1_results.mat')

# data 现在是一个字典，键是变量名，值是对应的数组
print(data.keys()) # 查看文件中的所有变量名
variable_data = data['Wt'] # 提取特定变量
print(variable_data)

# 写入 .mat 文件
# sio.savemat('new_file.mat', {'new_var': variable_data})