import os
import shutil
import random

# 设置路径
data_dir = '/data3/huhongx/adversarial_competition/attack_dataset/train'  # 你的数据集主文件夹
train_dir = '/data3/huhongx/adversarial_competition/attack_dataset/phase1/train_set'  # 用于保存训练集的文件夹
test_dir = '/data3/huhongx/adversarial_competition/attack_dataset/phase1/test_set'    # 用于保存测试集的文件夹

# 创建训练集和测试集文件夹dc
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 设置训练集和测试集的比例
train_ratio = 0.8

# 遍历每个类别的文件夹
for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    
    # 检查是否是目录
    if os.path.isdir(class_dir):
        # 获取该类别中的所有文件
        files = os.listdir(class_dir)
        random.shuffle(files)
        
        # 按比例分割
        train_size = int(len(files) * train_ratio)
        train_files = files[:train_size]
        test_files = files[train_size:]
        
        # 创建类别子文件夹
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        
        # 复制训练文件
        for file_name in train_files:
            src = os.path.join(class_dir, file_name)
            dest = os.path.join(train_dir, class_name, file_name)
            shutil.copy2(src, dest)
        
        # 复制测试文件
        for file_name in test_files:
            src = os.path.join(class_dir, file_name)
            dest = os.path.join(test_dir, class_name, file_name)
            shutil.copy2(src, dest)

print("数据集已成功按比例复制为训练集和测试集。")
