import os
import random
import shutil

# 定义数据集根目录和目标目录
dataset_root = 'pic_data'
train_dir = 'train'
val_dir = 'val'

# 目标数据集目录
new_dataset_root = 'pic_dataset'

# 创建目标数据集目录
os.makedirs(new_dataset_root, exist_ok=True)
os.makedirs(os.path.join(new_dataset_root, train_dir), exist_ok=True)
os.makedirs(os.path.join(new_dataset_root, val_dir), exist_ok=True)


# 定义函数用于将文件按照比例分配到目标目录
def split_dataset(source_dir, train_dest_dir, val_dest_dir, split_ratio=0.8):
    files = os.listdir(source_dir)
    random.shuffle(files)
    train_size = int(len(files) * split_ratio)
    train_files = files[:train_size]
    val_files = files[train_size:]

    # 将文件复制到目标目录
    for file in train_files:
        src_path = os.path.join(source_dir, file)
        dest_path = os.path.join(train_dest_dir, os.path.dirname(file))
        os.makedirs(dest_path, exist_ok=True)
        shutil.copy(src_path, os.path.join(dest_path, file))

    for file in val_files:
        src_path = os.path.join(source_dir, file)
        dest_path = os.path.join(val_dest_dir, os.path.dirname(file))
        os.makedirs(dest_path, exist_ok=True)
        shutil.copy(src_path, os.path.join(dest_path, file))


# 分别处理ai和nature文件夹
ai_source_dir = os.path.join(dataset_root, 'ai')
ai_train_dest_dir = os.path.join(new_dataset_root, train_dir, 'ai')
ai_val_dest_dir = os.path.join(new_dataset_root, val_dir, 'ai')
split_dataset(ai_source_dir, ai_train_dest_dir, ai_val_dest_dir)

nature_source_dir = os.path.join(dataset_root, 'nature')
nature_train_dest_dir = os.path.join(new_dataset_root, train_dir, 'nature')
nature_val_dest_dir = os.path.join(new_dataset_root, val_dir, 'nature')
split_dataset(nature_source_dir, nature_train_dest_dir, nature_val_dest_dir)

print("数据集已经成功分割成训练集和验证集！")
