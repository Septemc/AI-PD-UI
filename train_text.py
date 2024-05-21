import paddlehub as hub
import os
import numpy as np
import jieba
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from models.Nets import TextCNN

# 加载词向量模型
embedding = hub.Module(name='w2v_wiki_target_word-word_dim300')

# 加载计算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'当前使用设备:{device}')

# 加载数据集
dataset_path = 'datasets/TextDataset'  # 数据集路径
labels = []
file_list = []
vocab_size = 800
embedding_dim = 300
num_filters = 128
output_dim = 2
# 设置数据形状
target_shape = (vocab_size, 300)

# 读取数据
for root, dirs, files in os.walk(dataset_path):
    for dir_name in dirs:
        label = torch.tensor(int(dir_name))  # 将目录名转换为标签
        dir_path = os.path.join(root, dir_name)
        for file in os.listdir(dir_path):
            file_list.append(os.path.join(dir_path, file))
            labels.append(label)

text_labels = torch.stack(labels)
data_text_tensor = []  # 用于存放词向量化后的数据

for file_path in file_list:
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

        # 使用jieba进行分词
        words = jieba.cut(text)
        words_list = list(words)

        # 将分词后的词转化为向量
        text_vector = embedding.search(words_list)

        # 转换为numpy数组
        text_vector_np = np.array(text_vector)
        # 计算需要填充的行数
        pad_rows = target_shape[0] - text_vector_np.shape[0]
        # 创建用于填充的0矩阵
        padding = np.zeros((pad_rows, 300))
        # 填充张量数据
        text_padded_data = np.vstack([text_vector_np, padding])
        # 转换为tensor
        text_vector_tensor = torch.tensor(text_padded_data).float()
        data_text_tensor.append(text_vector_tensor)

text_datas = torch.stack(data_text_tensor)

text_train, text_test, label_train, label_test = train_test_split(text_datas, text_labels, test_size=0.2, random_state=42)


# 定义 DataLoader
train_dataset = TensorDataset(text_train.to(device), label_train.to(device))
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=False)

test_dataset = TensorDataset(text_test.to(device), label_test.to(device))
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, pin_memory=False)


# 初始化模型
model = TextCNN(embedding_dim, num_filters, output_dim)

# 加载当前模型权重
model_path = 'weights/text/model_1.pth'
model.load_state_dict(torch.load(model_path))

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 设备
model.to(device)

# 模型训练
num_epochs = 5
model.train()
for epoch in range(num_epochs):
    total_losses = 0
    correct_num = 0
    for i, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data.to(device)
        target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_losses += loss.item()

        # 计算准确率
        predictions = output.argmax(dim=1)
        correct_predictions = (predictions == target).float().sum().item()
        correct_num += correct_predictions

    accuracy = correct_num/len(train_loader.dataset)
    print(f'Epoch {epoch + 1} Average Loss: {total_losses} Accuracy: {accuracy}')

    # 在每个epoch结束后评估模型在测试集上的准确率
    model.eval()
    correct_num = 0
    with torch.no_grad():
        for data, target in test_loader:
            data.to(device)
            target.to(device)
            output = model(data)
            predictions = output.argmax(dim=1)
            correct_predictions = (predictions == target).float().sum().item()
            correct_num += correct_predictions

        test_accuracy = correct_num/len(test_loader.dataset)
        print(f'Test Accuracy: {test_accuracy}')

# 保存模型文件
torch.save(model.state_dict(), 'logs/weights/text/model_temp.pth')
print('Training completed!')



