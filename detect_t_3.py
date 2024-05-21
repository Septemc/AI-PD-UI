# -*- coding: utf-8 -*-
from docx import Document
import paddlehub as hub
import os
import numpy as np
import jieba
import torch
from models.Nets import TextCNN


vocab_size = 800
embedding_dim = 300
num_filters = 128
output_dim = 2
# 设置数据形状
target_shape = (vocab_size, 300)

# 加载模型文件
model = TextCNN(embedding_dim, num_filters, output_dim)
model.load_state_dict(torch.load('logs/weights/model_1.pth'))
# 读取Word文档
doc = Document('testdata/吕鸿成202231090197.docx')

# 初始化文本和结果列表
text_chunks = []
avg_ai_prob = 0
avg_real_prob = 0
word_num = 0
# 将文本按照每100字分割，并进行预测
for paragraph in doc.paragraphs:
    text = paragraph.text
    if text:
        text_chunks.append(text)
        word_num += len(text)

# 加载词向量模型
embedding = hub.Module(name='w2v_wiki_target_word-word_dim300')
data_text_tensor = []

for text in text_chunks:
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
text_shape = text_datas.shape

outputs = model(text_datas)
predictions = outputs.argmax(dim=1)
total_weighted_ai_prob = 0
total_weighted_real_prob = 0

for i, text in enumerate(text_chunks):
    text_length = len(text)
    weighted_ai_prob = outputs[i][0].item() * 100 * (text_length / word_num)
    weighted_real_prob = outputs[i][1].item() * 100 * (text_length / word_num)
    total_weighted_ai_prob += weighted_ai_prob
    total_weighted_real_prob += weighted_real_prob
    print(f'文本{i+1}:')
    print(text)
    print(f'AI:{outputs[i][0].item() * 100:.3f}%')
    print(f'真实:{outputs[i][1].item() * 100:.3f}%')

print("整体加权检测结果:")
print(f'AI:{total_weighted_ai_prob:.3f}%')
print(f'真实:{total_weighted_real_prob:.3f}%')
