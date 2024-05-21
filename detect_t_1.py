# -*- coding: utf-8 -*-
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

input_text = ['探索未知，挑战自我，我们的旅程即将启程。在这片神秘的土地上，你将遇见无数奇妙的风景，体验前所未有的冒险。无论是翻越高山，还是穿越沙漠，都将是一次难忘的旅程。带上你的勇气和决心，让我们一起踏上这段冒险之旅，寻找属于你的传奇故事！',
              '我觉得wsy有点唐',
              '我们趋行在人生这个亘古的旅途，在坎坷中奔跑，在挫折里涅槃，忧愁缠满全身，痛苦飘洒一地。我们累，却无从止歇；我们苦，却无法回避',
              '你觉得西红柿抄番茄好吃吗？',
              '在过去的工作经历中，我曾参与开发多个AI项目，包括自然语言处理模型和机器学习算法的实现。我喜欢挑战自己，不断探索新的知识领域，并乐于与团队协作共同实现目标。除了技术方面，我还注重人际交往，善于倾听他人意见并与人沟通合作。我具备良好的团队合作精神和解决问题的能力，能够在压力下保持冷静和高效。',
              '大家好，我的名字是陶炳雨，现在正在学习自然语言处理，喜欢研究和大家一起探讨人工智能的知识，很高兴认识大家。']

# 加载词向量模型
embedding = hub.Module(name='w2v_wiki_target_word-word_dim300')
data_text_tensor = []

for text in input_text:
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
for i, output in enumerate(outputs):
    res1 = output[0].item() * 100
    res2 = output[1].item() * 100
    print(f'文本{i+1}:')
    print(input_text[i])
    print(f'AI:{res1:.3f}%')
    print(f'真实:{res2:.3f}%')
