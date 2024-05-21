from docx import Document
import paddlehub as hub
import torch
from models.Nets import TextCNN
import numpy as np
import jieba
import json


def read_and_segment_txt(file_path):
    # 打开txt文件
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # 假设每个段落是由空行分隔的，您可以根据实际情况调整这里
    paragraphs = text.split('\n')  # 使用两个换行符来分隔段落

    return paragraphs
