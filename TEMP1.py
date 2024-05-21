import os
from flask import Flask, request, render_template
from funModule.text_detection import doc_detection, txt_detection, pdf_detection
from docx import Document
import chardet

model_path = 'logs/weights/model_1.pth'

# 临时保存文件
upload_file_path = 'uploads/text/新建 文本文档.txt'
filename = os.path.basename(upload_file_path)  # 获取文件名

if '.txt' in filename:
    prediction_result = txt_detection(upload_file_path, model_path)
elif '.doc' in filename:
    prediction_result = doc_detection(upload_file_path, model_path)
elif '.pdf' in filename:
    prediction_result = pdf_detection(upload_file_path, model_path)
else:
    print("Error: 仅支持.txt、.doc和.pdf格式的文件。请上传正确格式的文件。")

print('OK')
