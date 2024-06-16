import os
from funModule.text_detection import doc_detection, txt_detection, pdf_detection
from funModule.picture_detection import pic_detection

model_path = 'weights/picture/model_ac=0.846.pth'
# 临时保存文件
filename = 'test4.jpg'
upload_file_path = os.path.join('uploads/image', filename)
prediction_result = pic_detection(upload_file_path, model_path)
