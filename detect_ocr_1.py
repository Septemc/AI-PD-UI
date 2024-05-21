import easyocr
import json

# 创建OCR对象并加载模型
reader = easyocr.Reader(['ch_sim', 'en'], model_storage_directory='weights/ocr', gpu=False)

# 读取图像并提取文字
image_path = 'uploads/ocr/test_ocr.png'  # 替换为自己的图像路径
result = reader.readtext(image_path)

texts = []
# 输出提取的文字
for detection in result:
    texts.append(detection[1])
res = texts

# 将列表转换为JSON字符串
res_json = json.dumps(res, ensure_ascii=False, indent=4)
print(res_json)
