import easyocr
import json


def ocr_detection(file_path, model_path):
    # 创建OCR对象并加载模型
    reader = easyocr.Reader(['ch_sim', 'en'], model_storage_directory=model_path)

    # 读取图像并提取文字
    image_path = file_path  # 替换为自己的图像路径
    result = reader.readtext(image_path)
    texts = []

    # 输出提取的文字
    for detection in result:
        texts.append(detection[1])
    res = texts

    # 将列表转换为JSON字符串
    res_json = json.dumps(res, ensure_ascii=False, indent=4)
    return res_json
