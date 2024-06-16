import easyocr
import json


def ocr_detection(file_path, model_path):
    # 创建OCR对象并加载模型
    reader = easyocr.Reader(['ch_sim', 'en'], model_storage_directory=model_path)

    # 读取图像并提取文字
    image_path = file_path  # 替换为自己的图像路径
    result = reader.readtext(image_path)
    texts = ""

    # 输出提取的文字
    for detection in result:
        texts = texts + '\n' + detection[1]
    res = texts

    # # 将列表转换为JSON字符串
    # res_json = json.dumps(res, ensure_ascii=False, indent=4)
    return res


if __name__ == "__main__":
    file_path = '../uploads/ocr/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-06-14%20171426_20240616212447A004.png'
    model_path = 'F:\桌面\Git\AI-PD-UI\weights\ocr'
    de = ocr_detection(file_path, model_path)
