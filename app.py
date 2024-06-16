import os
import requests
from flask import Flask, request
from urllib.parse import urlparse, unquote
from funModule.text_detection import doc_detection, txt_detection, pdf_detection
from funModule.picture_detection import pic_detection
from funModule.OCR_detection import ocr_detection

app = Flask(__name__)


def download(url, mode):
    req = requests.get(url)
    decoded_url = unquote(url)
    parsed_url = urlparse(decoded_url)
    # 获取文件名称
    filename = parsed_url.path.split('/')[-1]
    # 保存到项目static目录下
    save_path = os.path.join(f'uploads/{mode}', filename)
    if req.status_code != 200:
        print('下载异常')
        return
    try:
        with open(save_path, 'wb') as f:
            f.write(req.content)
            return save_path
    except Exception as e:
        print(e)


@app.route('/text', methods=['GET', 'POST'])
def text_home():
    if request.method == 'POST':
        file_url = request.get_json()['url'][0]
        model_idx = request.get_json()['model'][0]

        mode = 'text'
        upload_file_path = download(file_url, mode)

        model_path = f'weights/text/model_{model_idx}.pth'

        if '.txt' in upload_file_path:
            prediction_result = txt_detection(upload_file_path, model_path)
        elif '.doc' in upload_file_path:
            prediction_result = doc_detection(upload_file_path, model_path)
        elif '.pdf' in upload_file_path:
            prediction_result = pdf_detection(upload_file_path, model_path)
        else:
            return "Error: file"
        return prediction_result


@app.route('/image', methods=['GET', 'POST'])
def image_home():
    if request.method == 'POST':
        file_url = request.get_json()['url'][0]
        mode = 'image'
        # 临时保存文件
        upload_file_path = download(file_url, mode)

        model_path = 'weights/picture/model_ac=0.846.pth'
        prediction_result = pic_detection(upload_file_path, model_path)
        return prediction_result


@app.route('/OCR', methods=['GET', 'POST'])
def ocr_home():
    if request.method == 'POST':
        file_url = request.get_json()['url'][0]
        model_path = 'weights/ocr'
        # 临时保存文件
        mode = 'ocr'
        # 临时保存文件
        upload_file_path = download(file_url, mode)
        prediction_result = ocr_detection(upload_file_path, model_path)
        return prediction_result


if __name__ == '__main__':
    app.run(host='172.20.10.8', port=4010, debug=True)
