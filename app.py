import os
from flask import Flask, request
from funModule.text_detection import doc_detection, txt_detection, pdf_detection
from funModule.picture_detection import pic_detection
from funModule.OCR_detection import ocr_detection

app = Flask(__name__)


@app.route('/text', methods=['GET', 'POST'])
def text_home():
    if request.method == 'POST':
        file = request.files['file']
        model_idx = request.form['model']
        model_path = f'weights/text/model_{model_idx}.pth'
        # 临时保存文件
        upload_file_path = os.path.join('uploads/text', file.filename)
        file.save(upload_file_path)
        if '.txt' in file.filename:
            prediction_result = txt_detection(upload_file_path, model_path)
        elif '.doc' in file.filename:
            prediction_result = doc_detection(upload_file_path, model_path)
        elif '.pdf' in file.filename:
            prediction_result = pdf_detection(upload_file_path, model_path)
        else:
            return "Error: file"
        return prediction_result


@app.route('/image', methods=['GET', 'POST'])
def image_home():
    if request.method == 'POST':
        file = request.files['file']
        model_path = 'weights/picture/model_ac=0.846.pth'
        # 临时保存文件
        upload_file_path = os.path.join('uploads/picture', file.filename)
        file.save(upload_file_path)
        prediction_result = pic_detection(upload_file_path, model_path)
        return prediction_result


@app.route('/OCR', methods=['GET', 'POST'])
def ocr_home():
    if request.method == 'POST':
        file = request.files['file']
        model_path = 'weights/ocr'
        # 临时保存文件
        upload_file_path = os.path.join('uploads/ocr', file.filename)
        file.save(upload_file_path)
        prediction_result = ocr_detection(upload_file_path, model_path)
        return prediction_result


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4010, debug=False)
