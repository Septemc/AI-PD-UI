import os
from flask import Flask, render_template, request, send_from_directory, make_response
from funModule.text_detection import doc_detection, txt_detection, pdf_detection
from funModule.picture_detection import pic_detection
from funModule.OCR_detection import ocr_detection

app = Flask(__name__)

# 如果需要，可以明确设置静态文件的路由
app.config['STATIC_FOLDER'] = 'static'


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/text', methods=['GET', 'POST'])
def text_home():
    if request.method == 'POST' or request.method == 'GET':
        file = request.files['file']
        # model_idx = request.form['model']
        model_idx = 1
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
            return "Error: 仅支持.txt、.doc和.pdf格式的文件。请上传正确格式的文件。"
        return make_response(prediction_result)


@app.route('/image', methods=['GET', 'POST'])
def image_home():
    if request.method == 'POST':
        file = request.files['file']
        model_path = 'weights/picture/model_ac=0.846.pth'
        # 临时保存文件
        upload_file_path = os.path.join('uploads/picture', file.filename)
        file.save(upload_file_path)
        prediction_result = pic_detection(upload_file_path, model_path)
        return make_response(prediction_result)


@app.route('/OCR', methods=['GET', 'POST'])
def ocr_home():
    if request.method == 'POST':
        file = request.files['file']
        model_path = 'weights/ocr'
        # 临时保存文件
        upload_file_path = os.path.join('uploads/ocr', file.filename)
        file.save(upload_file_path)
        prediction_result = ocr_detection(upload_file_path, model_path)
        return make_response(prediction_result)


if __name__ == '__main__':
    app.run()
