from flask import Flask, render_template, request
import os

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/text', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'
    upload_file_path = os.path.join('uploads/text', file.filename)
    file.save(upload_file_path)
    return 'File uploaded successfully'


if __name__ == '__main__':
    app.run(debug=True)
