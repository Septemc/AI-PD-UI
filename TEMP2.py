# 假设您已经导入了Python的docx库
from docx import Document
import jieba


def read_and_segment_txt(file_path):
    # 打开txt文件
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # 使用jieba进行分词
    words = jieba.cut(text)
    words_list = list(words)

    # 假设每个段落是由空行分隔的，您可以根据实际情况调整这里
    paragraphs = text.split('\n\n')  # 使用两个换行符来分隔段落

    return paragraphs


# 假设您已经有了上传文件的路径
upload_file_path = 'uploads/text/reat_car_sql.txt'

# 调用函数进行读取和分段处理
paragraphs = read_and_segment_txt(upload_file_path)

# 接下来您可以根据需要处理每个段落
for paragraph in paragraphs:
    # 处理每个段落
    print(paragraph)
