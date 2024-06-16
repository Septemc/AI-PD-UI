import torch
import torch.nn
import json
import torchvision.transforms as transforms
from PIL import Image
from models.Nets import resnet50


def pic_detection(file_path, model_path, crop=448):
    logs_detection = []
    model = resnet50()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    result_details = []
    # Transform
    trans_init = []
    # 定义数据转换操作
    transform = transforms.Compose([
        transforms.Resize(512),  # 将图像大小调整为256x256
        transforms.CenterCrop(448),  # 从图像中心裁剪出224x224的区域
        transforms.ToTensor(),  # 将图像转换为PyTorch张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 图像标准化
    ])
    img = transform(Image.open(file_path).convert('RGB'))
    # # 显示图像
    # plt.imshow(img.permute(1, 2, 0))  # 将维度从(C, H, W)转换为(H, W, C)以便显示
    # plt.axis('off')  # 关闭坐标轴
    # plt.show()

    with torch.no_grad():
        in_tens = img.unsqueeze(0)
        in_tens = in_tens.to(device)
        prob = model(in_tens).sigmoid()
        res = prob[0, 0].item() * 100
        res = round(res, 3)
        # .sigmoid().item() * 100

    result = {
        "ai_prob": res,
    }

    # 将Python字典转换为JSON格式的字符串
    result_json = json.dumps(result, ensure_ascii=False, indent=4)
    return result_json


# if __name__ == '__main__':
#     pic_detection('../uploads/image/test3.png', '../weights/image/blur_jpg_prob0.5.pth', crop=256)
