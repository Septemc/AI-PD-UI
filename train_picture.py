import torch
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from models.Nets import resnet50


def get_dataset(opt):
    dset_lst = []
    for cls in opt.classes:
        root = opt.dataroot + '/' + cls
        dset = dataset_folder(opt, root)
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)


def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler


from pic_pack.options.train_options import TrainOptions
from pic_pack.data.datasets import dataset_folder

"""Currently assumes jpg_prob, blur_prob 0 or 1"""


def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt


if __name__ == '__main__':
    # 加载图像数据
    data_root = 'datasets/big_pic'
    train_path = data_root + '/train'
    val_path = data_root + '/val'

    # 定义数据转换操作
    transform = transforms.Compose([
        transforms.Resize(512),  # 将图像大小调整为256x256
        transforms.CenterCrop(448),  # 从图像中心裁剪出224x224的区域
        transforms.ToTensor(),  # 将图像转换为PyTorch张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 图像标准化
    ])

    dataset_train = datasets.ImageFolder(root=train_path, transform=transform)
    dataset_val = datasets.ImageFolder(root=val_path, transform=transform)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=4, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=4, shuffle=True)

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'当前使用设备:{device}')
    model = resnet50().to(device)
    epochs = 100

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_losses = 0
        correct_num = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            images_data, labels_data = images.to(device), labels.to(device)
            outputs = model(images_data)
            loss = criterion(outputs, labels_data)
            loss.backward()
            optimizer.step()
            total_losses += loss.item()

            # 计算准确率
            predictions = outputs.argmax(dim=1)
            correct_predictions = (predictions == labels_data).float().sum().item()
            correct_num += correct_predictions

        accuracy = correct_num / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{epochs} Average Loss: {total_losses} Accuracy: {accuracy}')

        # 在每个epoch结束后评估模型在测试集上的准确率
        model.eval()
        correct_num = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images_data, labels_data = images.to(device), labels.to(device)
                outputs = model(images_data)
                predictions = outputs.argmax(dim=1)
                correct_predictions = (predictions == labels_data).float().sum().item()
                correct_num += correct_predictions

            test_accuracy = correct_num / len(val_loader.dataset)
            print(f'Test Accuracy: {test_accuracy}')

    # 保存模型文件
    torch.save(model.state_dict(), 'logs/weights/picture/model_1.pth')
    print('Training completed!')
