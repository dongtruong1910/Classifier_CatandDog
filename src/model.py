# src/model.py
import torch.nn as nn
from torchvision import models


def create_model(num_classes=2):
    """
    Hàm này tải ResNet18 đã được train
    và thay thế lớp cuối cùng (classifier).
    """

    # 1. Tải model ResNet18 đã được train trên ImageNet
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # 2. "Đóng băng" (Freeze) tất cả các trọng số
    for param in model.parameters():
        param.requires_grad = False

    # 3. Thay thế lớp cuối cùng (fc) để phù hợp với số lớp
    # Lấy số lượng features đầu vào của lớp cuối
    num_ftrs = model.fc.in_features

    # Tạo một lớp Linear mới (với 2 output) để thay thế
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model


# ---
# ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()

        # Tải model ResNet đã được tùy chỉnh
        self.resnet = create_model(num_classes)

    def forward(self, x):
        return self.resnet(x)