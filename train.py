# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import os

# Import các thành phần từ 'src'
import src.config as config
from src.model import SimpleCNN
from src.dataset import get_dataloaders
from src.engine import train_one_epoch, validate_one_epoch


def main():
    print(f"Sử dụng thiết bị: {config.DEVICE}")

    # 1. Tải dữ liệu
    dataloaders, class_names = get_dataloaders()
    print(f"Số lớp: {len(class_names)}")

    # 2. Khởi tạo model
    model = SimpleCNN(num_classes=len(class_names)).to(config.DEVICE)

    # 3. Định nghĩa Loss và Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # (Tùy chọn) Tạo thư mục lưu model nếu chưa có
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)

    # 4. Vòng lặp training chính
    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        print("-" * 10)

        # Chạy 1 epoch train
        train_loss, train_acc = train_one_epoch(
            model, dataloaders['train'], criterion, optimizer, config.DEVICE
        )
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")

        # Chạy 1 epoch validation
        val_loss, val_acc = validate_one_epoch(
            model, dataloaders['val'], criterion, config.DEVICE
        )
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        print()

    # 5. Lưu model sau khi train xong
    torch.save(model.state_dict(), config.MODEL_PATH)
    print(f"Đã lưu model tại: {config.MODEL_PATH}")


if __name__ == "__main__":
    main()