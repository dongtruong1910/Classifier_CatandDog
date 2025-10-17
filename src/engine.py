# src/engine.py
import torch


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Thực hiện training trong 1 epoch."""
    model.train()  # Chuyển model sang chế độ train

    running_loss = 0.0
    running_corrects = 0

    # Lặp qua từng batch dữ liệu
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 1. Xóa gradient
        optimizer.zero_grad()

        # 2. Forward
        outputs = model(inputs)

        # 3. Tính loss
        loss = criterion(outputs, labels)

        # 4. Backward
        loss.backward()

        # 5. Cập nhật trọng số
        optimizer.step()

        # Thống kê
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    return epoch_loss, epoch_acc


def validate_one_epoch(model, dataloader, criterion, device):
    """Thực hiện validation trong 1 epoch."""
    model.eval()  # Chuyển model sang chế độ eval
    val_loss = 0.0
    val_corrects = 0

    with torch.no_grad():  # Tắt tính toán gradient
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)

    val_epoch_loss = val_loss / len(dataloader.dataset)
    val_epoch_acc = val_corrects.double() / len(dataloader.dataset)

    return val_epoch_loss, val_epoch_acc