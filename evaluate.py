# evaluate.py
import torch
import torch.nn as nn

# Import các thành phần từ 'src'
import src.config as config
from src.model import SimpleCNN
from src.dataset import get_dataloaders
from src.engine import validate_one_epoch


def main():
    print(f"Bắt đầu đánh giá model trên thiết bị: {config.DEVICE}")
    print(f"Bắt đầu đánh giá model trên tập TEST...")

    # 1. Tải dữ liệu TEST
    dataloaders, class_names = get_dataloaders(get_train=False, get_val=False, get_test=True)
    test_dataloader = dataloaders['test']

    print(f"Đã tải {len(test_dataloader.dataset)} ảnh test.")

    # 2. Khởi tạo model
    model = SimpleCNN(num_classes=len(class_names)).to(config.DEVICE)

    # 3. Tải trọng số đã train
    try:
        model.load_state_dict(
            torch.load(config.MODEL_PATH, map_location=config.DEVICE)
        )
        print(f"Đã tải trọng số từ: {config.MODEL_PATH}")
    except FileNotFoundError:
        print(f"LỖI: không tìm thấy file model tại {config.MODEL_PATH}")
        print("Bạn cần chạy train.py trước.")
        return
    except Exception as e:
        print(f"LỖI khi tải model: {e}")
        return

    # 4. Định nghĩa Loss
    criterion = nn.CrossEntropyLoss()

    # 5. Chạy đánh giá
    print("Đang chạy đánh giá...")
    test_loss, test_acc = validate_one_epoch(
        model, test_dataloader, criterion, config.DEVICE
    )

    # 6. In kết quả
    print("\n" + "=" * 30)
    print("       KẾT QUẢ ĐÁNH GIÁ")
    print("=" * 30)
    print(f"> Test Loss: {test_loss:.4f}")
    print(f"> Độ chính xác (Accuracy): {test_acc * 100:.2f} %")
    print("=" * 30)


if __name__ == "__main__":
    main()