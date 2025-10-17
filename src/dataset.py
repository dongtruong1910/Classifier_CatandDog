# src/dataset.py
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import src.config as config

image_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size=(config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    'val': transforms.Compose([
        transforms.Resize(size=(config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}



def get_dataloaders(get_train=True, get_val=True, get_test=False):
    """
    Tạo và trả về DataLoaders.
    Có thể chọn tạo 'train', 'val', hoặc 'test'.
    """
    image_datasets = {}
    dataloaders = {}

    if get_train:
        print("Đang tải dữ liệu TRAIN...")
        image_datasets['train'] = datasets.ImageFolder(
            config.TRAIN_DIR,
            transform=image_transforms['train']
        )
        dataloaders['train'] = DataLoader(
            image_datasets['train'],
            batch_size=config.BATCH_SIZE,
            shuffle=True
        )

    if get_val:
        print("Đang tải dữ liệu VALIDATION...")
        image_datasets['val'] = datasets.ImageFolder(
            config.VAL_DIR,
            transform=image_transforms['val']
        )
        dataloaders['val'] = DataLoader(
            image_datasets['val'],
            batch_size=config.BATCH_SIZE,
            shuffle=False
        )

    if get_test:
        print("Đang tải dữ liệu TEST...")
        image_datasets['test'] = datasets.ImageFolder(
            config.TEST_DIR,
            transform=image_transforms['val']  # Dùng transform của val
        )
        dataloaders['test'] = DataLoader(
            image_datasets['test'],
            batch_size=config.BATCH_SIZE,
            shuffle=False  # Không xáo trộn tập test
        )

    # Lấy class_names từ train (nếu có) hoặc các tập khác
    class_names = []
    if get_train:
        class_names = image_datasets['train'].classes
    elif get_val:
        class_names = image_datasets['val'].classes
    elif get_test:
        class_names = image_datasets['test'].classes

    if class_names:
        print(f"Các lớp: {class_names}")

    return dataloaders, class_names