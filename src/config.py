from albumentations.pytorch.transforms import ToTensorV2
import albumentations as a
import data as d
import os 
import torch
from utils import collate_fn, display_image
width = 224
height = 224
root = 'drive/MyDrive'
data_folder = os.path.join(root, "Slope RGB Fixed Val Augmentation")

train_folder = os.path.join(data_folder, 'train')
test_folder = os.path.join(data_folder, 'test')
val_folder = os.path.join(data_folder, 'val')

train_data = os.path.join(train_folder, 'data')
train_labels = os.path.join(train_folder, 'classification.csv')

test_data = os.path.join(test_folder, 'data')
test_labels = os.path.join(test_folder, 'classification.csv')

val_data = os.path.join(val_folder, 'data')
val_labels = os.path.join(val_folder, 'classification.csv')

parts_data = os.path.join(data_folder, 'parts')

batch_size = 10

train_transform = a.Compose([
    a.Resize(width, height),
    a.HorizontalFlip(p=0.5),
    a.RandomBrightnessContrast(p=0.2),
    a.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
], bbox_params=a.BboxParams(format='pascal_voc', label_fields=['class_labels']))

test_transform = a.Compose([
     a.Resize(width, height),
     a.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
], bbox_params=a.BboxParams(format='pascal_voc', label_fields=['class_labels'])) 


train_dataset = d.Mound(train_labels, train_data, train_transform)
test_dataset = d.Mound(test_labels, test_data, test_transform)
val_dataset = d.Mound(val_labels, val_data, test_transform)

data_size = len(train_dataset)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=4,
    collate_fn=collate_fn
)


val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=4,
    collate_fn=collate_fn
)


test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=4,
    collate_fn=collate_fn
    )

# train on gpu if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 2 # one class (class 0) is dedicated to the "background"

# get the model using our helper function
# model = get_object_detection_model(num_classes)

# move model to the right device
# model.to(device)

# construct an optimizer
# params = [p for p in model.parameters() if p.requires_grad]
# optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 30 epochs
# lr_scheduler = torch.optim.lr_scheduler.StepLR(
#   optimizer,
#   step_size=30,
#   gamma=0.1
# )

if __name__ == "__main__":
    img, target = train_dataset[1]
    display_image(img, target)