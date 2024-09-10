import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

# Параметры
img_width, img_height = 150, 150
batch_size = 32
epochs = 50

# Путь к данным
base_data_dir = 'download'
train_data_dir = os.path.join(base_data_dir, 'train')
validation_data_dir = os.path.join(base_data_dir, 'validation')

# Преобразования для данных
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((img_width, img_height)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validation': transforms.Compose([
        transforms.Resize((img_width, img_height)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Загрузка данных
image_datasets = {
    x: datasets.ImageFolder(os.path.join(base_data_dir, x), data_transforms[x])
    for x in ['train', 'validation']
}

dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
    for x in ['train', 'validation']
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
class_names = image_datasets['train'].classes

# Создание модели
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * (img_width // 8) * (img_height // 8), 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(nn.ReLU()(self.bn1(self.conv1(x))))
        x = self.pool2(nn.ReLU()(self.bn2(self.conv2(x))))
        x = self.pool3(nn.ReLU()(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * (img_width // 8) * (img_height // 8))
        x = self.dropout(nn.ReLU()(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x

def train_model(model, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).float().view(-1, 1)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(outputs.round() == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model

if __name__ == '__main__':
    model = SimpleCNN()
    # Определение функции потерь и оптимизатора
    criterion = nn.BCELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)

    # Обучение модели
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = train_model(model, criterion, optimizer, num_epochs=epochs)

    # Сохранение модели
    torch.save(model.state_dict(), 'model.pth')

    # Вывод графиков точности и потерь
    # (Этот код нужно адаптировать под вашу задачу, так как PyTorch не сохраняет историю потерь и точности автоматически)