import os

import cv2
import numpy as np
from keras.src.utils import to_categorical

from create import create

# Директории с данными
train_dir = 'datasets/fer-2013/train'
validation_dir = 'datasets/fer-2013/test'
# train_dir = 'datasets/my-images/train'
# validation_dir = 'datasets/my-images/train'


# Функция для загрузки данных из директории
def load_data(directory):
    data = []
    labels = []
    for label, emotion in enumerate(os.listdir(directory)):
        emotion_dir = os.path.join(directory, emotion)
        for img in os.listdir(emotion_dir):
            img_path = os.path.join(emotion_dir, img)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (48, 48))
            image = image.astype('float32') / 255.0
            data.append(image)
            labels.append(label)
    return np.array(data), np.array(labels)


# Загрузка данных
x_train, y_train = load_data(train_dir)
x_val, y_val = load_data(validation_dir)

# Ресайз данных
x_train = x_train.reshape(-1, 48, 48, 1)
x_val = x_val.reshape(-1, 48, 48, 1)

# Преобразование меток в one-hot encoding
y_train = to_categorical(y_train, num_classes=7)
y_val = to_categorical(y_val, num_classes=7)
# y_train = to_categorical(y_train, num_classes=5)
# y_val = to_categorical(y_val, num_classes=5)

print("1" + str(x_train))
print("2" + str(y_train))
print("3" + str(x_val))
print("4" + str(y_val))

create(x_train, y_train, x_val, y_val)
