import pandas as pd
import os
import numpy as np
from keras.src.utils import load_img, img_to_array
from keras.src.utils import to_categorical
# from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten


def load_data_from_csv(csv_path, image_dir):
    df = pd.read_csv(csv_path)
    images = []
    labels = []

    for idx, row in df.iterrows():
        file_path = os.path.join(image_dir, row['filename'])
        if os.path.exists(file_path):
            img = load_img(file_path, target_size=(224, 224))  # измените размер согласно вашим требованиям
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(row['class'])

    images = np.array(images)
    labels = to_categorical(labels)  # Преобразование меток в one-hot encoding

    return images, labels


# Путь к CSV-файлам и директориям с изображениями
train_csv_path = '/datasets/no_name/train/_annotations.csv'
train_image_dir = '/datasets/no_name/train/'
open('')

valid_csv_path = '/datasets/no_name/valid/_annotations.csv'
valid_image_dir = '/datasets/no_name/valid/'

# Загрузка тренировочных данных
X_train, y_train = load_data_from_csv(train_csv_path, train_image_dir)

# Загрузка валидационных данных
X_valid, y_valid = load_data_from_csv(valid_csv_path, valid_image_dir)

# Построение модели
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')  # Количество классов динамически
])

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
