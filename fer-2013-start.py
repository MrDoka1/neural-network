import pandas as pd
import numpy as np
import tensorflow as tf
from keras.src.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.src.callbacks import ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Для визуализации
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('./datasets/FER-2013 aa/fer2013/fer2013.csv')

# Преобразование пикселей в изображения
def preprocess_pixels(pixels):
    pixels = np.array(pixels.split(), dtype='float32')
    return pixels.reshape(48, 48, 1)

data['pixels'] = data['pixels'].apply(preprocess_pixels)

# Нормализация пикселей
data['pixels'] = data['pixels'] / 255.0

# Преобразование меток в one-hot encoding
lb = LabelBinarizer()
data['emotion'] = lb.fit_transform(data['emotion'])

# Преобразование меток в one-hot encoding
y = to_categorical(data['emotion'], num_classes=7)  # 7 - это количество классов

# Разделение на признаки и метки
X = np.array(data['pixels'].tolist())

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test)

# Построение модели
model = Sequential()

# Первый сверточный слой
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Второй сверточный слой
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Третий сверточный слой
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Выравнивание и полносвязные слои
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(lb.classes_), activation='softmax'))

print(len(lb.classes_), lb.classes_)

# Компиляция модели
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Определение пути для сохранения модели
checkpoint_filepath = 'fer-2013_model_epoch_{epoch:02d}.keras'

# Создание колбэка для сохранения модели каждые 5 эпох
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_freq='epoch',  # Сохранение на уровне эпох
    save_weights_only=False,  # Сохранение всей модели (включая архитектуру)
    save_best_only=False,  # Сохранение каждой пятой эпохи, а не только лучшей модели
    verbose=1  # Выводить сообщение о сохранении модели
)

# Встроим логику для сохранения каждые 5 эпох
class CustomModelCheckpoint(Callback):
    def __init__(self, model_checkpoint, save_interval):
        super().__init__()
        self.model_checkpoint = model_checkpoint
        self.save_interval = save_interval

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_interval == 0:
            # Вызов ModelCheckpoint
            self.model_checkpoint.model = self.model
            self.model_checkpoint.on_epoch_end(epoch, logs)

# Создание кастомного колбэка для сохранения модели каждые 5 эпох
custom_checkpoint_callback = CustomModelCheckpoint(
    model_checkpoint=checkpoint_callback,
    save_interval=5
)

# Обучение модели с использованием кастомного колбэка
history = model.fit(X_train, y_train,
                    epochs=14,
                    batch_size=64,
                    validation_data=(X_test, y_test),
                    verbose=2,
                    # callbacks=[custom_checkpoint_callback]
                    )
model.save('fer-2013_model_epoch_14.keras')

# Оценка модели
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Построение графиков
plt.figure(figsize=(12, 4))

# График точности
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# График потерь
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
