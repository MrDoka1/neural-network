import cv2
import tensorflow as tf
from keras import Sequential, Input
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.optimizers import Adam
from keras.src.optimizers.schedules import ExponentialDecay
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.src.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# Пути к данным
data_dir = './datasets/fer-2013 — копия/train'  # Замените на путь к папке с данными

# Создание генераторов изображений
datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(48, 48),  # Размер изображений
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical',
    subset='training')

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical',
    subset='validation')

# # Создание модели
# model = Sequential([
#     Input(shape=(48, 48, 1)),
#     Conv2D(32, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(train_generator.num_classes, activation='softmax')
# ])
#
# # Компиляция модели
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks для сохранения модели
checkpoint = ModelCheckpoint('models/clear_model_epoch_3_{epoch:02d}.keras', save_freq='epoch')

emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
						input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

emotion_model.summary()

cv2.ocl.setUseOpenCL(False)

initial_learning_rate = 0.0001
lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=100000,
							decay_rate=0.96)

optimizer = Adam(learning_rate=lr_schedule)

emotion_model.compile(loss='categorical_crossentropy', optimizer=optimizer,
					metrics=['accuracy'])

emotion_model_info = emotion_model.fit(
    train_generator,
    # steps_per_epoch=28709 // 64,
    epochs=30,
    validation_data=validation_generator,
    # validation_steps=7178 // 64,
    callbacks=[checkpoint])

history = emotion_model_info


# пусто SGD
# 2 adam

# Обучение модели
# history = model.fit(
#     train_generator,
#     epochs=100,  # Количество эпох
#     validation_data=validation_generator,
#     callbacks=[checkpoint])

# Построение графиков потерь и правильности
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
