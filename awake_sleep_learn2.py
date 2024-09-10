import matplotlib.pyplot as plt
from keras import Sequential, Input
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.src.legacy.preprocessing.image import ImageDataGenerator

# Параметры
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Директории с данными
train_dir = './download/train'
validation_dir = './download/validation'

# Генераторы данных с аугментациями для тренировочных данных и без аугментаций для валидации
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.1, horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Загрузка данных
train_generator = train_datagen.flow_from_directory(train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

# Создание модели
model = Sequential([
    Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 класса: awake, sleep, none
])

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Коллбэки
checkpoint = ModelCheckpoint('model-{epoch:03d}-{val_accuracy:03f}.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

# Обучение модели
# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // BATCH_SIZE,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // BATCH_SIZE,
#     epochs=100,
#     callbacks=[checkpoint, early_stopping]
# )
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=100,
    callbacks=[early_stopping]
)


model.save('awake_sleep_2.keras')

# Визуализация процесса обучения
plt.figure(figsize=(12, 4))

# График точности
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# График потерь
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
