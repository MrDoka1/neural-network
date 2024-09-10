import cv2
import numpy as np
from keras.src.saving import load_model

# Параметры
IMG_SIZE = (224, 224)

# Загрузка обученной модели
model = load_model('awake_sleep_2.keras')

# Имена классов
class_names = ['awake', 'none', 'sleep']

# Захват видео с камеры
cap = cv2.VideoCapture(0)

while True:
    # Чтение кадра из видео
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразование кадра в формат, подходящий для модели
    img = cv2.resize(frame, IMG_SIZE)
    img = img / 255.0  # Нормализация
    img = np.expand_dims(img, axis=0)

    # Предсказание класса
    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    class_label = class_names[class_index]

    # Определение вероятности предсказанного класса
    confidence = np.max(predictions)

    # Добавление метки и вероятности на кадр
    label = f'{class_label} ({confidence:.2f})'
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Отображение кадра
    cv2.imshow('Video', frame)

    # Завершение по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
