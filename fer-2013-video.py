import cv2
import numpy as np
from keras.src.saving import load_model

# Загрузка модели
model = load_model('fer-2013_model_epoch_14.keras')

# Определение эмоций
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Захват видео с камеры
cap = cv2.VideoCapture(0)  # 0 - номер устройства камеры

while True:
    # Чтение кадра
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразование кадра в оттенки серого
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Изменение размера кадра на 48x48 пикселей (размер изображения в вашем наборе данных)
    resized_frame = cv2.resize(gray_frame, (48, 48))

    # Нормализация пикселей
    normalized_frame = resized_frame / 255.0

    # Преобразование в формат, ожидаемый моделью
    input_frame = np.expand_dims(normalized_frame, axis=0)  # Добавляем ось для пакета
    input_frame = np.expand_dims(input_frame, axis=-1)  # Добавляем ось для канала

    # Печать данных для отладки
    print(f"Input shape: {input_frame.shape}")
    print(f"Normalized pixel values: {input_frame[0, :5, :5, 0]}")  # Печать первых 5x5 пикселей

    # Предсказание эмоции
    predictions = model.predict(input_frame)
    emotion_index = np.argmax(predictions)
    emotion_label = emotion_labels[emotion_index]
    confidence = np.max(predictions)

    # Отображение результата на экране
    cv2.putText(frame, f'{emotion_label} ({confidence:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Отображение кадра с результатом
    cv2.imshow('Emotion Recognition', frame)

    # Выход из цикла при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
