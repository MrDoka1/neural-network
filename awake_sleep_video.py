import cv2
import numpy as np
from keras.src.losses import mean_squared_error, categorical_crossentropy
from keras.src.saving import load_model
import tensorflow as tf

def custom_loss(y_true, y_pred):
    bbox_true = y_true[:, :4]
    class_true = y_true[:, 4:]

    bbox_pred = y_pred[:, :4]
    class_pred = y_pred[:, 4:]

    bbox_loss = tf.reduce_mean(tf.square(bbox_true - bbox_pred))
    class_loss = tf.keras.losses.sparse_categorical_crossentropy(class_true, class_pred)

    return bbox_loss + class_loss

# def custom_loss(y_true, y_pred):
#     # Координаты bounding box: (xmin, ymin, xmax, ymax)
#     y_true_bbox = y_true[:, :4]
#     y_pred_bbox = y_pred[:, :4]
#
#     # Классы: awake, sleep, none
#     y_true_class = tf.one_hot(tf.cast(y_true[:, 4], tf.int32), depth=3)
#     y_pred_class = y_pred[:, 4:]
#
#     # 1. Loss на регрессию координат bounding box
#     bbox_loss = tf.reduce_mean(mean_squared_error(y_true_bbox, y_pred_bbox))
#
#     # 2. Loss на классификацию
#     class_loss = tf.reduce_mean(categorical_crossentropy(y_true_class, y_pred_class))
#
#     # Общая Loss
#     total_loss = bbox_loss + class_loss
#
#     return total_loss

IMG_SIZE = (224, 224)

# Загрузка обученной модели
model = load_model('models/awake_sleep.keras', custom_objects={'custom_loss': custom_loss})


# Функция для предсказания и визуализации результата на кадре
def predict_and_visualize_on_frame(frame, model):
    orig_size = frame.shape[:2]
    resized_frame = cv2.resize(frame, IMG_SIZE)
    resized_frame = np.expand_dims(resized_frame / 255.0, axis=0)

    prediction = model.predict(resized_frame)
    bbox_pred = prediction[0, :4]
    class_pred = np.argmax(prediction[0, 4:])

    # Восстановление координат bbox
    xmin = int(bbox_pred[0] * orig_size[1] / IMG_SIZE[0])
    ymin = int(bbox_pred[1] * orig_size[0] / IMG_SIZE[1])
    xmax = int(bbox_pred[2] * orig_size[1] / IMG_SIZE[0])
    ymax = int(bbox_pred[3] * orig_size[0] / IMG_SIZE[1])

    label = ['Awake', 'Sleep', 'None'][class_pred]

    if label != 'None':
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame


# Функция для обработки видеопотока с камеры
def process_camera_stream(model):
    cap = cv2.VideoCapture(0)  # Используем камеру по умолчанию (индекс 0)

    if not cap.isOpened():
        print("Ошибка: не удалось получить доступ к камере")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Обрабатываем кадр
        processed_frame = predict_and_visualize_on_frame(frame, model)

        # Отображение обработанного кадра
        cv2.imshow('Camera Stream - Awake or Sleep Detection', processed_frame)

        # Завершение работы по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()


# Запуск обработки видеопотока с камеры
process_camera_stream(model)
