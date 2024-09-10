import cv2
import time

# Загрузка каскадного классификатора для обнаружения лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Открытие видеопотока с камеры
cap = cv2.VideoCapture(0)

# Переменные для отслеживания времени и количества сохранённых изображений
last_saved_time = 0
image_count = 0
max_images = 10

while image_count < max_images:
    # Чтение кадра из видеопотока
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразование кадра в градации серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц на кадре
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Текущее время
    current_time = time.time()

    # Проверка, прошла ли секунда с последнего сохранения
    # if current_time - last_saved_time >= 1 and len(faces) > 0:
    if cv2.waitKey(1) & 0xFF == ord(' '):
        for (x, y, w, h) in faces:
            # Обрезка изображения лица`
            face_img = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)

            # Сохранение изображения лица
            filename = f"images/face_{image_count + 1}_{int(current_time)}.jpg"
            cv2.imwrite(filename, face_img)
            image_count += 1

            # Прерывание цикла, если достигнуто максимальное количество изображений
            if image_count >= max_images:
                break

        # Обновление времени последнего сохранения
        last_saved_time = current_time

    # Отображение кадра с нарисованными прямоугольниками вокруг лиц
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Отображение количества сохранённых изображений
    cv2.putText(frame, f"Saved Images: {image_count}/{max_images}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Face Detection', frame)

    # Завершение работы при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
