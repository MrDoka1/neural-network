import cv2
import mediapipe as mp

# Инициализация модели
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Захват видео
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Преобразуем изображение в RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Детекция лица
        results = face_detection.process(image_rgb)

        # Если обнаружены лица, обрабатываем их
        if results.detections:
            for detection in results.detections:
                # Получаем координаты ограничивающего прямоугольника
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)

                # Вырезаем область лица
                face_image = image[y:y+h, x:x+w]

                # Преобразуем изображение в оттенки серого
                face_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

                # Изменяем размер на 48x48
                face_resized = cv2.resize(face_gray, (48, 48))

                # Отображаем изображение лица в отдельном окне
                cv2.imshow('Face (48x48 grayscale)', face_resized)

                # Рисуем прямоугольник вокруг лица на основном изображении
                mp_drawing.draw_detection(image, detection)

        # Показ изображения с обнаружением лица
        cv2.imshow('Face Detection', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
