import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET


def load_data(data_dir, target_size=(224, 224)):
    images = []
    labels = []
    boxes = []

    # Проходим по каждой папке в data_dir (awake, sleep, none)
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)

        # Проходим по каждому файлу в папке
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.jpg'):  # Обрабатываем только изображения
                img_path = os.path.join(folder_path, file_name)
                xml_path = img_path.replace('.jpg', '.xml')

                # Загружаем изображение
                img = cv2.imread(img_path)
                original_height, original_width = img.shape[:2]

                # Изменяем размер изображения
                img_resized = cv2.resize(img, target_size)
                images.append(img_resized)

                if os.path.exists(xml_path):  # Если файл xml существует
                    # Парсим XML файл
                    tree = ET.parse(xml_path)
                    root = tree.getroot()

                    for obj in root.findall('object'):
                        label = obj.find('name').text
                        bbox = obj.find('bndbox')
                        xmin = int(bbox.find('xmin').text)
                        ymin = int(bbox.find('ymin').text)
                        xmax = int(bbox.find('xmax').text)
                        ymax = int(bbox.find('ymax').text)

                        # Корректируем координаты bbox с учетом изменения размера изображения
                        xmin_resized = int(xmin * target_size[0] / original_width)
                        ymin_resized = int(ymin * target_size[1] / original_height)
                        xmax_resized = int(xmax * target_size[0] / original_width)
                        ymax_resized = int(ymax * target_size[1] / original_height)

                        labels.append(label)
                        boxes.append([xmin_resized, ymin_resized, xmax_resized, ymax_resized])
                else:
                    # Если xml файла нет, то это изображение без аннотации (none)
                    labels.append('none')
                    boxes.append([0, 0, 0, 0])  # или можно оставить пустым список

    return np.array(images), np.array(labels), np.array(boxes)


# Пример использования функции:
train_dir = './download/train'
validation_dir = './download/validation'

train_images, train_labels, train_boxes = load_data(train_dir)
val_images, val_labels, val_boxes = load_data(validation_dir)
