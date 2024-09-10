import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image, ImageDraw


def load_data(data_dir, target_size=(224, 224)):
    images = []
    labels = []
    boxes = []

    # Проходим по каждой папке в data_dir (awake, sleep, none)
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)

        # Проверка, что это директория, а не файл
        if not os.path.isdir(folder_path):
            continue

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


                # if file_name == "sleep430.jpg":
                #     tree = ET.parse(xml_path)
                #     root = tree.getroot()
                #     box = []
                #     rbox = []
                #     for obj in root.findall('object'):
                #         bbox = obj.find('bndbox')
                #         xmin = int(bbox.find('xmin').text)
                #         ymin = int(bbox.find('ymin').text)
                #         xmax = int(bbox.find('xmax').text)
                #         ymax = int(bbox.find('ymax').text)
                #         box.append([xmin, ymin, xmax, ymax])
                #         xmin_resized = int(xmin * target_size[0] / original_width)
                #         ymin_resized = int(ymin * target_size[1] / original_height)
                #         xmax_resized = int(xmax * target_size[0] / original_width)
                #         ymax_resized = int(ymax * target_size[1] / original_height)
                #         rbox.append([xmin_resized, ymin_resized, xmax_resized, ymax_resized])
                #         print(box)
                #         print(rbox)
                #
                #     # Draw original bounding box
                #     img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                #     draw = ImageDraw.Draw(img_pil)
                #     for b in box:
                #         draw.rectangle(b, outline="red", width=2)
                #     img_pil.show()
                #     img_pil2 = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
                #     draw2 = ImageDraw.Draw(img_pil2)
                #     for b in rbox:
                #         draw2.rectangle(b, outline="red", width=2)
                #     img_pil2.show()

                if os.path.exists(xml_path):  # Если файл xml существует
                    # Парсим XML файл
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    if len(root.findall('object')) == 0:
                        images.append(img_resized)
                        labels.append('none')
                        boxes.append([0, 0, 0, 0])  # или можно оставить пустым список
                    else:
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

                            images.append(img_resized)
                            labels.append(label)
                            boxes.append([xmin_resized, ymin_resized, xmax_resized, ymax_resized])

    return np.array(images), np.array(labels), np.array(boxes)


train_dir = './download/train'
validation_dir = './download/validation'

train_images, train_labels, train_boxes = load_data(train_dir)
val_images, val_labels, val_boxes = load_data(validation_dir)


def load_train_data():
    return load_data(train_dir)


def load_validation_data():
    return load_data(validation_dir)


def display_image_with_bbox(images, labels, boxes, index):
    """
    Функция для отображения изображения с рамкой вокруг объекта.

    Параметры:
    - images: массив изображений
    - labels: массив меток
    - boxes: массив координат рамок (bbox)
    - index: индекс изображения, которое нужно отобразить
    """
    # Извлекаем изображение, метку и координаты рамки
    img = images[index]
    label = labels[index]
    bbox = boxes[index]

    # Отображаем изображение
    fig, ax = plt.subplots(1)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Конвертация из BGR в RGB для корректного отображения в matplotlib

    # Если рамка не пустая, добавляем её на изображение
    if not np.array_equal(bbox, [0, 0, 0, 0]):
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # Добавляем текст метки
    plt.text(xmin, ymin - 10, label, color='red', fontsize=12, weight='bold', backgroundcolor='white')

    plt.show()


# Пример использования функции:
display_image_with_bbox(train_images, train_labels, train_boxes, 600)

