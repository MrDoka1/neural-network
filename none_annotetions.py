import os
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString
from PIL import Image


def create_pascal_voc_annotation(filename, width, height):
    annotation = Element("annotation")

    folder = SubElement(annotation, "folder")
    folder.text = "download"

    fname = SubElement(annotation, "filename")
    fname.text = filename

    path = SubElement(annotation, "path")
    path.text = "C:\\Users\mrdok\PycharmProjects\\neural network\\download\\" + filename
    print(os.path.abspath(filename))

    source = SubElement(annotation, "source")
    database = SubElement(source, "database")
    database.text = "Unknown"

    size = SubElement(annotation, "size")
    w = SubElement(size, "width")
    w.text = str(width)
    h = SubElement(size, "height")
    h.text = str(height)
    depth = SubElement(size, "depth")
    depth.text = "3"  # Предполагается, что изображения RGB

    segmented = SubElement(annotation, "segmented")
    segmented.text = "0"

    return annotation


def save_xml(annotation, xml_filename):
    xml_str = tostring(annotation)
    dom = parseString(xml_str)
    with open(xml_filename, "w") as f:
        f.write(dom.toprettyxml(indent="  "))


def main(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            file_path = os.path.join(directory, filename)

            # Используем PIL для получения размеров изображения
            with Image.open(file_path) as img:
                width, height = img.size

            annotation = create_pascal_voc_annotation(filename, width, height)
            xml_filename = os.path.splitext(file_path)[0] + ".xml"
            save_xml(annotation, xml_filename)
            print(f"Создана аннотация для {filename}")


if __name__ == "__main__":
    directory = "./download"  # Задайте путь к вашей директории с изображениями
    main(directory)
