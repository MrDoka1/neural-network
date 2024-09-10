import os

# Укажите путь к папке с изображениями
folder_path = './download'

# Получаем список всех файлов в папке
files = os.listdir(folder_path)

# Проходимся по каждому файлу в папке
for filename in files:
    # Проверяем, соответствует ли имя файла формату "sleep<номер>.jpg"
    if filename.startswith("sleep") and filename.endswith(".jpg"):
        # Получаем номер из имени файла
        number = filename[5:-4]  # "sleep" - это 5 символов, ".jpg" - 4 символа
        # Создаем новое имя файла
        new_filename = f"awake{number}.jpg"
        # Переименовываем файл
        old_filepath = os.path.join(folder_path, filename)
        new_filepath = os.path.join(folder_path, new_filename)
        os.rename(old_filepath, new_filepath)
        print(f'Переименован файл: {filename} -> {new_filename}')
