from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import requests
import time
import os

# Установите путь, куда будут сохраняться изображения
save_path = "download"  # Укажите нужный путь

if not os.path.exists(save_path):
    os.makedirs(save_path)

# Настройка и запуск браузера
options = webdriver.ChromeOptions()
# options.add_argument('--headless')  # Чтобы браузер работал в фоновом режиме
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# URL страницы с результатами поиска
url = "https://yandex.ru/images/search?from=tabbar&text=мониторы%20в%20центре%20охраны"

# Переход на страницу
driver.get(url)
time.sleep(3)  # Дать время для загрузки страницы

# Прокручиваем страницу для загрузки изображений
for page in range(5):  # 35
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(4)

# Находим все элементы изображений
images = driver.find_elements(By.CSS_SELECTOR, "img")

# Скачиваем изображения
print(len(images))
for index, img in enumerate(images, start=1):  # Скачиваем первые 10 изображений
    img_url = img.get_attribute("src")
    print(index, img_url)
    if img_url and img_url.startswith("http"):  # Проверка, что ссылка на изображение корректная
        img_data = requests.get(img_url).content
        with open(os.path.join(save_path, f"none{388 + index}.jpg"), "wb") as handler:
            handler.write(img_data)
        print(f"Изображение {index} скачано.")

# Завершаем работу с браузером
driver.quit()
