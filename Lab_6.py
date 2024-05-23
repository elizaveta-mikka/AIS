import warnings

import keras
import numpy as np
from PIL import Image
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import utils



# Обучающая и тестовая выборки
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

# Категории
class_names = ['Футболка или топ', 'Штаны', 'Свитер', 'Платье', 'Пальто', 'Сандалии', 'Рубашка', 'Кроссовки', 'Сумка', 'Ботинки']

# Нормализация данных (данные находятся в диапозоне от 0 до 1)
X_train = X_train / 255
X_test = X_test / 255

# Создание модели нейронной сети
model = keras.Sequential([ # Последовательные слои
    keras.layers.Flatten(input_shape=(28, 28)), #Преобразоваие формата изображения
    keras.layers.Dense(128, activation='relu'), #Входной слой (поступает 28*28 от каждого изображения)
    keras.layers.Dense(10, activation='softmax') #Выходной слой
])

# Компиляция модели
model.compile(optimizer=keras.optimizers.SGD(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Вывод параметров модели
#model.summary()

# Обучение модели
model.fit(X_train, Y_train, epochs=30) # 10 эпох

# Проверка точности предсказаний на тестовой выборке
test_loss, test_acc = model.evaluate(X_test, Y_test)
print(f'Предсказания по тестовой выборке:\n'
      f'Потери: {test_loss}\n'
      f'Точность: {test_acc}\n')

def my_model(choice, address):
    # Заданное предсказание
    if choice == 1:
        address = int(address)
        prediction = model.predict(X_train)
        index = np.argmax(prediction[address])
        return X_train[address], class_names[index]
    if choice == 2:
        img = Image.open(address).resize((28, 28)).convert('L')  # Преобразование в подходящий размер и серый формат
        if img.getpixel((0, 0)) > 10:
            for x in range(28):
                for y in range(28):
                    brightness = 255 - img.getpixel((x, y))
                    img.putpixel((x, y), brightness)
        img_array = np.array(img) / 255  # Нормализация значений пикселей
        img_input = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_input)
        index = np.argmax(prediction)
        return img, class_names[index]


