import os
from Lab_6 import my_model
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

while 1:
    choice = 0
    print(f'Изображение для обработки\n'
          f'Взять из fashion_mnist - 1\n'
          f'Взять собственное - 2\n')
    try:
        choice = int(input("Выбрать - "))
    except ValueError:
        print("Значение введено некорректно\n")
    if choice == 1:
        print(f'Индекс изображения из fashion_mnist, которое берется для обработки - ')
        address = str(input())
        image, category = my_model(choice, address)
        print(f'Категория - {category}\n')
        plt.imshow(image)
        plt.show()
    elif choice == 2:
        print(f'Имя с расширением изображения, которое берется для обработки - ')
        address = str(input())
        if os.path.exists(address):
            my_model(choice, address)
            image, category = my_model(choice, address)
            print(f'Категория - {category}\n')
            plt.imshow(image)
            plt.show()
        else:
            print(f"Файл с таким именем не найден\n")

