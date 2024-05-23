from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo

#ЗАДАНИЕ 1 (Разделение на обучающую и тестовую выборки)

phishing_websites = fetch_ucirepo(id=327)

X = phishing_websites.data.features
Y = phishing_websites.data.targets

split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
Y_train, Y_test = Y[:split_index], Y[split_index:]
Y_train = Y_train.values.ravel()
Y_test = Y_test.values.ravel()

#ЗАДАНИЕ 2 (Масштабирвоание признаков)

# Создаем объект StandardScaler
# Стандартизация признаков путем удалениия ср. значения
# Масштабирование до единичной дисперсии
scaler = StandardScaler()

# Вычисление среднего и стандартного отклонений, стандартизация с их использованием
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#ЗАДАНИЕ 3-4 (Обучение нейроных сетей и оценка точности)

#Персептрон

# Создание и обучение персептрона
perceptron = Perceptron(max_iter=700)
perceptron.fit(X_train, Y_train)

# Предсказание и оценка точности
Y_pred = perceptron.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(f'Точность перцептрона: {accuracy:.10f}')

#MLP

# Создание и обучение MLP классификатора
mlp = MLPClassifier(max_iter=700)
mlp.fit(X_train, Y_train)

# Предсказание и оценка точности
Y_pred = mlp.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(f'Точность MLP классификатора: {accuracy:.10f}')

#ЗАДАНИЕ 5 (Наилучшие параметры)

# Коэффициент обучения
learning_rates = [0.00001,  0.00005, 0.0001,  0.0005, 0.001,  0.005, 0.01, 0.05, 0.1]
# Параметр регуляризации
alpha = [0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
# Функции оптимизации
fun_opt = ['lbfgs', 'sgd', 'adam']

lr_mpl = []
lr_pr = []

a_mpl = []
a_pr = []

opt_mpl = []

for lr in learning_rates:
    mlp = MLPClassifier(max_iter=700, learning_rate_init=lr)
    perceptron = Perceptron(max_iter=700, eta0=lr)
    mlp.fit(X_train_scaled, Y_train)
    perceptron.fit(X_train_scaled, Y_train)
    lr_mpl.append(accuracy_score(Y_test, mlp.predict(X_test_scaled)))
    lr_pr.append(accuracy_score(Y_test, perceptron.predict(X_test_scaled)))
best_mpl = lr_mpl.index(max(lr_mpl))
best_pr = lr_pr.index(max(lr_pr))
print(f'Наилучшие коэффициенты обучения:\n'
      f'Перцептрон - {learning_rates[best_pr]}\n'
      f'MLP - {learning_rates[best_mpl]}\n')

for a in alpha:
    mlp = MLPClassifier(max_iter=700, alpha=a)
    perceptron = Perceptron(max_iter=700, alpha=a)
    mlp.fit(X_train_scaled, Y_train)
    perceptron.fit(X_train_scaled, Y_train)
    a_mpl.append(accuracy_score(Y_test, mlp.predict(X_test_scaled)))
    a_pr.append(accuracy_score(Y_test, perceptron.predict(X_test_scaled)))
best_mpl = a_mpl.index(max(a_mpl))
best_pr = a_pr.index(max(a_pr))
print(f'Наилучшие параметры регуляризации:\n'
      f'Перцептрон - {alpha[best_pr]}\n'
      f'MLP - {alpha[best_mpl]}\n')

for opt in fun_opt:
    mlp = MLPClassifier(max_iter=700, solver=opt)
    mlp.fit(X_train_scaled, Y_train)
    opt_mpl.append(accuracy_score(Y_test, mlp.predict(X_test_scaled)))
best_mpl = opt_mpl.index(max(opt_mpl))
print(f'Наилучшая функция оптимизации:\n'
      f'MLP - {fun_opt[best_mpl]}')

plt.plot(learning_rates, lr_pr, label='Перцептрон')
plt.plot(learning_rates, lr_mpl, label='MPL')
plt.xlabel('Коэффициент обучения')
plt.ylabel('Точность')
plt.title('Зависимость точности от коэффициента обучения')
plt.legend()
plt.show()

plt.plot(alpha, a_pr, label='Перцептрон')
plt.plot(alpha, a_mpl, label='MPL')
plt.xlabel('Параметр регуляризации')
plt.ylabel('Точность')
plt.title('Зависимость точности от параметра регуляризации')
plt.legend()
plt.show()

plt.plot(fun_opt, opt_mpl, label='MPL')
plt.xlabel('Функция оптимизации')
plt.ylabel('Точность')
plt.title('Зависимость точности от функции отимизации')
plt.legend()
plt.show()