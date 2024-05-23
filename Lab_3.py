import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

#ЗАДАНИЕ 1

data = pd.read_excel('data_akbilgic.xlsx')

#Признаки
X = data[['SP', 'DAX', 'FTSE', 'NIKKEI', 'BOVESPA', 'EU', 'EM']]
#Целевой значение
y = data['TL ISE']

split_index = int(0.8 * len(X))
X_train_set, X_test_set = X[:split_index], X[split_index:]
Y_train_set, Y_test_set = y[:split_index], y[split_index:]

#ЗАДАНИЕ 2-3

#Выбор одного признака
X_train2 = X_train_set['FTSE'].to_frame()
X_test2 = X_test_set['FTSE'].to_frame()

#Целевой значение
Y_train2 = Y_train_set
Y_test2 = Y_test_set

#Создание модели линейной регрессии для предсказания целевого значения
model = LinearRegression()

#Обучение модели
model.fit(X_train2, Y_train2)

#Предсказнные значения
Y_pred = model.predict(X_test2)

#print("Коэффициенты: \n", model.coef_)
print("Среднеквадратичное отклонение: %.5f" % mean_squared_error(Y_test2, Y_pred))
print("Коэффициент детерминации: %.5f\n" % r2_score(Y_test2, Y_pred))

#Фактические значения
plt.scatter(X_test2, Y_test2, color="black")
plt.plot(X_test2, Y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

#ЗАДАНИЕ 4

#Степени
n_d = 15

#Коэффиценты детерминации
r2_train = [0] * n_d
r2_test = [0] * n_d

fig, axs = plt.subplots(1, 2)
fig.suptitle("Зависимость коэффициента детерминации от степени полиномиальной функции")
for i in range(15):
    polynomial_features = PolynomialFeatures(degree=i+1, include_bias=False)
    linear_regression = LinearRegression()

    #Контейер обработки данных
    pipeline = Pipeline(
        [
            ("polynomial_features", polynomial_features),
            ("linear_regression", linear_regression),
        ]
    )

    #Обучение модели в контейнере
    pipeline.fit(X_train2, Y_train2)

    #Предсказнные значения
    Y_train_pred = pipeline.predict(X_train2)
    Y_test_pred = pipeline.predict(X_test2)

    #Коэффицент детерминации (для обучающей и тестовой выборок)
    r2_train[i] = r2_score(Y_train2, Y_train_pred)
    r2_test[i] = r2_score(Y_test2, Y_test_pred)

best_r2 = max(r2_test) #Наиболее удачный коэффицент детерминации
best_degree = r2_test.index(best_r2) #Наиболее удачная степень полиномов
print('Высшую точность имеет степень - %d' %(best_degree + 1))
print('Лучший коэффицент детерминации:')
print('для обучающей выборки - %5f' % r2_train[best_degree])
print('для тестовой выборки - %5f\n' % r2_test[best_degree])

axs[0].plot(np.linspace(1, n_d, n_d), r2_train)
axs[0].set_title('Зависимость на обучающей выборки')

axs[1].plot(np.linspace(1, n_d, n_d), r2_test)
axs[1].set_title('на тестовой выборки')

fig.tight_layout()
plt.show()

#ЗАДАНИЕ 5

degree = best_degree
r2_train_alpha = [0] * 1000
r2_test_alpha = [0] * 1000

fig, axs = plt.subplots(1, 2)
fig.suptitle("Зависимость коэффициента детерминации от коэффициента регуляризации")
for i, alpha in enumerate(np.linspace(0, 1, 1000)):
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_regression = linear_model.Ridge(alpha=alpha)

    pipeline = Pipeline(
        [
            ("polynomial_features", polynomial_features),
            ("linear_regression", linear_regression),
        ]
    )

    pipeline.fit(X_train2, Y_train2)

    Y_train_pred = pipeline.predict(X_train2)
    Y_test_pred = pipeline.predict(X_test2)

    r2_train_alpha[i] = r2_score(Y_train2, Y_train_pred)
    r2_test_alpha[i] = r2_score(Y_test2, Y_test_pred)

best_r2 = max(r2_test_alpha)
best_alpha = r2_test_alpha.index(best_r2)
print('Лучший коэффицент регуляризации: %.5f' % best_alpha)
print('Лучший коэффицент детерминации при лучшем коэффициенте регуляризации: %.5f' % best_r2)
axs[0].plot(np.linspace(0, 1, 1000), r2_train_alpha)
axs[0].set_title('Зависимость на обучающей выборки')
axs[1].plot(np.linspace(0, 1, 1000), r2_test_alpha)
axs[1].set_title('на тестовой выборки')

fig.tight_layout()
plt.show()
