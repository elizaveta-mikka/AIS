import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

#ЗАДАНИЕ 1 (Масштабирование признаков)
wholesale_customers = fetch_ucirepo(id=292)
X_features = ['Channel', 'Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']

X = wholesale_customers.data.features
Y = wholesale_customers.data.targets

X = X.values
X = X[:, 1:]
Y = Y.values.ravel()

scaler = StandardScaler()

X = scaler.fit_transform(X)

plt.scatter(X[:, 2], X[:, 4], color='black', s=10)
plt.title('Исходная выборка')
plt.show()

#ЗАДАНИЕ 2 (методы кластеризации)

#Метод -KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
# Оценка качества кластеризации
kmeans_score = normalized_mutual_info_score(Y, kmeans.labels_)

plt.scatter(X[:, 2], X[:, 4], c=kmeans.labels_, s=10)
plt.title('Алгоритм кластеризации - Kmeans')
plt.show()

#Метод - AffinityPropagatio
affinity_propagation = AffinityPropagation()
affinity_propagation.fit(X)
affinity_propagation_score = normalized_mutual_info_score(Y, affinity_propagation.labels_)

plt.scatter(X[:, 2], X[:, 4], c=affinity_propagation.labels_, s=10)
plt.title('Алгоритм кластеризации - AffinityPropagatio')
plt.show()

#Метод - AgglomerativeClustering
agglomerative = AgglomerativeClustering(n_clusters=4)
agglomerative.fit(X)
agglomerative_score = normalized_mutual_info_score(Y, agglomerative.labels_)

plt.scatter(X[:, 2], X[:, 4], c=agglomerative.labels_, s=10)
plt.title('Алгоритм кластеризации - AgglomerativeClustering')
plt.show()

print(f"Точность кластеризации:\n"
      f"K-Means - {kmeans_score}\n"
      f"AffinityPropagatio - {affinity_propagation_score}\n"
      f"AgglomerativeClustering - {agglomerative_score}\n")

#ЗАДАНИЕ 3 (Подбор наилучших параметров)

#Метод - KMeans

acc = np.zeros((8, 20))
for cl in range(8): # Количество кластеров
    for init in range(20): # Количество раз сколько алгоритм запустить с различными центроидами (выбирается лучший). Стандарт - 10
        kmeans = KMeans(n_clusters=cl+1, n_init=init+1)
        kmeans.fit(X)
        acc[cl][init] = normalized_mutual_info_score(Y, kmeans.labels_)

best_acc = np.max(acc)
best_index = np.argmax(acc)
best_cl, best_init = np.unravel_index(best_index, acc.shape)

print(f"KMeans\n"
      f"Наилучшая точность - {best_acc}\n"
      f"Наилучший параметр n_clusters - {best_cl+1}\n"
      f"Наилучший параметр n_init - {best_init+1}\n")

#Метод - AffinityPropagatio

par_damping = [0.5, 0.6, 0.7, 0.8, 0.9]
par_preference = np.linspace(-50, 50, 20)
acc = np.zeros((5, 20))
for dm in range(5): # Уровень сглаживания процесса обновления сообщений между точками
    for pr in range(20): # Влияет на количество кластеров (сколько точек "хотят быть" экземплярами)
        affinity_propagation = AffinityPropagation(damping=par_damping[dm], preference=par_preference[pr])
        affinity_propagation.fit(X)
        acc[dm][pr] = normalized_mutual_info_score(Y, affinity_propagation.labels_)

best_acc = np.max(acc)
best_index = np.argmax(acc)
best_dm, best_pr = np.unravel_index(best_index, acc.shape)

print(f"AffinityPropagatio\n"
      f"Наилучшая точность - {best_acc}\n"
      f"Наилучший параметр damping - {par_damping[best_dm]}\n"
      f"Наилучший параметр preference - {par_preference[best_pr]}\n")

#Метод - AgglomerativeClustering

par_linkage = ['ward', 'average', 'complete', 'single']
acc = np.zeros((8, 4))
for cl in range(8): # Количество кластеров
    for la in range(4): # Способ объединения кластеров
        agglomerative = AgglomerativeClustering(n_clusters=cl+1, linkage=par_linkage[la])
        agglomerative.fit(X)
        acc[cl][la] = normalized_mutual_info_score(Y, agglomerative.labels_)

best_acc = np.max(acc)
best_index = np.argmax(acc)
best_cl, best_la = np.unravel_index(best_index, acc.shape)

print(f"AgglomerativeClustering\n"
      f"Наилучшая точность - {best_acc}\n"
      f"Наилучший параметр n_clusters - {best_cl+1}\n"
      f"Наилучший параметр linkage - {par_linkage[best_la]}\n")