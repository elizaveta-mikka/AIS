import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


def less_mean(x, mean):
      if x < mean:
            return x


filename = 'function.csv'
filename2 = 'less_mean.csv'
x1 = np.linspace(-4 * math.pi, -math.pi, num=3000, dtype=float)
x2 = np.linspace(math.pi * 0.5, math.pi * 3.5, num=3000, dtype=float)
x1_3d, x2_3d = np.meshgrid(x1, x2)
y = np.tan(x1) * (1/np.tan(x2))

df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
df.to_csv(filename, index=False)

const_x1 = 0.5
const_x2 = 0.5
print("Функция имеет следующий вид: y = tan(x1) * ctg(x2)\n")
print(f"Константа x1: {const_x1}")
print(f"Константа x2: {const_x2}\n")

data1 = pd.read_csv('function.csv')
x = data1['x1']
ctg_x2 = 1 / np.tan(const_x2)
fun = np.tan(x) * ctg_x2
plt.xlabel('X1')
plt.ylabel('Y')
plt.scatter(x, fun, color="blue", s=1)
plt.show()

x = data1['x2']
tan_x1 = np.tan(const_x1)
fun = tan_x1 * (1/np.tan(x))
plt.xlabel('X2')
plt.ylabel('Y')
plt.scatter(x, fun, color="red", s=1)
plt.show()

col_x1 = data1['x1']
col_x2 = data1['x2']
col_y = data1['y']
mean_x1 = col_x1.mean()
mean_x2 = col_x2.mean()
print(f"Средние значения:\n"
      f"    Колонка 'x1' - {mean_x1}\n"
      f"    Колонка 'x2' - {mean_x2}\n"
      f"    Колонка 'y' - {col_y.mean()}\n")
print(f"Минимальные значения:\n"
      f"    Колонка 'x1' - {col_x1.min()}\n"
      f"    Колонка 'x2' - {col_x2.min()}\n"
      f"    Колонка 'y' - {col_y.min()}\n")
print(f"Максимальные значения:\n"
      f"    Колонка 'x1' - {col_x1.max()}\n"
      f"    Колонка 'x2' - {col_x2.max()}\n"
      f"    Колонка 'y' - {col_y.max()}\n")

col_mean1 = [less_mean(x1, mean_x1) for x1 in col_x1.to_numpy()]
col_mean2 = [less_mean(x2, mean_x2) for x2 in col_x2.to_numpy()]
col_mean1 = [x for x in col_mean1 if x is not None]
col_mean2 = [x for x in col_mean2 if x is not None]

df = pd.DataFrame({'x1<mean_x1': col_mean1, 'x2<mean_x2': col_mean2})
df.to_csv(filename2, index=False)

x = x1_3d
y = x2_3d
z = np.tan(x) * (1/np.tan(y))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='magma')
ax.set_zlim(-2000, 2000)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
plt.show()