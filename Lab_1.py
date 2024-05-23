import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def popular_male_name(name):
    name2 = name.split(',')
    name3 = name2[1].split()
    new_name = name3[1]
    return new_name

#np.set_printoptions(precision=2, floatmode='fixed')
pd.set_option("display.precision", 2)
data = pd.read_csv('titanic_train.csv',
                  index_col='PassengerId')
#pd.set_option('display.max_columns', None)

#Задание_1
filter1_1 = data[data['Sex'] == 'female']
filter1_2 = data[data['Sex'] == 'male']
count1_1 = filter1_1.shape[0]
count1_2 = filter1_2.shape[0]
print(f"1. На борту было {count1_1} женщин и {count1_2} мужчин.")
#Задание_2
#Распредедление женщин по классам
filter2_1 = data[(data['Sex'] == 'female')&(data['Pclass'] == 1)]
filter2_2 = data[(data['Sex'] == 'female')&(data['Pclass'] == 2)]
filter2_3 = data[(data['Sex'] == 'female')&(data['Pclass'] == 3)]
Pclass_women = [filter2_1.shape[0], filter2_2.shape[0], filter2_3.shape[0]]
#Распредедление мужчин по классам
filter3_1 = data[(data['Sex'] == 'male')&(data['Pclass'] == 1)]
filter3_2 = data[(data['Sex'] == 'male')&(data['Pclass'] == 2)]
filter3_3 = data[(data['Sex'] == 'male')&(data['Pclass'] == 3)]
Pclass_men = [filter3_1.shape[0], filter3_2.shape[0], filter3_3.shape[0]]
print(f"2. На борту было {Pclass_men[1]} мужчин из второго класса.")
x = [1, 2, 3]
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.plot(x, Pclass_women, x, Pclass_men, linestyle = '-')
plt.title('Распределение пассажиров на борту по классам')
plt.xlabel('Класс')
plt.ylabel('Количество человек')
plt.legend(("Женщины", "Мужчины"))
plt.show()
#Задание_3
fare_median = round(data['Fare'].median(), 2)
fare_std = round(data['Fare'].std(), 2)
print(f"3. Медиана {fare_median}\n"
      f"   Cреднеквадратичное отклонение {fare_std}")
#Задание_4
survived = data[data['Survived'] == True]
died = data[data['Survived'] == False]
age_survived = survived['Age'].mean()
age_died = died['Age'].mean()
print(f"4. Средний возраст выживших - {round(age_survived, 2)}\n"
      f"   Средний возраст погибших - {round(age_died, 2)}")
#Задание_5
filter4_1 = data[(data['Survived'] == True)&(data['Age'] < 30)]
filter4_2 = data[(data['Survived'] == True)&(data['Age'] > 60)]
filter4_3 = data[data['Age'] < 30]
filter4_4 = data[data['Age'] > 60]
s_young = filter4_1.shape[0]
s_old = filter4_2.shape[0]
all_young = filter4_3.shape[0]
all_old = filter4_4.shape[0]
print(f"5. Доля выживших среди молодых - {round(s_young/all_young * 100, 2)}\n"
      f"   Доля выживших среди пожилых - {round(s_old/all_old * 100, 2)}")
#Задание_6
filter5_1 = data[(data['Survived'] == True)&(data['Sex'] == 'female')]
filter5_2 = data[(data['Survived'] == True)&(data['Sex'] == 'male')]
s_women = filter5_1.shape[0]
s_men = filter5_2.shape[0]
all_women = count1_1
all_men = count1_2
print(f"6. Доля выживших среди женщин - {round(s_women/all_women * 100, 2)}\n"
      f"   Доля выживших среди мужчин - {round(s_men/all_men * 100, 2)}")
#Задание_7
male_names = [popular_male_name(name) for name in filter1_2['Name'].to_numpy()]
names, k_names = np.unique(male_names, return_counts=True)
print(f"7. Самое популярное мужское имя на борту - {names[np.argmax(k_names)]}")
#Задание_8
class1_men = data[(data['Sex'] == 'male')&(data['Pclass'] == 1)]
class1_women = data[(data['Sex'] == 'female')&(data['Pclass'] == 1)]
class2_men = data[(data['Sex'] == 'male')&(data['Pclass'] == 2)]
class2_women = data[(data['Sex'] == 'female')&(data['Pclass'] == 2)]
class3_men = data[(data['Sex'] == 'male')&(data['Pclass'] == 3)]
class3_women = data[(data['Sex'] == 'female')&(data['Pclass'] == 3)]
people_class1 = data[data['Pclass'] == 1]
people_class2 = data[data['Pclass'] == 2]
people_class3 = data[data['Pclass'] == 3]
age_people1 = people_class1['Age'].mean()
age_people2 = people_class2['Age'].mean()
age_people3 = people_class3['Age'].mean()
mean_men1 = class1_men['Age'].mean()
mean_women1 = class1_women['Age'].mean()
mean_men2 = class2_men['Age'].mean()
mean_women2 = class2_women['Age'].mean()
mean_men3 = class3_men['Age'].mean()
mean_women3 = class3_women['Age'].mean()
men = "Мужчины"
women = "Женщины"
men2 = "мужчин"
women2 = "женщин"
print(f"8. Средний возраст мужчин в 1-ом классе - {round(mean_men1, 2)}\n"
      f"   Средний возраст женщин в 1-ом классе - {round(mean_women1, 2)}\n"
      f"   {men if mean_men1 > mean_women1 else women} 1-ого класса в среднем старше {women2 if mean_men1 > mean_women1 else men2} этого же класса.\n"
      f"   {men if mean_men2 > mean_women2 else women} 2-ого класса в среднем старше {women2 if mean_men2 > mean_women2 else men2} этого же класса.\n"
      f"   {men if mean_men3 > mean_women3 else women} 3-ого класса в среднем старше {women2 if mean_men3 > mean_women3 else men2} этого же класса.\n"
      f"   Средний возраст людей из 1-ом классе - {round(age_people1, 2)}, 2-ого - {round(age_people2, 2)}, 3-ого - {round(age_people3, 2)}")
