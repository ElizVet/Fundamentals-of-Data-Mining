import sys
import pandas as pds 
import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp 
import IPython
import sklearn
import mglearn

from sklearn.datasets import load_iris
iris_dataset = load_iris()                          # набор данных Iris

print("Ключи iris_dataset: \n{}".format(iris_dataset.keys()))

print(iris_dataset['DESCR'][:193] + "\n--------------------")        # описание набора данных

print("\nНазвания сортов: {}".format(iris_dataset['target_names']))

print("\nНазвания признаков: \n{}".format(iris_dataset['feature_names']))

# target и data хранят количественные измерения характеристик цветков

# DATA: столбец = признак (4); строка = цвет (150)
print("\nТип массива data: {}".format(type(iris_dataset['data'])))
print("\nФорма массива data: {}".format(iris_dataset['data'].shape))
print("\nПервые пять строк массива data:\n{}".format(iris_dataset['data'][:5]))

# TARET: уже измеренные сорта.
# один цветок - один элемент 0-2.
# 0 – setosa, 1 – versicolor, 2 – virginica.
print("\nТип массива target: {}".format(type(iris_dataset['target'])))
print("\nФорма массива target: {}".format(iris_dataset['target'].shape))

print("\nОтветы:\n{}".format(iris_dataset['target']))

print("\n------\n")


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
iris_dataset['data'], iris_dataset['target'], random_state=0)

print("форма массива X_train: {}".format(X_train.shape))
print("форма массива y_train: {}".format(y_train.shape))

print("форма массива X_test: {}".format(X_test.shape))
print("форма массива y_test: {}".format(y_test.shape))

iris_dataframe = pds.DataFrame(X_train, columns=iris_dataset.feature_names) 

from pandas.plotting import scatter_matrix # парные диаграммы рассеяния для отображения более, чем двух признаков
grr = scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
plt.show()
