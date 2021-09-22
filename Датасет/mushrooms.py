# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 16:38:28 2020

@author: stife
"""

#%% Загрузка

import os

PATH = os.path.join("datasets", "mushrooms")
NAME = "mushrooms.csv"

import pandas as pd

def load_data(path = PATH, name = NAME): # вернуть DataFrame из csv
    csv_path = os.path.join(path, name)
    return pd.read_csv(csv_path)

mushrooms = load_data()

#%% обзор данных в консоли

pd.options.display.max_columns = mushrooms.shape[1]
print(mushrooms.head())
mushrooms.info() # 23 категориальных признака 

#%% взглянем детальнее на каждый признак

print(mushrooms.describe())

for col in mushrooms: # 
    print(mushrooms[col].value_counts()) # обнаруживаем пропуски в stalk-root
    
# Нужно преобразовать каждый признак

#%% class {p, e} -> {0; 1}
    
print(mushrooms["class"].value_counts())
tr = {'e': 1, 'p': 0}
mushrooms["class"] = mushrooms["class"].map(tr)
print(mushrooms["class"][:10])

#%% cap-shape (каждое значение можно заменить числом, т.к. есть аналогия)

print(mushrooms["cap-shape"].value_counts())
tr = {'s': 0, 'f': 1, 'x': 2, 'k': 3, 'c': 3.3, 'b': 3.5}
mushrooms["cap-shape"] = mushrooms["cap-shape"].map(tr)
print(mushrooms["cap-shape"][:10])

#%% cap-surface

print(mushrooms["cap-surface"].value_counts())
tr = {'s': 0, 'f': 1, 'y': 2, 'g': 3}
mushrooms["cap-surface"] = mushrooms["cap-surface"].map(tr)
print(mushrooms["cap-surface"][:10])

#%% cap-color

print(mushrooms["cap-color"].value_counts())
tr_r = {'n': 165, 'b': 240, 'c': 210, 'g': 128, 'r': 0, 'p': 255, 'e': 255, 'w': 255, 'y': 255, 'u': 128}
tr_g = {'n': 42, 'b': 220, 'c': 105, 'g': 128, 'r': 128, 'p': 192, 'e': 0, 'w': 255, 'y': 255, 'u': 0}
tr_b = {'n': 42, 'b': 130, 'c': 40, 'g': 128, 'r': 0, 'p': 203, 'e': 0, 'w': 255, 'y': 0, 'u': 128}
mushrooms["cap-color-r"] = mushrooms["cap-color"].map(tr_r)
mushrooms["cap-color-g"] = mushrooms["cap-color"].map(tr_g)
mushrooms["cap-color-b"] = mushrooms["cap-color"].map(tr_b)
del mushrooms["cap-color"] # можно выкинуть
print(mushrooms["cap-color-r"][:10])
print(mushrooms["cap-color-g"][:10])
print(mushrooms["cap-color-b"][:10])
print(mushrooms.info())

#%% bruises

print(mushrooms["bruises"].value_counts())
tr = {'t': 1, 'f': 0}
mushrooms["bruises"] = mushrooms["bruises"].map(tr)
print(mushrooms["bruises"][:10])

#%% odor (запах). Этот признак нельзя соотнести с числом, поэтому закодируем его числом объектов у которых он есть

print(mushrooms["odor"].value_counts())
mushrooms["odor"] = mushrooms["odor"].map(mushrooms.groupby("odor").size())
print(mushrooms["odor"][:10])

#%% gill-attachment. Всё просто

print(mushrooms["gill-attachment"].value_counts())
tr = {'f': 0, 'a': 1}
mushrooms["gill-attachment"] = mushrooms["gill-attachment"].map(tr)
print(mushrooms["gill-attachment"][:10])

#%% gill-spacing. Всё просто

print(mushrooms["gill-spacing"].value_counts())
tr = {'c': 0, 'w': 1}
mushrooms["gill-spacing"] = mushrooms["gill-spacing"].map(tr)
print(mushrooms["gill-spacing"][:10])

#%% gill-color

print(mushrooms["gill-color"].value_counts())
tr_r.update({'k': 0, 'h': 123, 'o': 255})
tr_g.update({'k': 0, 'h': 63, 'o': 165})
tr_b.update({'k': 0, 'h': 0, 'o': 0})
mushrooms["gill-color-r"] = mushrooms["gill-color"].map(tr_r)
mushrooms["gill-color-g"] = mushrooms["gill-color"].map(tr_g)
mushrooms["gill-color-b"] = mushrooms["gill-color"].map(tr_b)
del mushrooms["gill-color"] # можно выкинуть
print(mushrooms["gill-color-r"][:10])
print(mushrooms["gill-color-r"][:10])
print(mushrooms["gill-color-r"][:10])

#%% gill-size

print(mushrooms["gill-size"].value_counts())
tr = {'n': 0, 'b': 1}
mushrooms["gill-size"] = mushrooms["gill-size"].map(tr)
print(mushrooms["gill-size"][:10])

#%% stalk-shape

print(mushrooms["stalk-shape"].value_counts())
tr = {'t': 0, 'e': 1}
mushrooms["stalk-shape"] = mushrooms["stalk-shape"].map(tr)
print(mushrooms["stalk-shape"][:10])

#%% stalk-root. Есть пропуски - применим one-hot-encoder

import numpy as np

print(mushrooms["stalk-root"].value_counts())
s = set(mushrooms["stalk-root"].unique())
s.remove('?')
for i in s:
    mushrooms["stalk-root" + '=' + i] = np.float32(mushrooms["stalk-root"] == i)
    print(mushrooms["stalk-root" + '=' + i][:10])
del mushrooms["stalk-root"]

#%% stalk-surface-above-ring

print(mushrooms["stalk-surface-above-ring"].value_counts())
tr = {'s': 0, 'k': 1, 'f': 2, 'y': 3}
mushrooms["stalk-surface-above-ring"] = mushrooms["stalk-surface-above-ring"].map(tr)
print(mushrooms["stalk-surface-above-ring"][:10])

#%% stalk-surface-below-ring

print(mushrooms["stalk-surface-below-ring"].value_counts())
tr = {'s': 0, 'k': 1, 'f': 2, 'y': 3}
mushrooms["stalk-surface-below-ring"] = mushrooms["stalk-surface-below-ring"].map(tr)
print(mushrooms["stalk-surface-below-ring"][:10])

#%% stalk-color-above-ring

print(mushrooms["stalk-color-above-ring"].value_counts())
mushrooms["stalk-color-above-ring-r"] = mushrooms["stalk-color-above-ring"].map(tr_r)
mushrooms["stalk-color-above-ring-g"] = mushrooms["stalk-color-above-ring"].map(tr_g)
mushrooms["stalk-color-above-ring-b"] = mushrooms["stalk-color-above-ring"].map(tr_b)
del mushrooms["stalk-color-above-ring"] # можно выкинуть
print(mushrooms["stalk-color-above-ring-r"][:10])
print(mushrooms["stalk-color-above-ring-g"][:10])
print(mushrooms["stalk-color-above-ring-b"][:10])

#%% stalk-color-below-ring

print(mushrooms["stalk-color-below-ring"].value_counts())
mushrooms["stalk-color-below-ring-r"] = mushrooms["stalk-color-below-ring"].map(tr_r)
mushrooms["stalk-color-below-ring-g"] = mushrooms["stalk-color-below-ring"].map(tr_g)
mushrooms["stalk-color-below-ring-b"] = mushrooms["stalk-color-below-ring"].map(tr_b)
del mushrooms["stalk-color-below-ring"] # можно выкинуть
print(mushrooms["stalk-color-below-ring-r"][:10])
print(mushrooms["stalk-color-below-ring-g"][:10])
print(mushrooms["stalk-color-below-ring-b"][:10])

#%% veil-type. Бесполезный признак

print(mushrooms["veil-type"].value_counts())
del mushrooms["veil-type"]

#%% veil-color

print(mushrooms["veil-color"].value_counts())
mushrooms["veil-color-r"] = mushrooms["veil-color"].map(tr_r)
mushrooms["veil-color-g"] = mushrooms["veil-color"].map(tr_g)
mushrooms["veil-color-b"] = mushrooms["veil-color"].map(tr_b)
del mushrooms["veil-color"] # можно выкинуть
print(mushrooms["veil-color-r"][:10])
print(mushrooms["veil-color-g"][:10])
print(mushrooms["veil-color-b"][:10])

#%% ring-number

print(mushrooms["ring-number"].value_counts())
tr = {'n': 0, 'o': 1, 't': 2}
mushrooms["ring-number"] = mushrooms["ring-number"].map(tr)
print(mushrooms["ring-number"][:10])

#%% ring-type. Закодируем числом экземпляров

print(mushrooms["ring-type"].value_counts())
mushrooms["ring-type"] = mushrooms["ring-type"].map(mushrooms.groupby("ring-type").size())
print(mushrooms["ring-type"][:10])

#%% spore-print-color

print(mushrooms["spore-print-color"].value_counts())
mushrooms["spore-print-color-r"] = mushrooms["spore-print-color"].map(tr_r)
mushrooms["spore-print-color-g"] = mushrooms["spore-print-color"].map(tr_g)
mushrooms["spore-print-color-b"] = mushrooms["spore-print-color"].map(tr_b)
del mushrooms["spore-print-color"] # можно выкинуть
print(mushrooms["spore-print-color-r"][:10])
print(mushrooms["spore-print-color-r"][:10])
print(mushrooms["spore-print-color-r"][:10])

#%% population

print(mushrooms["population"].value_counts())
tr = {'y': 1, 's': 2, 'v': 3, 'c': 4, 'a': 5, 'n': 6}
mushrooms["population"] = mushrooms["population"].map(tr)
print(mushrooms["population"][:10])

#%% habitat (one-hot-encoder)

print(mushrooms["habitat"].value_counts())

for i in mushrooms["habitat"].unique():
    mushrooms["habitat" + '=' + i] = np.float32(mushrooms["habitat"] == i)
    print(mushrooms["habitat" + '=' + i][:10])
    
del mushrooms["habitat"]

#%% результат проделанной работы

mushrooms.info()

#%% обзор данных в гистограммах
import matplotlib.pyplot as plt

mushrooms.hist(bins = 8, figsize = (20, 20))
plt.show()

#%% видим довольно много признаков, которые хорошо влияют на класс, хорошим показателем является цвет спор

corr_matrix = mushrooms.corr()

print(corr_matrix["class"].sort_values(ascending = False))

#%% scatter_matrix

from pandas.plotting import scatter_matrix

attributes = ["class", "ring-type", "spore-print-color-r", "spore-print-color-g"]
scatter_matrix(mushrooms[attributes], figsize = (15, 15))

# из графиков видно, например, что при определенных значениях некоторых параметров, можно однозначно предсказывать класс

#%% поэксперементируем с цветом аттрибут отвечающий за средний цвет

att_r = ["cap-color-r", "gill-color-r", "stalk-color-above-ring-r", "stalk-color-below-ring-r", "veil-color-r", "spore-print-color-r"]
att_g = ["cap-color-g", "gill-color-g", "stalk-color-above-ring-g", "stalk-color-below-ring-g", "veil-color-g", "spore-print-color-g"]
att_b = ["cap-color-b", "gill-color-b", "stalk-color-above-ring-b", "stalk-color-below-ring-b", "veil-color-b", "spore-print-color-b"]

mushrooms["r"] = 0
mushrooms["g"] = 0
mushrooms["b"] = 0
for i in range(6):
    mushrooms["r"] = mushrooms["r"] + mushrooms[att_r[i]]
    mushrooms["g"] = mushrooms["g"] + mushrooms[att_g[i]]
    mushrooms["b"] = mushrooms["b"] + mushrooms[att_b[i]]
    
mushrooms["r"] = mushrooms["r"] / 6
mushrooms["g"] = mushrooms["g"] / 6
mushrooms["b"] = mushrooms["b"] / 6

corr_matrix = mushrooms.corr()

print(corr_matrix["class"].sort_values(ascending = False))

#%% стало хуже поэтому удалим эти признаки

del mushrooms["r"]
del mushrooms["g"]
del mushrooms["b"]

#%% избавимся от "плохих" признаков

corr_matrix = mushrooms.corr()

for i in corr_matrix["class"].keys():
    if abs(corr_matrix["class"][i]) < 0.3:
        del mushrooms[i]

print(mushrooms.info()) # осталось 9 лучших признаков

#%% отделим тестовую выборку от тренировочной с помощью стратифицированной выборки по ring-type

from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
sss.get_n_splits(mushrooms)

for train_i, test_i in sss.split(mushrooms, mushrooms["ring-type"]):
    train_set = mushrooms.loc[train_i]
    test_set = mushrooms.loc[test_i]

train_set.info()
test_set.info()

print(train_set["ring-type"].value_counts())
print(test_set["ring-type"].value_counts())

#%% выполним масштабирование по минимаксу

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0, 1))
scaler.fit(train_set)
train_set = scaler.transform(train_set)

print(train_set)
