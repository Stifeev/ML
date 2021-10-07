# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 14:19:08 2021

@author: stife
"""

import numpy as np

import plotly.graph_objects as go
from plotly.offline import plot

import sympy as sp
from sympy import Symbol
from sympy.tensor.array import Array
import webbrowser

from matplotlib import pyplot as plt

import pygmo as pg

White = (255, 255, 255)
DeepPink = (255, 20, 147)
DarkBlue = (0, 0, 139)
Red = (255, 0, 0)

def normalize(color):
    return np.array(color, dtype=np.double) / 255

def skip_indicies(n, skip):
    return np.concatenate((np.arange(0, n-skip+1, skip, dtype=np.int32), [n - 1]))

#%% Тестовый пример 1 (Параболическая функция)

xs = [Symbol('x' + str(i)) for i in range(2)]
x, y = xs

f = 2 * x ** 2 + x * y + y ** 2 - 3 * x

x0 = np.array([-2., -2.])
lr = 1e0
tol = 1e-5

dot_skip = 1

gen = 10
n_pop = 10

x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)

print(f)

description = ", (x_1^{(0)}, x_2^{(0)}) = " + "({:.1f}, {:.1f}), ".format(*x0) +\
              "lr = " + "{:.1e}".format(lr)
title_text = "$ " + str(f) + description + " $"

#%% Тестовый пример 2 (Функция Розеброка)

xs = [Symbol('x' + str(i)) for i in range(2)]
x, y = xs
a = 100
f  = a * (y - x ** 2) ** 2 + (1 - x) ** 2

x0 = np.array([-2., -2.])
lr = 1e0
tol = 1e-5

dot_skip = 1000
gen = 15
n_pop = 120

x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)

print(f)

description = ", (x_1^{(0)}, x_2^{(0)}) = " + "({:.1f}, {:.1f}), ".format(*x0) +\
              "lr = " + "{:.1e}".format(lr)
title_text = "$ " + str(f) + description + " $"

#%% Тестовый пример 3 (функция Экли)

xs = [Symbol('x' + str(i)) for i in range(2)]
x, y = xs
f  = sp.E - \
     20 * sp.exp(-sp.sqrt((x ** 2 + y ** 2) / 50)) - \
     sp.exp((sp.cos(2 * sp.pi * x) + sp.cos(2 * sp.pi * y)) / 2)

x0 = np.array([-0.5, -1.6])
lr = 1e-1

skip_grad = 1
skip_pygmo = 1

gen = 10
n_pop = 30

x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)

print(f)

description = ", (x_1^{(0)}, x_2^{(0)}) = " + "({:.1f}, {:.1f}), ".format(*x0) +\
              "lr = " + "{:.1e}".format(lr)
title_text = "$ " + str(f) + description + " $"

#%%

def expr_eval(expr, x):
    return np.double(expr.subs([(xs[i], x[i]) for i in range(len(xs))]))

def grad_descent(f, x0,
                 lr=1e-2, lr_min=1e-8, 
                 tol=1e-8, 
                 method="rms-prop",
                 beta=0.9,
                 eps=1e-10,
                 callback=lambda n, xn, lr, norm: None):
    
    df = Array([sp.diff(f, _) for _ in xs]) # символьный градиент
    
    xn = np.copy(x0)
    x = [np.copy(xn)]
    t = 0 # номер текущей итерации
    norm = np.linalg.norm(expr_eval(df, xn))
    
    if method.lower() == "const-lr":
        """
        Спуск с постоянным шагом,
        уменьшение шага возможно только, если f(xn_next) >= f(xn)
        """
        while norm >= tol:
            if lr < lr_min: # достишли минимально возможного шага
                return np.array(x)
            
            xn_next = xn - lr * expr_eval(df, xn) # шаг спуска
            
            if expr_eval(f, xn_next) >= expr_eval(f, xn): # механизм уменьшения шага
                lr /= 2
                continue
            
            xn = np.copy(xn_next)
            norm = np.linalg.norm(expr_eval(df, xn))
            
            t += 1 # конец итерации
            callback(t, xn, lr, norm)
            x.append(np.copy(xn))
    
    elif method.lower() == "rmsprop":
        """
        Алгоритм RMSProp
        """
        s = np.zeros_like(xn)
        t = 1 # начинаем с единицы
        
        while norm >= tol:
            if lr < lr_min: # достишли минимально возможного шага
                break
            
            s_next = beta * s + (1 - beta) * expr_eval(df, xn) ** 2
            xn_next = xn - lr * expr_eval(df, xn) / np.sqrt(s_next + eps)  # шаг спуска
            
            if expr_eval(f, xn_next) >= expr_eval(f, xn): # механизм уменьшения шага
                lr /= 2
                continue
            
            s = s_next
            xn = xn_next
            norm = np.linalg.norm(expr_eval(df, xn))
            
            # конец итерации
            
            callback(t, xn, lr, norm)
            t += 1 
            x.append(np.copy(xn))
            
    return np.array(x)

#%% Применяем градиентный спуск

show_progress = True

callback = lambda n, xn, lr, norm: print("n = {:4d} || f = {:.4e} || lr = {:.2e} || norm = {:.2e}".format(n, expr_eval(f, xn), lr, norm)) if show_progress else lambda n, xn, lr, norm: None

grad_res = grad_descent(f, x0, lr, tol=tol,
                        method="rmsprop", beta=0.5, eps=1e-8,
                        callback = callback)
n_grad = grad_res.shape[0]
print("steps_count = {:d}, arg_min = {:s}, min = {:.4f}".format(n_grad - 1, str(grad_res[-1, :]), expr_eval(f, grad_res[-1, :])))

#%% Применяем Pygmo с covariance Matrix Evolutionary Strategy (CMA-ES)

class Ackley_function:
    
    def fitness(self, x):
        return [expr_eval(f, x)]
    
    def get_bounds(self):
        return ([x[0], y[0]], [x[-1], y[-1]])

prob = pg.problem(Ackley_function())

algo = pg.algorithm(pg.bee_colony(gen=gen, seed=42))
algo.set_verbosity(100)
pop = pg.population(prob, n_pop)
pop = algo.evolve(pop)

n_pygmo = pop.get_x().shape[0]
print(pop)

#%% Визуализация для функции двух аргументов через линии уровня

print("Градиентный Спуск: {:f}, Pygmo: {:f}".format(expr_eval(f, grad_res[-1, :]),
                                                    pop.champion_f[0]))

n_levels = 15 # число линий уровня
markersize = 5

indicies_grad = skip_indicies(n_grad, skip_grad)
indicies_pygmo = skip_indicies(n_pygmo, skip_pygmo)

z = np.zeros((len(y), len(x)), dtype=np.double)

for iy in range(len(y)):
    for ix in range(len(x)):
        z[iy, ix] = expr_eval(f, (x[ix], y[iy]))

fig, ax = plt.subplots(figsize=(14.40, 7.20), dpi=100)


ax.contourf(*np.meshgrid(x, y), z,
            levels=n_levels)

CS = ax.contour(*np.meshgrid(x, y), z,
                levels = n_levels,
                colors=[(0, 0, 0)] * n_levels)
ax.clabel(CS)

ax.scatter(grad_res[indicies_grad, 0], grad_res[indicies_grad, 1], color=normalize(Red), s=markersize, label="Итерация град. спуска", zorder=3)
ax.plot(grad_res[:, 0], grad_res[:, 1], color=normalize(White), label="Троаектория град. спуска")

ax.scatter(pop.get_x()[indicies_pygmo, 0], pop.get_x()[indicies_pygmo, 1], color=normalize(DarkBlue), s=markersize, label="Итерация эволюции", zorder=3)
ax.plot(pop.get_x()[indicies_pygmo, 0], pop.get_x()[indicies_pygmo, 1], color=normalize(DeepPink), label="Троаектория эволюции")

plt.legend()
plt.title(title_text)

#%% Визуализация для функции двух аргументов через 3D модель

print("Градиентный Спуск: {:f}, Pygmo: {:f}".format(expr_eval(f, grad_res[-1, :]),
                                                    pop.champion_f[0]))

marker_size = 4
line_width = 3
opacity = 0.6

indicies_grad = skip_indicies(n_grad, skip_grad)
indicies_pygmo = skip_indicies(n_pygmo, skip_pygmo)

z = np.zeros((len(y), len(x)), dtype=np.double)

for iy in range(len(y)):
    for ix in range(len(x)):
        z[iy, ix] = expr_eval(f, (x[ix], y[iy]))

fig = go.Figure([go.Surface(x = x,
                            y = y,
                            z = z,
                            opacity=opacity),
                 go.Scatter3d(x = grad_res[:, 0],
                              y = grad_res[:, 1],
                              z = [expr_eval(f, grad_res[i, :]) for i in range(n_grad)],
                              mode="lines",
                              line={'width': line_width, 'color': 'rgb'+str(White)},
                              name='Траектория град. спуска'),
                 go.Scatter3d(x = grad_res[indicies_grad, 0],
                              y = grad_res[indicies_grad, 1],
                              z = [expr_eval(f, grad_res[i, :]) for i in indicies_grad],
                              mode="markers",
                              marker={'size': marker_size, 'color': 'rgb'+str(Red)},
                              name="Итерация град. спуска"),
                 go.Scatter3d(x = pop.get_x()[:, 0],
                              y = pop.get_x()[:, 1],
                              z = pop.get_f()[:, 0],
                              mode="lines",
                              line={'width': line_width, 'color': 'rgb'+str(DeepPink)},
                              name='Траектория эволюции'),
                 go.Scatter3d(x = pop.get_x()[indicies_pygmo, 0],
                              y = pop.get_x()[indicies_pygmo, 1],
                              z = pop.get_f()[indicies_pygmo, 0],
                              mode="markers",
                              marker={'size': marker_size, 'color': 'rgb'+str(DarkBlue)},
                              name="Итерация эволюции")
                ])

fig.update_layout(title ={ 'text': title_text,  'font': { 'size': 15 } },
                  legend={ 'orientation': "h",
                           'yanchor': "bottom",
                           'y': 1.02,
                           'xanchor': "right",
                           'x': 1,
                           'bgcolor': "LightSteelBlue",
                           'bordercolor': "Black",
                           'borderwidth': 2 }
                 )
fig.show()

# Сохранение на диск
path = "grad_descent.html"

fig.update_layout(width=1920, height=1080)
plot(fig, auto_open=False, filename=path, include_mathjax="cdn")

webbrowser.open_new_tab(path)


#%%
