# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 14:19:08 2021

@author: stife
"""

import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot

#%%

def grad_descent(f, df, x0, lr=1e-2, eps=1e-7):
    xn = np.copy(x0)
    x = [np.copy(xn)]
    while np.linalg.norm(df(xn)) >= eps:
        xn -= lr * df(xn)
        x.append(np.copy(xn))
    return np.array(x)

#%% Тестовый пример 1

f  = lambda x: 2 * x[0] ** 2 + x[0] * x[1] + x[1] ** 2 - 3 * x[0]
df = lambda x: np.array([4 * x[0] + x[1] - 3, x[0] + 2 * x[1]])
x0 = np.array([20., 20.])
lr = 1e-1

f_text = "f(x_1, x_2) = 2 x_1^2 + x_1 x_ 2 + x_2^2 - 3x_1"
description = ", (x_1^{(0)}, x_2^{(0)}) = " + "({:.1f}, {:.1f}), ".format(*x0) +\
              "lr = " + "{:.1e}".format(lr)
title_text = "$ " + f_text + description + " $"

x = np.linspace(-20, 20, 50)
y = np.linspace(-20, 20, 50)

#%%

print(grad_descent(f, df, x0, lr)[-1, :])

#%% Тест 3D графиков

marker_size = 3
opacity = 0.6

z = np.zeros((len(y), len(x)), dtype=np.double)
grad_res = grad_descent(f, df, x0, lr)

for iy in range(len(y)):
    for ix in range(len(x)):
        z[iy, ix] = f((x[ix], y[iy]))


fig = go.Figure([go.Surface(x = x,
                            y = y,
                            z = z,
                            opacity=opacity), 
                 go.Scatter3d(x = grad_res[:, 0],
                              y = grad_res[:, 1],
                              z = [f(grad_res[m, :]) for m in range(grad_res.shape[0])],
                              mode="markers",
                              marker={'size': marker_size, 'color': 'red'})])
fig.update_layout(title={ 'text': title_text, 'font': { 'size': 15 } })
fig.show()
plot(fig, auto_open=False, filename="grad_descent.html", include_mathjax="cdn")
