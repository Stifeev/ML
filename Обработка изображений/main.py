# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 13:56:39 2021

@author: stife
"""

image_name = "2B.png"

#%% Открытие изображения

from PIL import Image

im = Image.open(image_name)

from matplotlib import pyplot as plt

def imshow(im, title="", coord=False, size=(720, 480), dpi=100, gray=False):
    plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
    plt.title(title)
    if not coord:
        plt.axis('off')
    plt.imshow(im, cmap="gray" if gray else None)

imshow(im, "2B")

#%% В оттенках серого

imshow(im.convert("L"), gray=True)

#%% Сохраняем в jpeg

from os import path

name, ext = path.splitext(image_name)

try:
    im.convert("RGB").save(name + ".jpeg")
except  IOError:
    print("Не могу преобразовать в jpeg")

#%% Создаём миниатюру

im_small = im.copy()
im_small.thumbnail((128, 128))

imshow(im_small)

#%% Копирование и вставка областей

imshow(im, "2B", True)

box = (500, 0, 1250, 580)
region = im.crop(box)

imshow(region, "Вырезанная область", False)

region = region.transpose(Image.ROTATE_180)
imshow(region, "Поворачиваем область", False)

new_im = im.copy()
new_im.paste(region, box)
imshow(new_im, "Результат вставки")

#%% Интерполяции и поворот

imshow(im.resize((640, 400)), "2B in 640x400", size=(640, 400))

imshow(im.rotate(45), "2B in rotate 45")

#%% Рисуем точки и прямые линии

import numpy as np

im = np.array(im)

def figure(title="", coord=False, size=(720, 480), dpi=100):
    plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
    plt.title(title)
    if not coord:
        plt.axis('off')
        
figure("Рисуем точки и отрезки", True)

x = [100, 100, 400, 400]
y = [200, 500, 200, 500]

plt.plot(x, y, 'r*')
plt.plot(x[:2], y[:2])

plt.imshow(im)

#%% Изолинии и гистограммы

im = np.array(Image.open(image_name).convert("L"))

figure("Контур", False)

plt.contour(im, origin='image')

figure("Гистограмма", True)
plt.hist(im.flatten(order='C'), 128)

#%% Преобразуем изображение

figure("Инвертированное изображение")
plt.imshow(255 - im)

figure("Приведённое к диапазону [100, 200]")
plt.imshow(100 / 255 * im + 100)

figure("Квадратичная функция")
plt.imshow(255 * (im / 255) ** 2)

#%% Выравнивание гистограммы

def histeq(im, n_bins=256):
    
    imhist, bins = np.histogram(im.flatten(), n_bins, density=True)
    cdf = imhist.cumsum()     # Функция распределения
    cdf = 255 * cdf / cdf[-1] # Нормировать
    
    im_ = np.interp(im.flatten(), bins[:-1], cdf)
    
    return im_.reshape(im.shape), cdf

figure("Изображение до выравнивания")
plt.gray()
plt.imshow(im)

figure("Гистограмма до выравнивания", True)
plt.hist(im.flatten(), 128)

im_eq, cdf = histeq(im)

figure("Преобразование", True)
plt.grid(True)
plt.plot(range(256), cdf)

figure("Изображение после выравнивания")
plt.gray()
plt.imshow(im_eq)
figure("Гистограмма после выравнивания", True)
plt.hist(im_eq.flatten(), 128)

#%% Среднее изображение

import os

def im_mean(path_to_im, size=(720, 480)):
    
    image_pathes = [os.path.join(path_to_im, _) for _ in  os.listdir(path_to_im)]
    
    im_avg = np.zeros((size[1], size[0], 3), dtype=np.float64)
    for im_path in image_pathes:
        im = Image.open(im_path).convert("RGB")
        im = np.array(im.resize(size))
        
        im_avg += im
    
    im_avg = im_avg / len(image_pathes) if len(image_pathes) > 0 else im_avg
    im_avg = np.uint8(np.round(im_avg))
    
    return im_avg

path_to_im = "images"

im_avg = im_mean(path_to_im)

figure("Среднее изображение")
plt.imshow(im_avg)

#%% Размытие изображения

from scipy.ndimage import filters

image_name = "2B.png"
sigma = [0, 2, 5, 10]

im = np.array(Image.open(image_name))

images = np.zeros((len(sigma), im.shape[0], im.shape[1], im.shape[2]), dtype=np.uint8)

for i in range(len(sigma)):
    for j in range(im.shape[-1]):
        images[i, :, :, j] = np.uint8(filters.gaussian_filter(im[:,:,j], sigma[i]))
        
    figure("$ \sigma = {:.2f} $".format(sigma[i]))
    plt.imshow(images[i, :, :, :])

#%% Вычисление производных и контуров по Собелю

im_gray = np.array(Image.open(image_name).convert("L"))

figure("Исходное изображение")
plt.imshow(im)

#======================================================================

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.20, 4.80), dpi=100)
plt.title("Производная по x")

imx = np.zeros(im_gray.shape)
filters.sobel(im_gray, 1, imx)
plt.set_cmap("viridis")
cax = ax.matshow(imx, interpolation ='nearest', vmin=-255, vmax=255)
fig.colorbar(cax, ax=ax)
plt.axis('off')

#======================================================================

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.20, 4.80), dpi=100)
plt.title("Производная по y")

imy = np.zeros(im_gray.shape)
filters.sobel(im_gray, 0, imy)
plt.set_cmap("viridis")
cax = ax.matshow(imy, interpolation ='nearest', vmin=-255, vmax=255)
fig.colorbar(cax, ax=ax)
plt.axis('off')

#======================================================================

figure("Градиент")

grad = np.sqrt(imx ** 2 + imy ** 2)
im_norm = np.minimum(255, grad)
plt.imshow(im_norm, cmap="gray")

#%% Очищаем изображение от шума

import sys

def denoise(im, U_unit, tol=1e-1, tau=0.125, tv_weight=100):
    m, n = im.shape
    
    U = U_unit
    Px = im
    Py = im
    error = 1.
    
    while error > tol:
        Uold = U
        
        GradUx = np.roll(U, -1, axis=1)-U # x-компонента по конечным разностям
        GradUy = np.roll(U, -1, axis=0)-U # y-компонента по конечным разностям
        
        PxNew = Px + (tau / tv_weight) * GradUx
        PyNew = Py + (tau / tv_weight) * GradUy
        NormNew = np.maximum(1, np.sqrt(PxNew ** 2 + PyNew ** 2))
        Px = PxNew / NormNew
        Py = PyNew / NormNew
        
        RxPx = np.roll(Px, 1, axis=1)
        RyPy = np.roll(Py, 1, axis=0)
        
        DivP = (Px - RxPx) + (Py - RyPy)
        U = im + tv_weight * DivP
        
        error = np.linalg.norm(U - Uold) / np.sqrt(n * m)
        
        sys.stdout.write("\rgoal = {:e}, error = {:e}".format(tol, error))
    
    sys.stdout.write("\n")
    return U, im - U

image_name = "2B.png"
sigma = 30 # сила замыления

im = np.array(Image.open(image_name))

figure("Исходное изображение")
plt.imshow(im)

im_noise = np.maximum(np.minimum(np.float64(im) + sigma * np.random.standard_normal(im.shape), 
                                 255), 
                      0)

figure("Зашумлённое изображение. $ \sigma = {:.1f} $".format(sigma))
plt.imshow(np.uint8(np.round(im_noise)))

channels = im.shape[-1]
tol = 5e-2

im_clear = np.zeros(im.shape, dtype=np.uint8)
for i in range(channels):
    im_clear[:,:,i] = denoise(im_noise[:,:,i], im_noise[:,:,i], tol=tol)[0]
    
figure("Очищенное изображение")
plt.imshow(im_clear)
