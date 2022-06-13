# -*- coding: utf-8 -*-

"""
import ml.Engine as ml
"""


import car
import pygame as pg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pli
from tensorflow import keras
import tensorflow as tf
"""

data = tf.keras.datasets.mnist.load_data(path="mnist.npz")

(train_image, train_label), (test_image, test_label) = data 
image_format = train_image.shape
N = train_image.size
n_target = test_label.shape[-1]

model = keras.Sequential()
model.add(keras.layers.Layer(input_shape=image_format))
model.add(keras.layers.Dense(2*N, activation=tf.sigmoid))
model.add(keras.layers.Dense(n_target))

model.train(train_image, train_label, solver=tf.AdamSolver())

N = 10

model = ml.Model("test")
model.add_layers(
    ml.Layer(5),
    ml.Layer(1)
)

model.W[0][0, 0] = 0.2


X = np.random.random(size=5*N).reshape(5, N)
Y = np.array([sum(X[i, j] for i in range(5)) for j in range(N)])
print(Y)
model.train(X, Y, ml.GD(learning_rate=0.1), epochs=10, debug=True)
"""

try :
    pg.init()
    screen_size = width, height = 500, 500
    window = pg.display.set_mode(screen_size, pg.RESIZABLE)
    pg.display.set_caption("SandBox")
    camera = car.Camera(ml.vec2(-5, -5), ml.vec2(20, 20), ml.vec2(*screen_size))
    route = car.Labyrinthe(5, 5, [-1, 0, 0, 1, 1, -1, 0, 1, 1, -1, 0, -1, -1])
    mobile = car.Car(scale=ml.vec2(2,1))
    launch = True
    t = pg.time.get_ticks()/1000
    while launch:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                launch = False
            
        t0 = t
        t = pg.time.get_ticks()/1000
        dt = t - t0
        
        mobile.update(dt)
            
        window.fill((255, 255, 255))
        camera.draw_route(route, window)
        camera.draw_car(mobile, window)
        pg.display.flip()

except Exception as e:
    pg.quit()
    raise e
pg.quit()
