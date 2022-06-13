# -*- coding: utf-8 -*-

"""
import ml.Engine as ml
"""


import car
import ml.Engine as ml
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
"""
model = keras.Sequential()
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(48, activation=tf.sigmoid))
model.add(keras.layers.Dense(1, activation=tf.sigmoid))


try :
    pg.init()
    screen_size = width, height = 500, 500
    window = pg.display.set_mode(screen_size, pg.RESIZABLE)
    pg.display.set_caption("TIPE")
    camera = car.Camera(ml.vec2(-5, -5), ml.vec2(50, 50), ml.vec2(*screen_size))
    route = car.Labyrinthe(5, 5, [-1, 0, 0, 1, 1, -1, 0, 1, 1, -1, 0, -1, -1])
    mobile = car.Car(pos = ml.vec2(2, 0), scale=ml.vec2(2,1))
    launch = True

    clock = pg.time.Clock()
    t = pg.time.get_ticks()/1000
    while launch:
        clock.tick(3000)
        print(clock.get_fps())
        for event in pg.event.get():
            if event.type == pg.QUIT:
                launch = False
            
        t0 = t
        t = pg.time.get_ticks()/1000
        dt = t - t0
        
        mobile.angle = np.sin(t)
        mobile.v = 1
        mobile.update(dt)
            
        window.fill((255, 255, 255))
        camera.pos = mobile.pos - camera.scale/2
        camera.draw_route(route, window)
        camera.draw_car(mobile, window)
        
        pg.display.flip()

except Exception as e:
    pg.quit()
    raise e
pg.quit()
