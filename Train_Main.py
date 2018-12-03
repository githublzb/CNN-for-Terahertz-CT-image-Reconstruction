# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from LoadData import DataGenerator
from Unet import UNet
import numpy as np
from tensorflow import keras

train_IDs = list(np.arange(1, 501))
val_IDs = list(np.arange(1, 51))
train_generator = DataGenerator(train_IDs, 'Data/TrainingData/', batch_size = 2)
val_generator = DataGenerator(val_IDs, 'Data/ValidationData/', batch_size = 2)

tbCallBack = keras.callbacks.TensorBoard(log_dir = 'TestTraining/logs', write_graph = True, write_images = True) #Tensorboard callback
checkpoint_path = 'TestTraining/checkpoints/test_cp.ckpt'
cpCallBack = keras.callbacks.ModelCheckpoint(checkpoint_path) #checkpoints callBack

model = UNet((512,512,1))
model.summary()
model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', metrics = ['accuracy'])
model.fit_generator(train_generator, epochs = 10, validation_data = val_generator,
                    workers = 6, callbacks = [tbCallBack, cpCallBack])

