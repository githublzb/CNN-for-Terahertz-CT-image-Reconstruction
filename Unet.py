# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 10:38:56 2018

@author: ShawnYe
"""

import numpy as np
from tensorflow import keras

def UNet(input_size, pretrained_weights = None):
    inputs = keras.Input(shape = input_size)
    
    #level 1
    conv64_1 = keras.layers.Conv2D(64, [3,3], [1,1], 'same', activation = 'relu')(inputs)
    conv64_1_BN = keras.layers.BatchNormalization()(conv64_1)
    conv64_2 = keras.layers.Conv2D(64, [3,3], [1,1], 'same', activation = 'relu')(conv64_1_BN)
    conv64_2_BN = keras.layers.BatchNormalization()(conv64_2)
    conv64_3 = keras.layers.Conv2D(64, [3,3], [1,1], 'same', activation = 'relu')(conv64_2_BN)
    conv64_3_BN = keras.layers.BatchNormalization()(conv64_3)
    
    #downsampling to level 2
    conv64_3_pool = keras.layers.MaxPool2D([2,2], padding = 'valid')(conv64_3_BN)
    conv128_1 = keras.layers.Conv2D(128, [3,3], [1,1], 'same', activation = 'relu')(conv64_3_pool)
    conv128_1_BN = keras.layers.BatchNormalization()(conv128_1)
    conv128_2 = keras.layers.Conv2D(128, [3,3], [1,1], 'same', activation = 'relu')(conv128_1_BN)
    conv128_2_BN = keras.layers.BatchNormalization()(conv128_2)
    
    #downsampling to level 3
    conv128_2_pool = keras.layers.MaxPool2D([2,2], padding = 'valid')(conv128_2_BN)
    conv256_1 = keras.layers.Conv2D(256, [3,3], [1,1], 'same', activation = 'relu')(conv128_2_pool)
    conv256_1_BN = keras.layers.BatchNormalization()(conv256_1)
    conv256_2 = keras.layers.Conv2D(256, [3,3], [1,1], 'same', activation = 'relu')(conv256_1_BN)
    conv256_2_BN = keras.layers.BatchNormalization()(conv256_2)
    
    #downsampling to level 4
    conv256_2_pool = keras.layers.MaxPool2D([2,2], padding = 'valid')(conv256_2_BN)
    conv512_1 = keras.layers.Conv2D(512, [3,3], [1,1], 'same', activation = 'relu')(conv256_2_pool)
    conv512_1_BN = keras.layers.BatchNormalization()(conv512_1)
    conv512_2 = keras.layers.Conv2D(512, [3,3], [1,1], 'same', activation = 'relu')(conv512_1_BN)
    conv512_2_BN = keras.layers.BatchNormalization()(conv512_2)
    
    #downsampling to level 5
    conv512_2_pool = keras.layers.MaxPool2D([2,2], padding = 'valid')(conv512_2_BN)
    conv1024_1 = keras.layers.Conv2D(1024, [3,3], [1,1], 'same', activation = 'relu')(conv512_2_pool)
    conv1024_1_BN = keras.layers.BatchNormalization()(conv1024_1)
    conv1024_2 = keras.layers.Conv2D(1024, [3,3], [1,1], 'same', activation = 'relu')(conv1024_1_BN)
    conv1024_2_BN = keras.layers.BatchNormalization()(conv1024_2)
    
    #upsampling to level 4
    level4_ini = keras.layers.Conv2D(512, [3,3], [1,1], 'same', activation='relu')(keras.layers.UpSampling2D((2,2))(conv1024_2_BN))
    level4_ini_concat = keras.layers.concatenate([conv512_2_BN, level4_ini], axis = -1)
    conv512_1_up = keras.layers.Conv2D(512, [3,3], [1,1], 'same', activation = 'relu')(level4_ini_concat)
    conv512_1_BN_up = keras.layers.BatchNormalization()(conv512_1_up)
    conv512_2_up = keras.layers.Conv2D(512, [3,3], [1,1], 'same', activation = 'relu')(conv512_1_BN_up)
    conv512_2_BN_up = keras.layers.BatchNormalization()(conv512_2_up)
    
    #upsampling to level 3
    level3_ini = keras.layers.Conv2D(256, [3,3], [1,1], 'same', activation='relu')(keras.layers.UpSampling2D((2,2))(conv512_2_BN_up))
    level3_ini_concat = keras.layers.concatenate([conv256_2_BN, level3_ini], axis = -1)
    conv256_1_up = keras.layers.Conv2D(256, [3,3], [1,1], 'same', activation = 'relu')(level3_ini_concat)
    conv256_1_BN_up = keras.layers.BatchNormalization()(conv256_1_up)
    conv256_2_up = keras.layers.Conv2D(256, [3,3], [1,1], 'same', activation = 'relu')(conv256_1_BN_up)
    conv256_2_BN_up = keras.layers.BatchNormalization()(conv256_2_up)
    
    #upsampling to level 2
    level2_ini = keras.layers.Conv2D(128, [3,3], [1,1], 'same', activation='relu')(keras.layers.UpSampling2D((2,2))(conv256_2_BN_up))
    level2_ini_concat = keras.layers.concatenate([conv128_2_BN, level2_ini], axis = -1)
    conv128_1_up = keras.layers.Conv2D(128, [3,3], [1,1], 'same', activation = 'relu')(level2_ini_concat)
    conv128_1_BN_up = keras.layers.BatchNormalization()(conv128_1_up)
    conv128_2_up = keras.layers.Conv2D(128, [3,3], [1,1], 'same', activation = 'relu')(conv128_1_BN_up)
    conv128_2_BN_up = keras.layers.BatchNormalization()(conv128_2_up)
    
    #upsampling to level 1
    level1_ini = keras.layers.Conv2D(64, [3,3], [1,1], 'same', activation='relu')(keras.layers.UpSampling2D((2,2))(conv128_2_BN_up))
    level1_ini_concat = keras.layers.concatenate([conv64_3_BN, level1_ini], axis = -1)
    conv64_1_up = keras.layers.Conv2D(64, [3,3], [1,1], 'same', activation = 'relu')(level1_ini_concat)
    conv64_1_BN_up = keras.layers.BatchNormalization()(conv64_1_up)
    conv64_2_up = keras.layers.Conv2D(64, [3,3], [1,1], 'same', activation = 'relu')(conv64_1_BN_up)
    conv64_2_BN_up = keras.layers.BatchNormalization()(conv64_2_up)
    conv1_up = keras.layers.Conv2D(1, [1,1], [1,1], 'same')(conv64_2_BN_up)
    outputs = keras.layers.add([inputs, conv1_up])
    
    
    ###################################################################################
    model = keras.models.Model(inputs = inputs, outputs = outputs)
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    
    return model

if __name__ == '__main__':
    model = UNet((512,512,1))
    model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
    x = np.random.random((1,512,512,1))
    final_out = model.predict(x)
    
    
