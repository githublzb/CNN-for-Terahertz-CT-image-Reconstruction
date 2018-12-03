# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 10:26:09 2018

@author: ShawnYe
"""

import numpy as np
import h5py
import scipy.io as sio
from tensorflow import keras

class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, data_dir, batch_size = 32, x_dim=(512,512,1), y_dim=(512,512,1), shuffle=True):
        self.list_IDs = list_IDs
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.floor(len(self.list_IDs)/self.batch_size))
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_IDs = [self.list_IDs[k] for k in batch_indexes]
        
        X, Y = self.__batch_data_generation(batch_IDs)
        
        return X,Y
    
    def __batch_data_generation(self, batch_IDs):
        X = np.empty((self.batch_size, *self.x_dim)) #* is used to unpack the self.dim tuple
        Y = np.empty((self.batch_size, *self.y_dim))
        
        for i, ID in enumerate(batch_IDs):
            X[i,:], Y[i,:] = self.load_mat(ID, 'LRecon1', 'HRecon')
            
        return X,Y
    
    def load_mat(self, ID, x_name, y_name):
        file_name = self.data_dir + str(ID) + '.mat'
        try:
            with h5py.File(file_name, 'r') as f:
                x = np.reshape(f[x_name], self.x_dim)
                y = np.reshape(f[y_name], self.y_dim)
                return x,y
        except OSError:
            f = sio.loadmat(file_name)
            x = np.reshape(f[x_name], self.x_dim)
            y = np.reshape(f[y_name], self.y_dim)
            return x,y

if __name__ == '__main__':
    generator = DataGenerator(list(np.arange(1, 501)), 'Data/TrainingData/')
    generator.load_mat(5, 'LRecon1', 'HRecon')