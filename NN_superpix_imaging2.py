# -*- coding: utf-8 -*-
"""
Created on Wed Jan 07, 2021

@author: sk1u19
"""

#%% Libraries:

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras import layers

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

#%% GPU:

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

#%% Downloading dataset:

itera = 1

#kk = 31

N_input = 31*512

print('Dataset size = ', N_input)

Npix = 128

D0 = np.zeros((N_input, Npix, Npix))
GTA = np.zeros((N_input, 9))

for i in range (N_input):
    filename = "Tdp" + str(i) + '.mat'
    M1 = scio.loadmat(filename)
    D1 = M1['E']
    D0[i,:,:] = D1[64:-64, 64:-64]
    GTA[i,:] = M1['labels']

D0 = D0/np.max(D0)

N_train = np.floor(N_input*(30/31)).astype(int)
N_val = N_input - N_train

img_size = (Npix, Npix)
origin_size = 1
#img_size = resize_size_Big

N_pixels = img_size
N_pixels_orgn = origin_size

if K.image_data_format() == 'channels_first':
    print('Channels First')     
    D0 = D0.reshape(N_input, 1, img_size[0], img_size[1])
    In_shape = (1,img_size[0], img_size[1])

else:
    print('Channels Last')     
    D0 = D0.reshape(N_input, img_size[0], img_size[1], 1)
    In_shape = (img_size[0], img_size[1],1)

imgs_train = D0[:N_train]
imgs_val = D0[N_train:]
origins_train = GTA[:N_train]
origins_val = GTA[N_train:]

dp_hl = 0.1

#Size of train and test sets
N_input = np.size(D0, axis=0)
N_train = np.size(origins_train, axis=0)
N_val = np.size(origins_val, axis=0)
assert (N_train+N_val) == N_input \
    ,"The train,val separation has problem."
    
#%% Neural network:
    
## NN architecture:
def create_model():
    
    model = keras.Sequential()
    model.add(layers.Conv2D(256, (3, 3), activation = 'relu', input_shape = In_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
    model.add(layers.Conv2D(32, (3, 3), activation = 'relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation = 'relu'))
    #model.add(layers.Dense(32, activation = 'relu'))
    model.add(layers.Dense(9, activation = 'sigmoid'))
    
    return model



model = create_model()

## Model training + predictions
batch_size = 20
    
model.compile(optimizer='adam',
              loss='mse',
              metrics=['mse'])

filepath = "m1"

callback1 = keras.callbacks.EarlyStopping(monitor = 'val_loss', 
                                          patience = 15, 
                                          mode = 'auto', 
                                          restore_best_weights = True)

callback2 = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, 
                                            save_best_only=False,save_weights_only=False, 
                                            mode='auto', save_freq='epoch', options=None)
    
history = model.fit(imgs_train, origins_train, epochs=70, 
                    batch_size = batch_size, 
                    callbacks = [callback1, callback2], 
                    validation_split = 0.1)

#Loading the model:
#model1 = create_model()
#model1.load_weights(filepath)
#
#model1.compile(optimizer='adam',
#              loss='mse',
#              metrics=['mse'])

test_loss, test_acc = model.evaluate(imgs_val, origins_val, verbose=2)
#
print('\nTest loss:', test_loss)

model.save("M" + str(itera))

predictions = model.predict(imgs_val)

model_name = 'model' + str(itera) + '.h5'

model.save(model_name)


## Saving the results:

fn1 = 'Results_reg1_1.mat'
scio.savemat(fn1, {'mse': history.history['mse'], 
                   'val_mse': history.history['val_loss'], 
                   'GroundTruth': origins_val, 
                   'Predictions': predictions})

#%% Downloading dataset for numerical experiment:

#del D0
#
#step = 18
#Nix = int(900/step)
#    
#N_input = Nix * Nix
#
#Npix = 128
#
#NE = np.zeros((N_input, Npix, Npix))
#df = np.zeros((N_input, 3))
#
#GTB = np.zeros((N_input, 9))
#
#count = 0
#for i in range (Nix):
#    for j in range (Nix):
#        filename = 'Rooster_DP' + str(j) + '-' + str(i) + '.mat'
#        M1 = scio.loadmat(filename)
#        D1 = M1['E']
#        NE[count,:,:] = D1[64:-64, 64:-64]
#        GTB[count,:] = M1['labels']
#        
#        df[count, 0] = count
#        df[count, 1] = i
#        df[count, 2] = j
#
#        count = count + 1
#
#NE = NE/np.max(NE)
#
#img_size = (Npix, Npix)
##img_size = resize_size_Big
#
#N_pixels = img_size
##N_pixels_orgn = origin_size
#
#if K.image_data_format() == 'channels_first':
#    print('Channels First')     
#    NE = NE.reshape(N_input, 1, img_size[0], img_size[1])
#    In_shape = (1,img_size[0], img_size[1])
#
#else:
#    print('Channels Last')     
#    NE = NE.reshape(N_input, img_size[0], img_size[1], 1)
#    In_shape = (img_size[0], img_size[1],1)    
#
##%% Make predictions:
#
#predictions1 = model.predict(NE)
#
### Saving the results:
##step = 220
##r = 50
##fname = 'Random_map_' + str(step) + '_' + str(r) + '.mat'
##M2 = scio.loadmat(fname)
##GT = M2['GroundTruth']
#
#fn1 = 'Results_rooster_1' + '_' + str(itera) + '.mat'
#scio.savemat(fn1, {'Predictions': predictions1, 'GT': GTB, 'df' : df})
    
#%%    
print('Game over!')
