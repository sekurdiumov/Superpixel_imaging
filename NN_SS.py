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
#%% Downloading dataset for numerical experiment:

#del D0
#
step = 12
Nix = int(1000/step)
#Nix = 50
    
N_input = Nix * Nix

Npix = 128

NE = np.zeros((N_input, Npix, Npix))
df = np.zeros((N_input, 3))

GTB = np.zeros((N_input, 9))

count = 0
for i in range (Nix):
    for j in range (Nix):
        filename = 'SS_DP' + str(i) + '-' + str(j) + '.mat'
        M1 = scio.loadmat(filename)
        D1 = M1['E']
        NE[count,:,:] = D1[64:-64, 64:-64]
        GTB[count,:] = M1['labels']
        
        df[count, 0] = count
        df[count, 1] = i
        df[count, 2] = j

        count = count + 1

NE = NE/np.max(NE)

img_size = (Npix, Npix)
#img_size = resize_size_Big

N_pixels = img_size
#N_pixels_orgn = origin_size

if K.image_data_format() == 'channels_first':
    print('Channels First')     
    NE = NE.reshape(N_input, 1, img_size[0], img_size[1])
    In_shape = (1,img_size[0], img_size[1])

else:
    print('Channels Last')     
    NE = NE.reshape(N_input, img_size[0], img_size[1], 1)
    In_shape = (img_size[0], img_size[1],1)    

#%% Make predictions:

filepath = "M1"

#Loading the model:
model1 = create_model()
model1.load_weights(filepath)

model1.compile(optimizer='adam',
              loss='mse',
              metrics=['mse'])

predictions1 = model1.predict(NE)

## Saving the results:
#step = 220
#r = 50
#fname = 'Random_map_' + str(step) + '_' + str(r) + '.mat'
#M2 = scio.loadmat(fname)
#GT = M2['GroundTruth']

fn1 = 'Results_SS' + '_' + str(step) + '0.mat'
scio.savemat(fn1, {'Predictions': predictions1, 'GT': GTB, 'df' : df})
    
#%%    
print('Game over!')
