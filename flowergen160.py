# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 14:59:21 2022

@author: sk1u19
"""

import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import scipy.io as scio
import time
import imageio

tm = time.time()

## ============================================================================
## Generating the random pixellated pattern:
## ============================================================================
step = 160
px = int(0.1*step)

N = 50

Xmax = step * N / 2

fname = 'flower' + str(step) + '.bmp'
A = imageio.imread(fname)

A0 = np.zeros(((N + 10)*px, (N + 10)*px))

A0[5*px:-5*px, 5*px:-5*px] = A[:,:,0]/255

A0[A0 > 0.5] = 1
A0[A0 < 0.5] = 0

plt.figure(1)
plt.imshow(A0)
plt.colorbar()

# Generating the input field:
X0 = 0
Y0 = 0

count = 0
N1 = N + 2
#N1 = 4
for i in range(N1):
    
    X0 = 0
    
    for j in range(N1):
        
        A1 = A0[X0 : X0 + 9*px, Y0 : Y0 + 9*px]
        
#        IF1 = IF[i, j, :, :]
#        GT1 = GT[count, :]
        
        fname = 'Flower' + str(i) + '_' + str(j) + '.mat'
        scio.savemat(fname, {'A1': A1})
        
        X0 = X0 + px
        count = count + 1
        
    Y0 = Y0 + px

print('elapsed = ', time.time() - tm)
