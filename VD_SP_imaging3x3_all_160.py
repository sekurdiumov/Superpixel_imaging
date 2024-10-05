# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 09:46:30 2019

@author: sk1u19
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from numpy.lib import scimath as SC
import time
import matplotlib.colors
import csv
import random as rnd
import scipy.io as scio

#tm = time.time()

matplotlib.rcParams.update({'font.size' : 18})
pi = np.pi

## ============================================================================
## Generating data for particle imaging:
## ============================================================================
## Input field for particle imaging:
def InputField_PI(inputField, spacing, T, step, r, Input):
    PWL = 1/spacing
    Nmax = inputField[0,:].size
    #Nmay = inputField[:,0].size
    
    #T = 0
    
    IF = np.ones_like(inputField)*T
    
    N = 9
   
    Ng = N*step
    Grid0 = np.ones((Ng, Ng))*T
      
    X1 = np.zeros(N)
    for i in range (N):
        X1[i] = step/2 + i*step
        
    Y1 = X1
    
    x = np.linspace(0, Ng, Ng)
    y = np.linspace(0, Ng, Ng)
    
    X, Y = np.meshgrid(x, y)
    
    for i in range(N):
        for j in range(N):
            Grid0[(X - X1[j])**2 + (Y - Y1[i])**2 <= r**2] = Input[i, j]
            
    IF[int(Nmax/2) - int(Ng/2) : int(Nmax/2) + int(Ng/2),
       int(Nmax/2) - int(Ng/2) : int(Nmax/2) + int(Ng/2)] = Grid0
    
       
    return IF

## ============================================================================
## Gaussian beam - source at z=0:
## ============================================================================
def Gauss(X, Y, w):
    
    Gauss = np.exp(-(X**2 + Y**2) / w**2)
    
    return Gauss

## ============================================================================
## 1D to 2D map:
## ============================================================================
def To_2D_Map(profile1D):
    l = int(len(profile1D)/2)
    profile2D_1 = np.empty([l, l])
    for x in range(l):
        for y in range(l):
            r = ((x**2 + y**2)**(1/2.))
            if r < l:
                profile2D_1[x, y] = profile1D[int(l + r)]
            else:
                profile2D_1[x, y] = profile1D[-1]
                
    profile2D_2 = np.fliplr(profile2D_1)
    profile2D_3 = np.flipud(profile2D_2)
    profile2D_4 = np.flipud(profile2D_1)
    
    p14 = np.concatenate((profile2D_4, profile2D_1), axis = 0)
    p23 = np.concatenate((profile2D_3, profile2D_2), axis = 0)
    profile2D = np.concatenate((p23, p14), axis = 1)
    
    return profile2D

## ============================================================================
## Direct Fourier transform of 2D array:
## ============================================================================
def DFT_2D (inputField, spacing):
    Nmax = inputField[0,:].size
    Nmay = inputField[:,0].size
    
    sigma_step_x = 1/(spacing*Nmax)
    sigma_step_y = 1/(spacing*Nmay)
    sigma_x = sigma_step_x*np.arange(Nmax) 
    sigma_y = sigma_step_y*np.arange(Nmay) 
    
    A0 = np.fft.fft2(inputField)
    
    #'''
    sigma_nyquist_x = sigma_step_x*Nmax/2.0
    sigma_nyquist_y = sigma_step_y*Nmay/2.0
    indsx = np.nonzero(sigma_x >= (sigma_nyquist_x + sigma_step_x/3.0))
    indsy = np.nonzero(sigma_y >= (sigma_nyquist_y + sigma_step_y/3.0))
    sigma_x[indsx] = sigma_x[indsx] - 1/spacing
    sigma_y[indsy] = sigma_y[indsy] - 1/spacing
    #'''
    sigmax, sigmay = np.meshgrid(sigma_x, sigma_y)
    
    return A0, sigmax, sigmay

## ============================================================================
## Inverse Fourier transform of 2D array:
## ============================================================================
def IDFT_2D(field, d_sigma_x, d_sigma_y):
    propF_z = np.fft.ifft2(field)
    return propF_z

## ============================================================================
## Including polarization terms:
## ============================================================================
def Psi_ab(sigmax, sigmay):
    
    sigmaz = SC.sqrt(1 - sigmax**2 - sigmay**2)
    
    psi_xx = 1 - sigmax**2
    psi_xy = -sigmax * sigmay
    psi_xz = -sigmax * sigmaz
    psi_yx = -sigmax * sigmay
    psi_yy = 1 - sigmay**2
    psi_yz = -sigmay * sigmaz
    
    return psi_xx, psi_xy, psi_xz, psi_yx, psi_yy, psi_yz

## ============================================================================
## Propagation in Free space:
## ============================================================================
def propFS(inputField, spacing, prop_D, flag):
    
    pi = np.pi                   # const
    
    sizex = inputField[0,:].size
    sizey = inputField[:,0].size
    
    #Ex = np.zeros((len(prop_D), sizex, sizey), dtype = complex)
    #Ey = np.zeros((len(prop_D), sizex, sizey), dtype = complex)
    #propF_z_x = np.zeros((len(prop_D), sizex, sizey), dtype = complex)
    #propF_z_y = np.zeros((len(prop_D), sizex, sizey), dtype = complex)
    propF_z = np.zeros((len(prop_D), sizex, sizey), dtype = complex)
    
    A0, sigmax, sigmay = DFT_2D(inputField, spacing)
    
    psi_xx, psi_xy, psi_xz, psi_yx, psi_yy, psi_yz = Psi_ab(sigmax, sigmay)
    
    ## Mansuripur 1989:
    if flag == 0:
        psi = psi_xx
    else:
        psi = psi_yy
        
    for i in np.arange(len(prop_D)):
        z0 = prop_D[i]
        psi = A0 * psi * np.exp(1j * 2*pi * z0  * SC.sqrt(1 - sigmax**2 - sigmay**2))
        propF_z[i] = np.fft.ifft2(psi)
    
    propF_z = np.abs(propF_z)**2
    
    return propF_z

## ============================================================================
## Write a .csv file from the image:
## ============================================================================
def csvWriter(fil_name, nparray):
    example = nparray.tolist()
    with open(fil_name+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(example)
   
## ============================================================================
## Program itself:
## ============================================================================
# constants
Nmax = 1280
Nmay = 1280
L0 = 20
spacing = L0/Nmax   
PWL = 1/spacing
print ('Points per wavelength: ', PWL)   
T = 0                                             
inputField = np.ones((Nmax, Nmay))*T

NX2 = int(Nmax/2)
NY2 = int(Nmay/2)
x = np.linspace(-NX2, NX2, Nmax)*spacing
y = np.linspace(-NY2, NY2, Nmay)*spacing
X,Y = np.meshgrid(x,y)

z = np.array([2])
#img = np.zeros((N, Nmax*Nmay))

Npixels = 9


## Interpolate the source and make in 2D:
#I0 = np.loadtxt('100x.txt')
#I2 = I0[:,1] - np.min(I0[:,1])
#x1 = np.linspace(-6, 6, len(I2)) * spacing
#x2 = np.linspace(-6, 6, 12*int(PWL)) * spacing
#
#I2a = np.interp(x2, x1, I2)
#I2a = I2a / np.max(I2a)
#
#I2b = np.ones(Nmax) * I2a[-1]
#I2b[4*int(PWL):16*int(PWL)] = I2a
#
#source = To_2D_Map(I2b)

## Gausian beam:
w0 = 0.291/0.64
Gauss1 = Gauss(X, Y, w0)

doplot = 0

N = 512
NNN = 0

step = 16
r = 6
N1 = 52

obj = ['Flower', 'Rooster', 'QR', 'Random_map']

Threshold = 0.5

count = 0
for k in range (len(obj)):
    for i in range (N1):
        for j in range (N1):
            
            tm = time.time()
            
            T = 0.0
            fname = obj[k] + str(i) + '_' + str(j) + '.mat'
            M1 = scio.loadmat(fname)
            IFF = M1['A1']
            #Input = M1['IF']/255
            
            
            #IF1 = InputField_PI(inputField, spacing, T, step, r, Input)
            IF1 = np.ones_like(inputField) * T
            
            box = int(4.5*step)
            X1 = NX2 - box
            X2 = NX2 + box
            IF1[X1:X2, X1:X2] = IFF
            
            GT = np.zeros(9)
            GT[0] = np.mean(IF1[NX2 - int(1.5*step) : NX2 - int(0.5*step), 
              NX2 - int(1.5*step) : NX2 - int(0.5*step)])
            
            GT[1] = np.mean(IF1[NX2 - int(0.5*step) : NX2 + int(0.5*step), 
              NX2 - int(1.5*step) : NX2 - int(0.5*step)])
            
            GT[2] = np.mean(IF1[NX2 + int(0.5*step) : NX2 + int(1.5*step), 
              NX2 - int(1.5*step) : NX2 - int(0.5*step)])
            
            GT[3] = np.mean(IF1[NX2 - int(1.5*step) : NX2 - int(0.5*step), 
              NX2 - int(0.5*step) : NX2 + int(0.5*step)])
            
            GT[4] = np.mean(IF1[NX2 - int(0.5*step) : NX2 + int(0.5*step), 
              NX2 - int(0.5*step) : NX2 + int(0.5*step)])
            
            GT[5] = np.mean(IF1[NX2 + int(0.5*step) : NX2 + int(1.5*step),
              NX2 - int(0.5*step) : NX2 + int(0.5*step)])
            
            GT[6] = np.mean(IF1[NX2 - int(1.5*step) : NX2 - int(0.5*step),
              NX2 + int(0.5*step) : NX2 + int(1.5*step)])
            
            GT[7] = np.mean(IF1[NX2 - int(0.5*step) : NX2 + int(0.5*step), 
              NX2 + int(0.5*step) : NX2 + int(1.5*step)])
            
            GT[8] = np.mean(IF1[NX2 + int(0.5*step) : NX2 + int(1.5*step), 
              NX2 + int(0.5*step) : NX2 + int(1.5*step)])
            
            for n in range(9):
                if GT[n] >= Threshold:
                    GT[n] = 1
                else:
                    GT[n] = 0
            
            print('GT:', GT)
            
            IF11 = IF1 * Gauss1
            
            Ex1 = propFS(IF11, spacing, z, 0)
            Ey1 = propFS(IF11, spacing, z, 1)
            E1 = np.abs(Ex1 + 1j*Ey1)
            
            lim = 2
            lim1 = NX2 - 1.5*step
            lim2 = NX2 + 1.5*step
            
            if doplot:
                
                plt.figure(figsize = [6, 5.2])
                plt.pcolor(IF1, cmap = 'gray')
                plt.colorbar()
                #plt.xlabel('x in wavelengths')
                #plt.ylabel('y in wavelengths')
                plt.xlim(lim1, lim2)
                plt.ylim(lim1, lim2)
                
    #            plt.figure(figsize = [6, 5.2])
    #            plt.pcolor(X, Y, IF11, cmap = 'hot')
    #            plt.colorbar()
    #            plt.xlabel('x in wavelengths')
    #            plt.ylabel('y in wavelengths')
    #            plt.xlim(-lim, lim)
    #            plt.ylim(-lim, lim)
    #            
    #            plt.figure(figsize = [6, 5.2])
    #            plt.pcolor(IFF, cmap = 'gray')
    #            
    #            if np.sum(E1) != 0:
    #                plt.figure(figsize = [7, 7])
    #                plt.pcolormesh(X, Y, np.abs(E1[-1,:,:])/np.max(E1[-1,:,:]), cmap = 'hot')
    #                plt.colorbar()
    #                plt.xlabel('x in wavelengths')
    #                plt.ylabel('y in wavelengths')
    #                plt.xlim(-lim, lim)
    #                plt.ylim(-lim, lim)
    #                
    #            else:
    #                plt.figure(figsize = [7, 7])
    #                plt.pcolormesh(X, Y, np.abs(E1[-1,:,:]), cmap = 'hot')
    #                plt.colorbar()
    #                plt.xlabel('x in wavelengths')
    #                plt.ylabel('y in wavelengths')
    #                plt.xlim(-lim, lim)
    #                plt.ylim(-lim, lim)
            
        
            x1 = int(8*Nmax/20)
            x2 = int(12*Nmax/20)
            y1 = int(8*Nmax/20)
            y2 = int(12*Nmax/20)
        #    
            if np.sum(E1) == 0:
                E2 = E1[0, x1:x2, y1:y2]
            else:
                E2 = E1[0, x1:x2, y1:y2]/np.max(E1)
        #
            #si = str(i + NNN*512)
            #fname = 'Tdp' + si + '.mat'
            fname = obj[k] + '_' + str(i) + '-' + str(j) + '.mat'
        #    
            scio.savemat(fname, {'E': E2**2, 'labels' : GT, 'Input' : IFF})
            
            count = count + 1
    
            elapsed = time.time() - tm
            print ('Time elapsed = ', elapsed, ' i = ', count)

print('Game over!')



      
    

