# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 14:20:44 2015

@author: alexander
"""
import lasagne
import theano
import theano.tensor as T
from skimage import transform
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import sys


def doMaxPool(im):
    input_var = T.tensor4('input_variable')
    l_in_orig = lasagne.layers.InputLayer(shape=(None, 1, 28*4, 28*4),
                                        input_var=input_var)
    l_mp_1 = lasagne.layers.MaxPool2DLayer(l_in_orig, pool_size=(2,2))
    l_mp_2 = lasagne.layers.MaxPool2DLayer(l_mp_1, pool_size=(2,2))
    l_mp_3 = lasagne.layers.MaxPool2DLayer(l_mp_2, pool_size=(2,2))
    l_mp_4 = lasagne.layers.MaxPool2DLayer(l_mp_3, pool_size=(2,2))
    l_mp_5 = lasagne.layers.MaxPool2DLayer(l_mp_4, pool_size=(2,2))

    out1 = lasagne.layers.get_output(l_mp_1, input_var)
    out2 = lasagne.layers.get_output(l_mp_2, input_var)
    out3 = lasagne.layers.get_output(l_mp_3, input_var)
    out4 = lasagne.layers.get_output(l_mp_4, input_var)
    out5 = lasagne.layers.get_output(l_mp_5, input_var)

    f1 = theano.function([input_var], out1)
    f2 = theano.function([input_var], out2)
    f3 = theano.function([input_var], out3)
    f4 = theano.function([input_var], out4)
    f5 = theano.function([input_var], out5)

    im1 = f1(im)    
    im2 = f2(im)
    im3 = f3(im)    
    im4 = f4(im)
    im5 = f5(im)
    return im1, im2, im3, im4, im5

whale = imread('w_7489.jpg')

print whale.shape
whale = whale[:,0:2048,:]
print whale.shape

max1, max2, max3, max4, max5 = doMaxPool(np.expand_dims(np.reshape(whale,(3,1,2)), axis=0))
sys.exit()
skim1 = transform.resize(whale, (whale[0]/2, whale[1]/2))
skim2 = transform.resize(whale, (whale[0]/4, whale[1]/4))
skim3 = transform.resize(whale, (whale[0]/8, whale[1]/8))
skim4 = transform.resize(whale, (whale[0]/16, whale[1]/16))
skim5 = transform.resize(whale, (whale[0]/32, whale[1]/32))

im1024 = np.zeros((1024, 2048, 3), dtype=whale.dtype)
im1024[:,0:1024,3] = skim1 
im1024[:,1024:,3] = max1
im512 = np.zeros((512, 1024, 3), dtype=whale.dtype)
im512[:,0:512,3] = skim2 
im512[:,512:,3] = max2
im256 = np.zeros((256, 512, 3), dtype=whale.dtype)
im256[:,0:256,3] = skim3
im256[:,256:,3] = max3
im128 = np.zeros((128, 256, 3), dtype=whale.dtype)
im128[:,0:128,3] = skim4
im128[:,128:,3] = max4
im64 = np.zeros((64, 128, 3), dtype=whale.dtype)

plt.imsave(fname="whale_1024", arr=whale)
plt.imsave(fname="whale_512", arr=whale)
plt.imsave(fname="whale_256", arr=whale)
plt.imsave(fname="whale_128", arr=whale)
plt.imsave(fname="whale_64", arr=whale)
