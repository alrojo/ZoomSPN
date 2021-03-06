# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 14:20:44 2015

@author: alexander
"""
import lasagne
import theano
import theano.tensor as T
from scipy import ndimage
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import sys


def doMaxPool(im):
    input_var = T.tensor3('input_variable')
    l_in_orig = lasagne.layers.InputLayer(shape=(2048, 2048, 3),
                                        input_var=input_var)
    l_dim = lasagne.layers.DimshuffleLayer(l_in_orig, ('x', 2, 0, 1))
    l_mp_1 = lasagne.layers.MaxPool2DLayer(l_dim, pool_size=(2,2))
    l_mp_2 = lasagne.layers.MaxPool2DLayer(l_mp_1, pool_size=(2,2))
    l_mp_3 = lasagne.layers.MaxPool2DLayer(l_mp_2, pool_size=(2,2))
    l_mp_4 = lasagne.layers.MaxPool2DLayer(l_mp_3, pool_size=(2,2))
    l_mp_5 = lasagne.layers.MaxPool2DLayer(l_mp_4, pool_size=(2,2))
    l_mp_6 = lasagne.layers.MaxPool2DLayer(l_mp_5, pool_size=(2,2))

    l_dim1 = lasagne.layers.DimshuffleLayer(l_mp_1, (2, 3, 1))
    l_dim2 = lasagne.layers.DimshuffleLayer(l_mp_2, (2, 3, 1))
    l_dim3 = lasagne.layers.DimshuffleLayer(l_mp_3, (2, 3, 1))
    l_dim4 = lasagne.layers.DimshuffleLayer(l_mp_4, (2, 3, 1))
    l_dim5 = lasagne.layers.DimshuffleLayer(l_mp_5, (2, 3, 1))
    l_dim6 = lasagne.layers.DimshuffleLayer(l_mp_6, (2, 3, 1))

    out1 = lasagne.layers.get_output(l_dim1, input_var)
    out2 = lasagne.layers.get_output(l_dim2, input_var)
    out3 = lasagne.layers.get_output(l_dim3, input_var)
    out4 = lasagne.layers.get_output(l_dim4, input_var)
    out5 = lasagne.layers.get_output(l_dim5, input_var)
    out6 = lasagne.layers.get_output(l_dim6, input_var)

    f1 = theano.function([input_var], out1)
    f2 = theano.function([input_var], out2)
    f3 = theano.function([input_var], out3)
    f4 = theano.function([input_var], out4)
    f5 = theano.function([input_var], out5)
    f6 = theano.function([input_var], out6)

    im1 = f1(im)    
    im2 = f2(im)
    im3 = f3(im)    
    im4 = f4(im)
    im5 = f5(im)
    im6 = f6(im)
    return im1, im2, im3, im4, im5, im6

whale = imread('w_7489.jpg')

whale = whale[:,0:2048,:]

print "Computing maxpools ..."
max1, max2, max3, max4, max5, max6 = doMaxPool(whale)

print "Computing scipy zooms ..."
skim1 = ndimage.zoom(whale, [.5, .5, 1])
skim2 = ndimage.zoom(whale, [.25, .25, 1])
skim3 = ndimage.zoom(whale, [.125, .125, 1])
skim4 = ndimage.zoom(whale, [.0625, .0625, 1])
skim5 = ndimage.zoom(whale, [.03125, .03125, 1])
skim6 = ndimage.zoom(whale, [.015625, .015625, 1])
# Only works for gray scale, not N dimentional.
#skim1 = transform.resize(whale, (whale[0]/2, whale[1]/2, 3))
#skim2 = transform.resize(whale, (whale[0]/4, whale[1]/4, 3))
#skim3 = transform.resize(whale, (whale[0]/8, whale[1]/8, 3))
#skim4 = transform.resize(whale, (whale[0]/16, whale[1]/16, 3))
#skim5 = transform.resize(whale, (whale[0]/32, whale[1]/32, 3))

print "Concatting images ..."
im1024 = np.zeros((1024, 2048, 3), dtype=whale.dtype)
im1024[:,0:1024,:] = skim1
im1024[:,1024:,:] = max1
im512 = np.zeros((512, 1024, 3), dtype=whale.dtype)
im512[:,0:512,:] = skim2 
im512[:,512:,:] = max2
im256 = np.zeros((256, 512, 3), dtype=whale.dtype)
im256[:,0:256,:] = skim3
im256[:,256:,:] = max3
im128 = np.zeros((128, 256, 3), dtype=whale.dtype)
im128[:,0:128,:] = skim4
im128[:,128:,:] = max4
im64 = np.zeros((64, 128, 3), dtype=whale.dtype)
im64[:,0:64,:] = skim5
im64[:,64:,:] = max5
im32 = np.zeros((32, 64, 3), dtype=whale.dtype)
im32[:, 0:32,:] = skim6
im32[:, 32:, :] = max6

print "Saving all images ..."
plt.imsave(fname="whale_1024", arr=im1024)
plt.imsave(fname="whale_512", arr=im512)
plt.imsave(fname="whale_256", arr=im256)
plt.imsave(fname="whale_128", arr=im128)
plt.imsave(fname="whale_64", arr=im64)
plt.imsave(fname="whale_32", arr=im32)
print "Images saved!!!"
