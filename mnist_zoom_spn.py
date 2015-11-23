#!/usr/bin/env python

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
from skimage import transform
import matplotlib.pyplot as plt

import lasagne
import string

# ########################### Getting the data #############################
# Downloads and modifies the MNIST to be a noise grid of twice it's size 
# with Mnist numbers randomly placed on the grid

def load_dataset():
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        #############
        # @@@@@ Making the scaled version for Zooming-st @@@@@
        #############
        print("Making mnist grid and scaling ...")
        data_orig = np.random.randint(
            low=0, high=255,
            size=(data.shape[0], data.shape[1], data.shape[2]*4, data.shape[3]*4)).astype('float32')
        data_rescaled_5 = np.zeros((data.shape[0], data.shape[1], data.shape[2]*2, data.shape[3]*2), dtype='float32')
        data_rescaled_25 = np.zeros((data.shape[0], data.shape[1], data.shape[2], data.shape[3]), dtype='float32')
        np.random.seed(seed=42)
        spacing_1_a = np.random.randint(low=0, high=data_orig.shape[2]-data.shape[2], size=data.shape[0])
        spacing_1_b = spacing_1_a + data.shape[2]
        np.random.seed(seed=24)
        spacing_2_a = np.random.randint(low=0, high=28, size=data.shape[0])
        spacing_2_b = spacing_2_a + data.shape[3]
        data_orig = data_orig.astype('float32') / np.float32(256)
        data = data.astype('float32') / np.float32(256)
        for i in range(data.shape[0]):
            data_orig[i, :, spacing_1_a[i]:spacing_1_b[i], spacing_2_a[i]:spacing_2_b[i]] = data[i, :, :, :]
            data_rescaled_5[i,0,:,:] = transform.resize(data_orig[i,0,:,:], (data.shape[2]*2,data.shape[3]*2))            
            data_rescaled_25[i,0,:,:] = transform.resize(data_orig[i,0,:,:], (data.shape[2],data.shape[3]))

        print("shapes of data_orig, data_rescaled 50% and data_rescaled 25%")        
        print(data_orig.shape)
        print(data_rescaled_5.shape)
        print(data_rescaled_25.shape)
        print("saving images of the grid and rescaled grid ...")
        plt.imsave(fname="orig", arr=data_orig[100,0,:,:], cmap=plt.cm.gray)
        plt.imsave(fname="rescaled_5", arr=data_rescaled_5[100,0,:,:], cmap=plt.cm.gray)
        plt.imsave(fname="rescaled_25", arr=data_rescaled_25[100,0,:,:], cmap=plt.cm.gray)
        print("images saved!")
        return data_orig, data_rescaled_5, data_rescaled_25

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    X_train_orig, X_train_rescaled_5, X_train_rescaled_25 = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test_orig, X_test_rescaled_5, X_test_rescaled_25 = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    X_train_orig, X_val_orig = X_train_orig[:-10000], X_train_orig[-10000:]
    X_train_rescaled_5, X_val_rescaled_5 = X_train_rescaled_5[:-10000], X_train_rescaled_5[-10000:]
    X_train_rescaled_25, X_val_rescaled_25 = X_train_rescaled_25[:-10000], X_train_rescaled_25[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    return X_train_orig, X_train_rescaled_5, X_train_rescaled_25, y_train, X_val_orig, X_val_rescaled_5, X_val_rescaled_25, y_val, X_test_orig, X_test_rescaled_5, X_test_rescaled_25, y_test


# ##################### Build the neural network model #######################
# Builds an st with option of zooming on the original image


def build_st_cnn(input_var_orig=None, input_var_rescaled_5=None, input_var_rescaled_25=None, zoom=True):

    # Input images in 3 different scale
    l_in_orig = lasagne.layers.InputLayer(shape=(None, 1, 28*4, 28*4),
                                        input_var=input_var_orig)

    l_in_rescaled_5 = lasagne.layers.InputLayer(shape=(None, 1, 28*2, 28*2),
                                        input_var=input_var_rescaled_5)
    l_in_rescaled_25 = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var_rescaled_25)

    # CNN for the ST_1 - Zooming 50%
    if zoom:
        l_conv_st_1_1_a = lasagne.layers.Conv2DLayer(
                l_in_rescaled_25, num_filters=16, filter_size=(3, 3), pad='same')
    else:
        l_pool_st_1_1_a = lasagne.layers.MaxPool2DLayer(l_in_orig, pool_size=(2, 2))
        l_pool_st_1_1_b = lasagne.layers.MaxPool2DLayer(l_pool_st_1_1_a, pool_size=(2, 2))        
        l_conv_st_1_1_a = lasagne.layers.Conv2DLayer(
                l_pool_st_1_1_b, num_filters=16, filter_size=(3, 3), pad='same')        
    l_conv_st_1_1_b = lasagne.layers.Conv2DLayer(
            l_conv_st_1_1_a, num_filters=32, filter_size=(3, 3), pad='same')
    l_mp_st_1_1 = lasagne.layers.MaxPool2DLayer(l_conv_st_1_1_b, pool_size=(2, 2))
    l_dense_st_1_1 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_mp_st_1_1, p=.5), num_units=128)
    b_1 = np.zeros((2, 3), dtype='float32')
    b_1[0, 0] = 1
    b_1[1, 1] = 1
    b_1 = b_1.flatten()
    W_1 = lasagne.init.Constant(0.0)
    l_dense_st_out_1 = lasagne.layers.DenseLayer(
            l_dense_st_1_1, num_units=6, W=W_1, b=b_1)
    if zoom:
        l_st_1 = lasagne.layers.TransformerLayer(l_in_rescaled_5, l_dense_st_out_1, downsample_factor=2)
    else:
        l_st_1 = lasagne.layers.TransformerLayer(l_in_orig, l_dense_st_out_1, downsample_factor=2)
        l_pool_st_1_1_a = lasagne.layers.MaxPool2DLayer(l_st_1, pool_size=(2, 2))
    # CNN for the ST_2 - Zooming 25%
    l_conv_st_2_1_a = lasagne.layers.Conv2DLayer(
            l_pool_st_1_1_a, num_filters=16, filter_size=(3, 3), pad='same')
    l_conv_st_2_1_b = lasagne.layers.Conv2DLayer(
            l_conv_st_2_1_a, num_filters=32, filter_size=(3, 3), pad='same')
    l_mp_st_2_1 = lasagne.layers.MaxPool2DLayer(l_conv_st_2_1_b, pool_size=(2, 2))
    l_dense_st_2_1 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_mp_st_2_1, p=.5), num_units=128)
    b_2 = np.zeros((2, 3), dtype='float32')
    b_2[0, 0] = 1
    b_2[1, 1] = 1
    b_2 = b_2.flatten()
    W_2 = lasagne.init.Constant(0.0)
    l_dense_st_out_2 = lasagne.layers.DenseLayer(
            l_dense_st_2_1, num_units=6, W=W_2, b=b_2)
    if zoom:
        l_st_2_1 = lasagne.layers.TransformerLayer(l_in_orig, l_dense_st_out_1, downsample_factor=2)
        l_st_2_2 = lasagne.layers.TransformerLayer(l_st_2_1, l_dense_st_out_2, downsample_factor=2)
    else:
        l_st_2_2 = lasagne.layers.TransformerLayer(l_st_1, l_dense_st_out_2, downsample_factor=2)

    # Convnet
    convnet = lasagne.layers.Conv2DLayer(
            l_st_2_2, num_filters=16, filter_size=(3, 3), pad='same')
    convnet = lasagne.layers.Conv2DLayer(
            convnet, num_filters=32, filter_size=(3, 3), pad='same')
    convnet = lasagne.layers.MaxPool2DLayer(convnet, pool_size=(2, 2))
    convnet = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(convnet, p=.5), num_units=128)
    l_out = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(convnet, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)
    return l_out

def build_st_cnn_large(input_var_orig=None, input_var_rescaled_5=None, input_var_rescaled_25=None, zoom=True):
    # Input images in 3 different scale
    l_in_orig = lasagne.layers.InputLayer(shape=(None, 1, 28*4, 28*4),
                                        input_var=input_var_orig)

    l_in_rescaled_5 = lasagne.layers.InputLayer(shape=(None, 1, 28*2, 28*2),
                                        input_var=input_var_rescaled_5)
    l_in_rescaled_25 = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var_rescaled_25)

    # CNN for the ST_1 - Zooming 50%
    if zoom:
        l_conv_st_1_1_a = lasagne.layers.Conv2DLayer(
                l_in_rescaled_25, num_filters=16, filter_size=(3, 3), pad='same')
    else:
        l_pool_st_1_1_a = lasagne.layers.MaxPool2DLayer(l_in_orig, pool_size=(2, 2))
        l_pool_st_1_1_b = lasagne.layers.MaxPool2DLayer(l_pool_st_1_1_a, pool_size=(2, 2))        
        l_conv_st_1_1_a = lasagne.layers.Conv2DLayer(
                l_pool_st_1_1_b, num_filters=32, filter_size=(3, 3), pad='same')        
    l_conv_st_1_1_b = lasagne.layers.Conv2DLayer(
            l_conv_st_1_1_a, num_filters=32, filter_size=(3, 3), pad='same')
    l_mp_st_1_1 = lasagne.layers.MaxPool2DLayer(l_conv_st_1_1_b, pool_size=(2, 2))
    l_dense_st_1_1 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_mp_st_1_1, p=.5), num_units=256)
    b_1 = np.zeros((2, 3), dtype='float32')
    b_1[0, 0] = 1
    b_1[1, 1] = 1
    b_1 = b_1.flatten()
    W_1 = lasagne.init.Constant(0.0)
    l_dense_st_out_1 = lasagne.layers.DenseLayer(
            l_dense_st_1_1, num_units=6, W=W_1, b=b_1)
    if zoom:
        l_st_1 = lasagne.layers.TransformerLayer(l_in_rescaled_5, l_dense_st_out_1, downsample_factor=2)
    else:
        l_st_1 = lasagne.layers.TransformerLayer(l_in_orig, l_dense_st_out_1, downsample_factor=2)
        l_pool_st_1_1_a = lasagne.layers.MaxPool2DLayer(l_st_1, pool_size=(2, 2))
    # CNN for the ST_2 - Zooming 25%
    l_conv_st_2_1_a = lasagne.layers.Conv2DLayer(
            l_pool_st_1_1_a, num_filters=32, filter_size=(3, 3), pad='same')
    l_conv_st_2_1_b = lasagne.layers.Conv2DLayer(
            l_conv_st_2_1_a, num_filters=32, filter_size=(3, 3), pad='same')
    l_mp_st_2_1 = lasagne.layers.MaxPool2DLayer(l_conv_st_2_1_b, pool_size=(2, 2))
    l_dense_st_2_1 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_mp_st_2_1, p=.5), num_units=256)
    b_2 = np.zeros((2, 3), dtype='float32')
    b_2[0, 0] = 1
    b_2[1, 1] = 1
    b_2 = b_2.flatten()
    W_2 = lasagne.init.Constant(0.0)
    l_dense_st_out_2 = lasagne.layers.DenseLayer(
            l_dense_st_2_1, num_units=6, W=W_2, b=b_2)
    if zoom:
        l_st_2_1 = lasagne.layers.TransformerLayer(l_in_orig, l_dense_st_out_1, downsample_factor=2)
        l_st_2_2 = lasagne.layers.TransformerLayer(l_st_2_1, l_dense_st_out_2, downsample_factor=2)
    else:
        l_st_2_2 = lasagne.layers.TransformerLayer(l_st_1, l_dense_st_out_2, downsample_factor=2)

    # Convnet
    convnet = lasagne.layers.Conv2DLayer(
            l_st_2_2, num_filters=32, filter_size=(3, 3), pad='same')
    convnet = lasagne.layers.Conv2DLayer(
            convnet, num_filters=32, filter_size=(3, 3), pad='same')
    convnet = lasagne.layers.MaxPool2DLayer(convnet, pool_size=(2, 2))
    convnet = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(convnet, p=.5), num_units=256)
    l_out = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(convnet, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)
    return l_out

def iterate_minibatches(inputs_orig, inputs_rescaled_5, inputs_rescaled_25, targets, batchsize, shuffle=False):
    assert len(inputs_orig) == len(targets)
    assert len(inputs_rescaled_5) == len(targets)
    assert len(inputs_rescaled_25) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs_orig))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs_orig) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs_orig[excerpt], inputs_rescaled_5[excerpt], inputs_rescaled_25[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(model='st_cnn', zoom=True, num_epochs=500):
    print("Loading data...")
    X_train_orig, X_train_rescaled_5, X_train_rescaled_25, y_train, X_val_orig, X_val_rescaled_5, X_val_rescaled_25, y_val, X_test_orig, X_test_rescaled_5, X_test_rescaled_25, y_test = load_dataset()

    input_var_orig = T.tensor4('inputs_orig')
    input_var_rescaled_5 = T.tensor4('inputs_rescaled_5')
    input_var_rescaled_25 = T.tensor4('inputs_rescaled_25')
    target_var = T.ivector('targets')

    print("Building model and compiling functions...")

    if model == 'st_cnn':
        l_out = build_st_cnn(input_var_orig, input_var_rescaled_5, input_var_rescaled_25, zoom)
    elif model == 'st_cnn_large':
        l_out = build_st_cnn_large(input_var_orig, input_var_rescaled_5, input_var_rescaled_25, zoom)
    else:
        print("Unrecognized model type %r." % model)
        return
    all_layers = lasagne.layers.get_all_layers(l_out)
    num_params = lasagne.layers.count_params(l_out)
    print("  number of parameters: %d" % num_params)
    print("  layer output shapes:")
    for layer in all_layers:
        name = string.ljust(layer.__class__.__name__, 32)
        print("    %s %s" % (name, lasagne.layers.get_output_shape(layer)))
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(l_out)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(l_out, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(l_out, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var_orig, input_var_rescaled_5, input_var_rescaled_25, target_var], loss, updates=updates, on_unused_input='ignore')

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var_orig, input_var_rescaled_5, input_var_rescaled_25, target_var], [test_loss, test_acc], on_unused_input='ignore')

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train_orig, X_train_rescaled_5, X_train_rescaled_25, y_train, 500, shuffle=True):
            inputs_orig, inputs_rescaled_5, inputs_rescaled_25, targets = batch
            train_err += train_fn(inputs_orig, inputs_rescaled_5, inputs_rescaled_25, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val_orig, X_val_rescaled_5, X_val_rescaled_25, y_val, 500, shuffle=False):
            inputs_orig, inputs_rescaled_5, inputs_rescaled_25, targets = batch
            err, acc = val_fn(inputs_orig, inputs_rescaled_5, inputs_rescaled_25, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test_orig, X_test_rescaled_5, X_test_rescaled_25, y_test, 500, shuffle=False):
        inputs_orig, inputs_rescaled_5, inputs_rescaled_25, targets = batch
        err, acc = val_fn(inputs_orig, inputs_rescaled_5, inputs_rescaled_25, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'st_cnn' for a small st_cnn (default: st_cnn)")
        print("       'st_cnn_large' for a large st_cnn")
        print("ZOOM:  '1' for a zoom_st_cnn or '0' for std. st_cnn (default: 1)")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['zoom'] = bool(int(sys.argv[2]))
        if len(sys.argv) > 3:
            kwargs['num_epochs'] = int(sys.argv[3])
        main(**kwargs)
