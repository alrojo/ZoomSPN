Zooming spn cnn, use -h flag for options menu
This contains two main files:

mnist_zoom_spn.py allows to train a zooming spn where you can either use maxpool rescaling or scikit-image rescaling(ZOOM option) and try dbl gradients(dbl_calc) for the first spn.

Use mnist_zoom_spn.py -h to see all the args options.

whaleresize.py will try take a 2048x2048 image of a whale and rescale it using maxpool and scipy for illustration purposes.
