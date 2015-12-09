# Scale spn

The purpose of this project is to investigate the use of spartial transformer networks on very large images with spartial structures that can be seen, but not nessesarily distinquished, from smaller scalings of the same image.

The algorithms will take a large image, scale it to different sizes and the progressively use transformer networks to crop "into" the larger images. E.g This will allow the algorithm to take crops of 2048x2048 images without ever working on anything above 256x256.

The essense of this project is to test two types of scaling algorithms; Maxpool and scipys resize. Maxpool is currently the algorithm of choice in many deep neural networks, but it is not nessesarily the best algorithm for scaling.

I have(so far) made two contributions to test this:
Firstly, coded a network working on a self-made variant of the mnist.
>>python mnist_zoom_spn.py -h
Either it uses maxpooling to device three different sizes of the mnist variant, and uses transformer networks twice to zoom "into" the larger image. Or it uses scipy to supply images of three different sizes and performs the same computation.

Secondly, whaleresize.py will try take a 2048x2048 image of a whale and rescale it using maxpool and scipy for illustrating the difference between the algorithms.
Here are a few examples, which illustrates how maxpool performs whose the more severe the scaling is.
![alt text](https://github.com/alrojo/ZoomSPN/whale_1024.jpg "downscaling with a factor of 2 - from 2048 to 1024")
![alt text](https://github.com/alrojo/ZoomSPN/whale_1024.jpg "downscaling with a factor of 8 - from 2048 to 256")
![alt text](https://github.com/alrojo/ZoomSPN/whale_1024.jpg "downscaling with a factor of 32 - from 2048 to 64")
![alt text](https://github.com/alrojo/ZoomSPN/whale_1024.jpg "downscaling with a factor of 64 - from 2048 to 32")
