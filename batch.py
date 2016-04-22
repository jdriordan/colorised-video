import numpy as np
import caffe
import os
import sys

import skimage.color as color
import scipy.ndimage.interpolation as sni

from scipy import misc

# (1) <b> Opening the model </b> First, we need to open the caffemodel. The model input blob is <i>data_l</i>, and has shape 1x1x224x224 by default. The model output is <i>class8_ab</i>, and has shape 1x2x56x56. We also need to set the temperature T for the annealed mean operation. Blob <i>Trecip</i> is the <i> reciprocal </i> of the temperature.


gpu_id = 0
caffe.set_mode_gpu()
caffe.set_device(gpu_id)
net = caffe.Net('colorization_deploy_v0.prototxt', 'colorization_release_v0.caffemodel', caffe.TEST)

(H_in,W_in) = net.blobs['data_l'].data.shape[2:] # get input shape
(H_out,W_out) = net.blobs['class8_ab'].data.shape[2:] # get output shape
net.blobs['Trecip'].data[...] = 6/np.log(10) # 1/T, set annealing temperature

# <b> (2) Loading the image </b> Next, we need to load our image of choice. We will convert the image at its full resolution to <i>Lab</i> and keep it's <i>L</i> value; we will concatenate the network's color to it! We then resize the image to the network input size, convert to <i>Lab</i>, and only keep the resized <i>L</i>, since network of course does not get any color inputs!


# load the original image
input_filename = sys.argv[1]
print "Processing ", input_filename
img_rgb = caffe.io.load_image(input_filename)
img_lab = color.rgb2lab(img_rgb) # convert image to lab color space
img_l = img_lab[:,:,0] # pull out L channel
(H_orig,W_orig) = img_rgb.shape[:2] # original image size

# resize image to network input size
img_rs = caffe.io.resize_image(img_rgb,(H_in,W_in)) # resize image to network input size
img_lab_rs = color.rgb2lab(img_rs)
img_l_rs = img_lab_rs[:,:,0]


# <b> (3) Colorization time!</b> Now it is time to run the network. We subtract 50 from the <i>L</i> channel (for mean centering), push it into the network, and run a forward pass. Then, we take the output from <i>class8_ab</i>, resize it to the full resolution, concatenate with the <i>L</i> channel, convert to rgb, and display the result.

net.blobs['data_l'].data[0,0,:,:] = img_l_rs-50 # subtract 50 for mean-centering
net.forward() # run network

ab_dec = net.blobs['class8_ab'].data[0,:,:,:].transpose((1,2,0)) # this is our result
ab_dec_us = sni.zoom(ab_dec,(1.*H_orig/H_out,1.*W_orig/W_out,1)) # upsample to match size of original image L
img_lab_out = np.concatenate((img_l[:,:,np.newaxis],ab_dec_us),axis=2) # concatenate with original image L
img_rgb_out = np.clip(color.lab2rgb(img_lab_out),0,1) # convert back to rgb

# Assume three letter filename extension
output_filename = input_filename[0:-4]+"_c"+input_filename[-4:]
misc.imsave(output_filename,img_rgb_out)
print "Wrote " + output_filename + ". Done!"
