# ------------------------------------------------------------------------------------
# This code helps to find the saliency map over the input image given a pre-trained 
# CNN and the output label (vector), for which saliency map has to be estimated.
# Author: Sukrit Shankar 
# ------------------------------------------------------------------------------------

# -------------------------------------
import numpy as np 
import matplotlib.pyplot as plt
import sys
import operator
from PIL import Image
from PIL import ImageDraw
import os

# -------- Import Caffe --------------- 
caffe_root = '/home/sukrit/Desktop/caffe_latest/'  
sys.path.insert(0, caffe_root + 'python')
import caffe

# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# You can configure your stuff here
CNN_ARCH_FILE = 'models/deploy_fc8.prototxt'				# Ensure force_backward: true is included to enable BP to input  
PRETRAINED_MODEL = 'models/bvlc_reference_caffenet.caffemodel'		# Pretrained CNN Model using which saliency needs to be computed
IMAGE_FILE = 'input_images/cat.jpg'					# Input Image for which saliency has to be computed 
MEAN_FILE = 'models/ilsvrc_2012_mean.npy'				# Mean file of the dataset with which PRETRAINED_MODEL was trained
IMAGE_HW_ORIG = 256							# Original Height = Width of Images expected for the PRETRAINED_MODEL
NUM_OUTPUT_LABELS = 1000						# Total number of output labels - Should confirm with PRETRAINED_MODEL 
LABEL_FOR_SALIENCY = 281						# The label with respect to which saliency is calculated (Starts from 0)
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

# -------------------------------------
# Load the input image with Caffe
caffe.set_mode_gpu()
net = caffe.Classifier(CNN_ARCH_FILE, PRETRAINED_MODEL,
                       mean=np.load(MEAN_FILE).mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(IMAGE_HW_ORIG, IMAGE_HW_ORIG))
input_image = caffe.io.load_image(IMAGE_FILE)

# -------------------------------------
# Form the final label vector 
finalLabelVector = np.zeros((1,NUM_OUTPUT_LABELS))
finalLabelVector[0,LABEL_FOR_SALIENCY] = 1

# -------------------------------------
# Do forward pass with input image
net.predict([input_image])

# -------------------------------------
# Do backward pass using finalLabelVector - Now the net params have been populated 
backwardpassData = net.backward(**{net.outputs[0]: finalLabelVector})

# -------------------------------------
# Get derivatives (delta) - delta at a layer equals gradient of the loss below that layer with respect to the output at the layer 
# Here we get derivatives with respect to the input (data layer)
# This will require force_backward: true in the deploy file for Caffe to do a backward pass till the data layer 
delta = backwardpassData['data']

# -------------------------------------
# As stated in "Deep Inside Convolutional Networks ..." by Simonyan et al., we now compute the image saliency map
delta = delta - delta.min()		      # Subtract min
delta = delta / delta.max()		      # Normalize by dividing by max 
saliency = np.amax(delta,axis=1)	            # Find max across RGB channels 

# -------------------------------------
# Show the saliency map and the image
# Note that the image shown is the one considered by Caffe after reverting preprocessing (this may result in aspect ratio changes)
plt.subplot(1,2,1)
plt.imshow (net.transformer.deprocess('data', net.blobs['data'].data[0]))
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow (saliency[0,:,:],cmap='copper')
plt.axis('off')

plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('saliency_visualization.png')
# -------------------------------------





