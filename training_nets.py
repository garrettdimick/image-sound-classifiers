#################
## Garrett Dimick
## A01227378
## Tests and Trains four neural networks:
## One ANN and one Convnet for classifying Images
## and one ANN and one Convnet for classifying Sounds
## Datasets were gathered by Professor Vladimir Kulyukin of Utah State University
#################

import cv2
import numpy as np
import glob
import random
import scipy as sp
import network
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import pickle as cPickle
import tflearn.datasets.mnist as mnist

def save(obj, file_name):
    with open(file_name, 'wb') as fp:
        cPickle.dump(obj, fp)
## First, turn data into numpy arrays, first do this for images
## each image will be represented as a tuple, where the first entry is the image
## represented as a 32x32 array of 3-tuples (B,G,R) and the second is the
## classification of the image, a 1 for a bee and a 0 for a not bee
## the test data is in a similar format
## 38,139 images in the training data
## img is a 32x32 array of 3-tuples(B,G,R) each value in B,G,R is in [0,255],
## scale them to between 0 and 1 by /float(255), convert each image to grayscale
## for the ANN
# ann_train_data = [np.zeros((38139, 1024))]
# convnet_train_pre_data = np.zeros((38139, 1024))
ann_train_data = []
convnet_train_pre_data = []
## Process all the Bee Training Data
## first add the data from the bee_train, 19,082 images
f = glob.glob("/Users/garrettdimick/Google Drive/Fall 2018/CS5600-AI/Project_01_Data/BEE2Set/bee_train/*/*.png")
for file in f:
    img = cv2.imread(file)
    convnet_train_pre_data.append((img/float(255), 1))
    ## Grayscale the images for the ANN
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ann_train_data.append((gray_img/float(255), 1))
## now the no bee from no_bee_train, 19,057 images
f = glob.glob("/Users/garrettdimick/Google Drive/Fall 2018/CS5600-AI/Project_01_Data/BEE2Set/no_bee_train/*/*.png")
for file in f:
    img = cv2.imread(file)
    convnet_train_pre_data.append((img/float(255), 0))
    ## Grayscale the images for the ANN
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ann_train_data.append((gray_img/float(255), 0))

## reshape the data for better use in the ANN
atd = [np.reshape(x[0], (1024, 1)) for x in ann_train_data]
ard = [y[1] for y in ann_train_data]
ann_image_training_data = zip(atd, ard)

## training data for ANN that will classify images
ann_image_train_data = ann_image_training_data[0:35000]
## eval data for ANN that will classify images
ann_image_eval_data = ann_image_training_data[35000:]

## turn the convnet data into a usable format for tflearn stuff
x = [i[0] for i in convnet_train_pre_data]
y = [i[1] for i in convnet_train_pre_data]
x, y = shuffle(x, y)
trainX = x[0:35000]
image_evalX = x[35000:]
image_trainY = y[0:35000]
image_evalY = y[35000:]
## reshape the data
image_trainX = trainX.reshape([-1, 32, 32, 1])

## Get test data for the image ANN and convnet
## Process all the Bee Test Data
## first add the data from the bee_test
f = glob.glob("/Users/garrettdimick/Google Drive/Fall 2018/CS5600-AI/Project_01_Data/BEE2Set/bee_test/*/*.png")
for file in f:
    img = cv2.imread(file)
    convnet_test_pre_data.append((img/float(255), 1))
    ## Grayscale the images for the ANN
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ann_test_data.append((gray_img/float(255), 1))
## now the no bee from no_bee_train, 19,057 images
f = glob.glob("/Users/garrettdimick/Google Drive/Fall 2018/CS5600-AI/Project_01_Data/BEE2Set/no_bee_test/*/*.png")
for file in f:
    img = cv2.imread(file)
    convnet_test_pre_data.append((img/float(255), 0))
    ## Grayscale the images for the ANN
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ann_test_data.append((gray_img/float(255), 0))



## now turn data into numpy arrays for sounds

## Build an image classifying ANN
## Architecture: 1024 x 60 x 30 x 15 x 2
## 3 hidden layers
image_ann = Network([1024, 60, 30, 15, 2])

# Begin training the ANN using stochastic gradient descent
