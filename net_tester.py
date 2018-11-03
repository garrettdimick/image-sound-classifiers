import cv2
import numpy as np
import glob
import random
import scipy as sp
import network as nw
import tflearn
import training_nets as tn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d, conv_1d, max_pool_1d
from tflearn.layers.estimator import regression
import cPickle
from scipy.io import wavfile
from tensorflow import reset_default_graph

def load(file_name):
    with open(file_name, 'rb') as fp:
        nn = cPickle.load(fp)
    return nn

def ternary_result(j):
    """Return a 3-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  A 1.0 in the first position indicates a bee noise and
    a 1.0 in the second position indicates a cricket noise while a 1.0 in the
    third position indicates an ambient noise
    """
    e = np.zeros((3, 1))
    e[j] = 1
    return np.asarray(e)

def binary_result(j):
    """Return a 2-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  A 1.0 in the first position indicates no_bee and
    a 1.0 in the second position indicates yes_bee
    """
    e = np.zeros((2, 1))
    e[j] = 1
    return np.asarray(e)
# returns [ann_image_train_data, image_trainX, image_trainY, ann_image_test_data,
#           ann_image_eval_data, image_testX, image_testY, image_evalX, image_evalY]
# image_data = tn.load_image_data

# returns [sound_train_data, sound_test_data, trainX, trainY, testX, testY, validX, validY,
#           audio_training_data, audio_testing_data, valid_data]
## get the bee test data
# sound_data = tn.load_sound_data

# reads image from a file, pre-processes image, feeds to ann, returns two element
# numpy binary array where the first element is set to 1 if the image belongs to
# the yes bee class and second element is set to 1 if the image belongs to no bee
# class, like array([1,0]) for yes or array([0,1]) for no

def fit_image_ann(im_ann, image_path):
    # read image into numpy array
    img = cv2.imread(image_path)
    ## Grayscale the image for the ANN
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ## reshape data
    processed_img = np.reshape(gray_img/float(255.0), (1024, 1))
    ## test image on ann
    return binary_result(np.argmax(im_ann.feedforward(processed_img))).astype(int)
    # return res

def fit_image_convnet(im_convnet, image_path):
    # read image into numpy array
    img = (cv2.imread(image_path)/float(255))
    ## reshape data for convnet
    processed_img = img.reshape([-1, 32, 32, 3])
    ## test image on convnet
    pred = im_convnet.predict(processed_img)
    return binary_result([np.where(r==1)[0][0] for r in np.round(pred)]).astype(int)

def fit_audio_ann(aud_ann, audio_path):
    samplerate, audio = wavfile.read(audio_path)
    scaled_audio = tn.condense_data(audio/float(np.max(audio)))
    ## reshape audio data
    processed_audio = np.reshape(scaled_audio, (7921, 1))
    ## test audio on ann
    return ternary_result(np.argmax(aud_ann.feedforward(processed_audio))).astype(int)

def fit_audio_convnet(aud_convnet, audio_path):
    samplerate, audio = wavfile.read(audio_path)
    scaled_audio = tn.condense_data(audio/float(np.max(audio)))
    ## reshape audio data
    processed_audio = np.reshape(scaled_audio, [-1, 89, 89, 1])
    ## test audio on convnet
    pred = aud_convnet.predict(processed_audio)
    return ternary_result([np.where(r==1)[0][0] for r in np.round(pred)]).astype(int)

#UNCOMMENT THE NEXT 7 LINES TO TEST LOADING AND TESTING IMAGE ON IMAGE ANN
# im_ann = nw.load("/Users/garrettdimick/5600project/networks/ImageANN.pck")
# result = fit_image_ann(im_ann, "/Users/garrettdimick/5600project/BEE2Set/bee_test/img6/696_5_yb.png")
# print "Yes bee image: "
# print result
# result = fit_image_ann(im_ann, "/Users/garrettdimick/5600project/BEE2Set/no_bee_test/img6/192_168_4_8-2017-05-08_17-45-28_65_74_86.png")
# print "No bee image: "
# print result

#DEFINE CONVNET TO LOAD IT
##UNCOMMENT THROUGHT LINE 121 TO TEST LOADING AND TESTING IMAGE ON IMAGE CONVNET
# input_layer = input_data(shape=[None, 32, 32, 3])
# conv_layer = conv_2d(input_layer,
#                     nb_filter=32,
#                     filter_size=3,
#                     activation='relu',
#                     name='conv_layer_1')
# pool_layer = max_pool_2d(conv_layer, 2, name='pool_layer_1')
#
# fc_layer_1 = fully_connected(pool_layer, 512, activation='relu', name='fc_layer_1')
# dropout_layer = dropout(fc_layer_1, 0.5)
# fc_layer_2 = fully_connected(dropout_layer, 2, activation='softmax', name='fc_layer_2')
# network = regression(fc_layer_2, optimizer='sgd',
#                     loss='categorical_crossentropy', learning_rate=0.01)
# im_convnet = tflearn.DNN(network)
# im_convnet.load("/Users/garrettdimick/5600project/networks/ImageConvnet/ImageConvnet.tfl")
# result = fit_image_convnet(im_convnet, "/Users/garrettdimick/5600project/BEE2Set/bee_test/img6/696_5_yb.png")
# print "Yes bee image: "
# print result
# result = fit_image_convnet(im_convnet, "/Users/garrettdimick/5600project/BEE2Set/no_bee_test/img6/192_168_4_8-2017-05-08_17-45-28_65_74_86.png")
# print "No bee image: "
# print result

## UNCOMMENT THE NEXT 10 LINES TO TEST LOADING AND TESTING AUDIO ON AUDIO ANN
# aud_ann = nw.load("/Users/garrettdimick/5600project/networks/AudioANN.pck")
# result = fit_audio_ann(aud_ann, "/Users/garrettdimick/5600project/BUZZ2Set/test/bee_test/bee2318_192_168_4_9-2017-06-30_13-00-01.wav")
# print "Bee Buzz: "
# print result
# result = fit_audio_ann(aud_ann, "/Users/garrettdimick/5600project/BUZZ2Set/test/cricket_test/cricket45_192_168_4_9-2017-07-31_02-15-01.wav")
# print "Cricket Chirp: "
# print result
# result = fit_audio_ann(aud_ann, "/Users/garrettdimick/5600project/BUZZ2Set/test/noise_test/noise20.wav")
# print "Ambient Noise: "
# print result

## UNCOMMENT THROUGH LINE 161 TO TEST LOADING AND TESTING AUDIO ON AUDIO CONVNET
# input_layer = input_data(shape=[None, 89, 89, 1])
# conv_layer = conv_2d(input_layer,
#                     nb_filter=32,
#                     filter_size=2,
#                     activation='sigmoid',
#                     name='conv_layer_1')
# pool_layer = max_pool_2d(conv_layer, 2, name='pool_layer_2')
# dropout_layer = dropout(pool_layer, 0.25)
# flat_layer = flatten(dropout_layer, name="flatten_layer")
# fc_layer_1 = fully_connected(flat_layer, 256, activation='relu', name='fc_layer_1')
# dropout_layer_1 = dropout(fc_layer_1, 0.25)
# fc_layer_2 = fully_connected(dropout_layer_1, 3, activation='softmax', name='fc_layer_2')
# network = regression(fc_layer_2, optimizer='sgd',
#                     loss='categorical_crossentropy', learning_rate=0.01)
# aud_convnet = tflearn.DNN(network)
# aud_convnet.load("/Users/garrettdimick/5600project/networks/AudioConvnet/AudioConvnet.tfl")
# result = fit_audio_convnet(aud_convnet, "/Users/garrettdimick/5600project/BUZZ2Set/test/bee_test/bee2318_192_168_4_9-2017-06-30_13-00-01.wav")
# print "Bee Buzz: "
# print result
# result = fit_audio_convnet(aud_convnet, "/Users/garrettdimick/5600project/BUZZ2Set/test/cricket_test/cricket45_192_168_4_9-2017-07-31_02-15-01.wav")
# print "Cricket Chirp: "
# print result
# result = fit_audio_convnet(aud_convnet, "/Users/garrettdimick/5600project/BUZZ2Set/test/noise_test/noise20.wav")
# print "Ambient Noise: "
# print result
