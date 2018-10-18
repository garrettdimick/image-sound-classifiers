#################
## Garrett Dimick
## A01227378
## Tests and Trains four neural networks:
## One ANN and one Convnet for classifying Images
## and one ANN and one Convnet for classifying Sounds
## Datasets were gathered by Professor Vladimir Kulyukin of Utah State University
#################
####train

import cv2
import numpy as np
import glob
import random
import scipy as sp
import network as nw
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d, conv_1d, max_pool_1d
from tflearn.layers.estimator import regression
import pickle as cPickle
from scipy.io import wavfile
from tensorflow import reset_default_graph

## condenses a list into a list that is 1/25 the size by averaging 100 elements
## at a time
def condense_data(d):
    return [np.mean(d[i:i+10]) for i in xrange(0,len(d),10)][0:7921]
# use save to persist networks
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
## load_image_data returns all the image data necessary for constructing
## an ANN and a Convnet and returns it in a list with the following structure
## use the indexes to get the data needed
## returns [0-ann_image_train_data, 1-image_trainX, 2-image_trainY, 3-ann_image_test_data,
## 4-ann_image_eval_data, 5-image_testX, 6-image_testY, 7-image_evalX, 8-image_evalY]
def load_image_data():
    ann_train_data = []
    convnet_train_pre_data = []
    ## Process all the Bee Training Data
    ## first add the data from the bee_train, 19,082 images
    f = glob.glob("/Users/garrettdimick/Google Drive/Fall 2018/CS5600-AI/Project_01_Data/BEE2Set/bee_train/*/*.png")
    for file in f:
        img = cv2.imread(file)
        convnet_train_pre_data.append((img/float(255), np.array([1, 0])))
        ## Grayscale the images for the ANN
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ann_train_data.append((gray_img/float(255), 1))
    ## now the no bee from no_bee_train, 19,057 images
    f = glob.glob("/Users/garrettdimick/Google Drive/Fall 2018/CS5600-AI/Project_01_Data/BEE2Set/no_bee_train/*/*.png")
    for file in f:
        img = cv2.imread(file)
        convnet_train_pre_data.append((img/float(255), np.array([0, 1])))
        ## Grayscale the images for the ANN
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ann_train_data.append((gray_img/float(255), 0))

    ## reshape the data for better use in the ANN
    atd = [np.reshape(x[0], (1024, 1)) for x in ann_train_data]
    ard = [y[1] for y in ann_train_data]
    ## training data for ANN that will classify images
    # ann_image_train_data = zip(atd, ard)
    ann_image_train_data = []
    for i in xrange(len(atd)):
        ann_image_train_data.append((atd[i], ard[i]))
    # print type(ann_image_train_data)
    # print type(ann_image_train_data[0])
    # print
    # print ann_image_train_data[0][0]

    shuffle(ann_image_train_data)
    ## turn the convnet data into a usable format for tflearn stuff
    ## x is training data, y is labels
    x = [i[0] for i in convnet_train_pre_data]
    y = [i[1] for i in convnet_train_pre_data]
    x = np.array(x)
    y = np.array(y)
    image_trainX, image_trainY = shuffle(x, y)
    ## reshape the data
    image_trainX = image_trainX.reshape([-1, 32, 32, 3])

    ####### Now test data
    ann_test_data = []
    convnet_test_pre_data = []
    ## Get test data for the image ANN and convnet
    ## Process all the Bee Test Data
    ## first add the data from the bee_test
    f = glob.glob("/Users/garrettdimick/Google Drive/Fall 2018/CS5600-AI/Project_01_Data/BEE2Set/bee_test/*/*.png")
    for file in f:
        img = cv2.imread(file)
        convnet_test_pre_data.append((img/float(255), np.array([1, 0])))
        ## Grayscale the images for the ANN
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ann_test_data.append((gray_img/float(255), 1))
    ## now the no bee from no_bee_train, 19,057 images
    f = glob.glob("/Users/garrettdimick/Google Drive/Fall 2018/CS5600-AI/Project_01_Data/BEE2Set/no_bee_test/*/*.png")
    for file in f:
        img = cv2.imread(file)
        convnet_test_pre_data.append((img/float(255), np.array([0, 1])))
        ## Grayscale the images for the ANN
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ann_test_data.append((gray_img/float(255), 0))

    ## 12724 files
    ## reshape the data for better use in the ANN
    ann_test_d = [np.reshape(x[0], (1024, 1)) for x in ann_test_data]
    ann_res_d = [y[1] for y in ann_test_data]
    ann_image_testing_data = zip(ann_test_d, ann_res_d)
    shuffle(ann_image_testing_data)
    ann_image_test_data = ann_image_testing_data[0:11000]
    ## eval data for ANN that will classify images
    ann_image_eval_data = ann_image_testing_data[11000:]

    ## turn the convnet data into a usable format for tflearn stuff
    x = [i[0] for i in convnet_test_pre_data]
    y = [i[1] for i in convnet_test_pre_data]
    x = np.array(x)
    y = np.array(y)
    testX = x[0:11000]
    image_evalX = x[11000:]
    testY = y[0:11000]
    image_evalY = y[11000:]
    image_testX, image_testY = shuffle(testX, testY)
    image_evalX, image_evalY = shuffle(image_evalX, image_evalY)

    ## reshape the data
    image_testX = image_testX.reshape([-1, 32, 32, 3])
    return [ann_image_train_data, image_trainX, image_trainY, ann_image_test_data,
            ann_image_eval_data, image_testX, image_testY, image_evalX, image_evalY]

## now turn data into numpy arrays for sounds
## load_sound_data loads the sounds data and returns it as a list with the format:
## [0-sound_train_data, 1-sound_test_data, 2-trainX, 3-trainY,
## 4-testX, 5-testY, 6-validX, 7-validY]
def load_sound_data():
    sound_t_data = []
    sound_train_data = []
    ## Process all the Bee Training sound data
    ## first add bee train sounds
    f = glob.glob("/Users/garrettdimick/Google Drive/Fall 2018/CS5600-AI/Project_01_Data/BUZZ2Set/train/bee_train/*.wav")
    for file in f:
        samplerate, audio = wavfile.read(file)
        sound_t_data.append(condense_data(audio/float(np.max(audio))))
    y_train = [np.array([1, 0, 0]) for i in sound_t_data]
    bee_train_data = zip(sound_t_data, y_train)
    for i in bee_train_data:
        sound_train_data.append(i)

    ## now cricket sounds
    sound_t_data = []
    f = glob.glob("/Users/garrettdimick/Google Drive/Fall 2018/CS5600-AI/Project_01_Data/BUZZ2Set/train/cricket_train/*.wav")
    for file in f:
        samplerate, audio = wavfile.read(file)
        sound_t_data.append(condense_data(audio/float(np.max(audio))))
    y_train = [np.array([0, 1, 0]) for i in sound_t_data]
    cricket_train_data = zip(sound_t_data, y_train)
    for i in cricket_train_data:
        sound_train_data.append(i)

    ## now ambient Sounds
    sound_t_data = []
    f = glob.glob("/Users/garrettdimick/Google Drive/Fall 2018/CS5600-AI/Project_01_Data/BUZZ2Set/train/noise_train/*.wav")
    for file in f:
        samplerate, audio = wavfile.read(file)
        sound_t_data.append(condense_data(audio/float(np.max(audio))))
    y_train = [np.array([0, 0, 1]) for i in sound_t_data]
    noise_train_data = zip(sound_t_data, y_train)
    for i in noise_train_data:
        sound_train_data.append(i)

    ## Now testing data for sounds
    sound_t_data = []
    sound_test_data = []
    f = glob.glob("/Users/garrettdimick/Google Drive/Fall 2018/CS5600-AI/Project_01_Data/BUZZ2Set/test/bee_test/*.wav")
    for file in f:
        samplerate, audio = wavfile.read(file)
        sound_t_data.append(condense_data(audio/float(np.max(audio))))
    y_test = [np.array([1, 0, 0]) for i in sound_t_data]
    bee_test_data = zip(sound_t_data, y_test)
    for i in bee_test_data:
        sound_test_data.append(i)
    ## now cricket sounds
    sound_t_data = []
    f = glob.glob("/Users/garrettdimick/Google Drive/Fall 2018/CS5600-AI/Project_01_Data/BUZZ2Set/test/cricket_test/*.wav")
    for file in f:
        samplerate, audio = wavfile.read(file)
        sound_t_data.append(condense_data(audio/float(np.max(audio))))
    y_test = [np.array([0, 1, 0]) for i in sound_t_data]
    cricket_test_data = zip(sound_t_data, y_test)
    for i in cricket_test_data:
        sound_test_data.append(i)
    ## now ambient Sounds
    sound_t_data = []
    f = glob.glob("/Users/garrettdimick/Google Drive/Fall 2018/CS5600-AI/Project_01_Data/BUZZ2Set/test/noise_test/*.wav")
    for file in f:
        samplerate, audio = wavfile.read(file)
        sound_t_data.append(condense_data(audio/float(np.max(audio))))
    y_test = [np.array([0, 0, 1]) for i in sound_t_data]
    noise_test_data = zip(sound_t_data, y_test)
    for i in noise_test_data:
        sound_test_data.append(i)

    trainX = [i[0] for i in sound_train_data]
    trainX = np.array(trainX)
    trainX = trainX.reshape([-1, 89, 89, 1])
    trainY = [i[1] for i in sound_train_data]
    trainY = np.array(trainY)

    testX = [i[0] for i in sound_test_data]
    testX = np.array(testX)
    testx = testX.reshape([-1, 89, 89, 1])
    testY = [i[1] for i in sound_test_data]
    testy = np.array(testY)

    testX = testx[0:2100]
    validX = testx[2100:]
    testY = testy[0:2100]
    validY = testy[2100:]

    return [sound_train_data, sound_test_data, trainX, trainY, testX, testY, validX, validY]
## Build an image classifying ANN
## Architecture: 1024 x  x 2
## 3 hidden layers
## initially create the ANN, after this comment this out and the train_image_ann()
## routine will take care of loading the ann and then persisting it in the same
## location

##[0-ann_image_train_data, 1-image_trainX, 2-image_trainY, 3-ann_image_test_data,
## 4-ann_image_eval_data, 5-image_testX, 6-image_testY, 7-image_evalX, 8-image_evalY]
image_ann = nw.Network([1024, 40, 20, 2])
image_ann.save("/Users/garrettdimick/Google Drive/Fall 2018/CS5600-AI/Project/networks/ImageANN.pck")
def train_image_ann(filepath):
    ## use stochastic gradient descent to train the ANN
    ## 10 epochs, mbs=5, eta=0.1, lmbda=0.0
    data = load_image_data()
    ann_image_train_data = data[0]
    ann_image_test_data = data[3]
    image_ann = nw.load(filepath)
    num_epochs = 10
    mbs = 1
    eta = 0.1
    lmbda = 0.05
    ## Train the ANN using Stochastic Gradient descent
    image_ann.SGD2(ann_image_train_data, num_epochs, mbs, eta, lmbda,
                            ann_image_test_data,
                            monitor_evaluation_cost=True,
                            monitor_evaluation_accuracy=True,
                            monitor_training_cost=True,
                            monitor_training_accuracy=True)
    image_ann.save(filepath)

## [0-sound_train_data, 1-sound_test_data, 2-trainX, 3-trainY,
## 4-testX, 5-testY, 6-validX, 7-validY]
audio_ann = nw.Network([7921, 387, 60, 15, 3])
audio_ann.save("/Users/garrettdimick/Google Drive/Fall 2018/CS5600-AI/Project/networks/AudioANN.pck")
def train_sound_ann(filepath):
    ## use stochastic gradient descent to train the ANN
    ## 10 epochs, mbs=5, eta=0.1, lmbda=0.0
    data = load_sound_data()
    ann_audio_train_data = data[0]
    ann_audio_eval_data = data[1]
    audio_ann = nw.load(filepath)
    num_epochs = 10
    mbs = 1
    eta = 0.1
    lmbda = 0.05
    ## Train the ANN using Stochastic Gradient descent
    audio_ann.SGD2(ann_audio_train_data, num_epochs, mbs, eta, lmbda,
                            ann_audio_eval_data,
                            monitor_evaluation_cost=True,
                            monitor_evaluation_accuracy=True,
                            monitor_training_cost=True,
                            monitor_training_accuracy=True)
    audio_ann.save(filepath)

def train_image_convnet(filepath):
    data = load_image_data()
    trainX = data[1]
    trainY = data[2]
    testX = data[5]
    testY = data[6]
    ## design the network
    ## input layer is 32 x 32
    input_layer = input_data(shape=[None, 32, 32, 3])
    ## convolutional layer
    conv_layer = conv_2d(input_layer,
                        nb_filter=32,
                        filter_size=3,
                        activation='relu',
                        name='conv_layer_1')
    ## max pooling layer
    pool_layer = max_pool_2d(conv_layer, 2, name='pool_layer_1')
    ## convolutional layer
    # conv_layer_2 = conv_2d(pool_layer,
    #                     nb_filter=64,
    #                     filter_size=3,
    #                     activation='relu',
    #                     name='conv_layer_2')
    # ## convolutional layer
    # conv_layer_3 = conv_2d(conv_layer_2,
    #                     nb_filter=64,
    #                     filter_size=3,
    #                     activation='relu',
    #                     name='conv_layer_3')
    # ## max pooling layer, window of 2x2
    # pool_layer_2 = max_pool_2d(conv_layer_3, 2, name='pool_layer_2')
    ## fully connected layer with 512 nodes, half of 32 x 32
    fc_layer_1 = fully_connected(pool_layer, 512, activation='relu', name='fc_layer_1')
    ## dropout layer
    dropout_layer = dropout(fc_layer_1, 0.5)
    ## fully connected with 2 layers, 0 is not a bee, 1 is a bee
    fc_layer_2 = fully_connected(dropout_layer, 2, activation='softmax', name='fc_layer_2')
    ## network is trained with sgd, categorical cross entropy loss function, and eta = 0.01
    network = regression(fc_layer_2, optimizer='sgd',
                        loss='categorical_crossentropy', learning_rate=0.01)
    ## turn the network into a model
    model = tflearn.DNN(network)
    ## now do the training on the network
    NUM_EPOCHS = 50
    BATCH_SIZE = 32
    model.fit(trainX, trainY, n_epoch=NUM_EPOCHS,
                shuffle=True,
                validation_set=(testX, testY),
                show_metric=True,
                batch_size = BATCH_SIZE,
                run_id='image_convnet')
    model.save(filepath)

## [0-sound_train_data, 1-sound_test_data, 2-trainX, 3-trainY,
## 4-testX, 5-testY, 6-validX, 7-validY]
def train_sound_convnet(filepath):
    data = load_sound_data()
    trainX = data[2]
    trainY = data[3]
    testX = data[4]
    testY = data[5]

    ## design the network
    ## input is a 89 x 89 nparray reshaped from the original 1 x 7921
    ## with the audio information
    input_layer = input_data(shape=[None, 89, 89, 1])
    ## convolutional layer
    conv_layer = conv_2d(input_layer,
                        nb_filter=32,
                        filter_size=2,
                        activation='sigmoid',
                        name='conv_layer_1')
    ## max pooling layer, window of 2x2
    pool_layer = max_pool_2d(conv_layer, 2, name='pool_layer_2')
    ## dropout layer
    dropout_layer = dropout(pool_layer, 0.25)
    flat_layer = flatten(dropout_layer, name="flatten_layer")
    ## fully connected layer with 128
    fc_layer_1 = fully_connected(flat_layer, 256, activation='relu', name='fc_layer_1')
    ## dropout layer
    dropout_layer_1 = dropout(fc_layer_1, 0.25)
    ## fully connected with 3 layers, 0 is bee, 1 is cricket, 2 is ambient
    fc_layer_2 = fully_connected(dropout_layer_1, 3, activation='softmax', name='fc_layer_2')
    ## network is trained with sgd, categorical cross entropy loss function, and eta = 0.01
    network = regression(fc_layer_2, optimizer='sgd',
                        loss='categorical_crossentropy', learning_rate=0.01)
    ## turn the network into a model
    model = tflearn.DNN(network)
    ## now do the training on the network
    NUM_EPOCHS = 100
    BATCH_SIZE = 1
    model.fit(trainX, trainY, n_epoch=NUM_EPOCHS,
                shuffle=True,
                validation_set=(testX, testY),
                show_metric=True,
                batch_size = BATCH_SIZE,
                run_id='audio_convnet')
    model.save(filepath)

### Load the data using these functions, each train function calls these, so
## it is only necessary if you want to see the data for some other reason
# load_image_data()
# load_sound_data()

## uncomment the following function calls to train a network in the given path
## Train the networks
## Image ANN
# train_image_ann("/Users/garrettdimick/Google Drive/Fall 2018/CS5600-AI/Project/networks/ImageANN.pck")

## Image Convnet
# reset_default_graph()
# train_image_convnet("/Users/garrettdimick/Google Drive/Fall 2018/CS5600-AI/Project/networks/ImageConvnet/ImageConvnet.tfl")

## Sound ANN
train_sound_ann("/Users/garrettdimick/Google Drive/Fall 2018/CS5600-AI/Project/networks/AudioANN.pck")

## Sound Convnet
# reset_default_graph()
# train_sound_convnet("/Users/garrettdimick/Google Drive/Fall 2018/CS5600-AI/Project/networks/AudioConvnet/AudioConvnet.tfl")
