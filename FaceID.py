# -*- coding: utf-8 -*-

# https://www.cnblogs.com/neo-T/p/6477378.html

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib
from keras import backend as K
from keras.models import Model

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import keras
import loadImg
import model
import history_plot
import dataset_process
import model_logouput
import time
import os
import numpy as np

# ------------------------------- operation sources Settings ----------------------------------------
gpu_no = "0"  # GPU_index
# setting use GPU run
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no
#use_GPU = ['/gpu:0']
#use_GPU = ['/gpu:0','/gpu:1']
n_GPU = '/gpu:0'
I_CPU = '/cpu:0'

# GPU memory limit
#config = tf.ConfigProto(device_count={'gpu':0})
##config.gpu_options.per_process_gpu_memory_fraction = 0.3
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config)
# KTF.set_session(session)

# ---------------------------------------------------------------------------------------------------

# -------------------------------global argumentsettings-------------------------------------------
# select face data class
# ex: face_1
face_class = '4'

# check current project location path
currentPath = os.getcwd()
print("Current Path: " + currentPath)

# figure storage directory
figure_dir = currentPath + '\\plot_figure\\'

# figure classname
figure_classname = 'face_'+face_class

# face image dataSet storage detail root directory
FaceImgDataSet_Path = currentPath + '\\FaceImg\\face_' + face_class + '\\'
print("FaceImg DataSet Path: " + FaceImgDataSet_Path)

# save model and weight name
model_path = os.path.join(currentPath, 'model', str('face_'+face_class))
model_name = str("face_"+face_class+"_FaceID_model.h5")
model_weight_name = str("face_"+face_class+"_FaceID_weight.h5")

# face image file type
faceimg_type = ".jpg"

# img resize
img_resize = 96

# time format settings
time_format = '%Y-%m-%d %H:%M:%S'

# --------------------------------------------------------------------------------------------------


# ------------------------------- train argument settings -----------------------------------------
# train argument settings
batch_size = 23
epochs = 84
verbose = 1
# --------------------------------------------------------------------------------------------------


# -------------------------------model argument settings-------------------------------------------
# you can use these optimizer option: "SGD", "RMSprop", "Adam", "Adagrad",
# "Adadelta"
optimizer = 'Adam'
keras.optimizers.Adam(lr=0.00015, beta_1=0.9, beta_2=0.999,
                      epsilon=None, decay=0.0, amsgrad=False)

# you can use these loss option:
# "categorical_crossentropy","mean_squared_error","mean_absolute_error","cosine_proximity"
loss = 'categorical_crossentropy'

# net layer argument settings
# input layer
inputlayer_Activation = 'relu'
inputlayer_conv2D_hidden_unit = 32
inputlayer_conv2D_kernel_size = (3, 3)
inputlayer_conv2D_padding = 'same'

# one layer
onelayer_conv2D_hidden_unit = 32
onelayer_conv2D_kernel_size = (3, 3)
onelayer_conv2D_padding = 'same'
onelayer_Activation = 'relu'
onelayer_MaxPooling2D_pool_size = (2, 2)
onelayer_Dropout = 0.25

# two layer
twolayer_conv2D_hidden_unit = 64
twolayer_conv2D_kernel_size = (3, 3)
twolayer_conv2D_padding = 'same'
twolayer_Activation = 'relu'
twolayer_MaxPooling2D_pool_size = (2, 2)
twolayer_Dropout = 0.25

# three layer
threelayer_conv2D_hidden_unit = 128
threelayer_conv2D_kernel_size = (3, 3)
threelayer_conv2D_padding = 'same'
threelayer_Activation = 'relu'
threelayer_MaxPooling2D_pool_size = (2, 2)
threelayer_Dropout = 0.25

# four layer
fourlayer_conv2D_hidden_unit = 256
fourlayer_conv2D_kernel_size = (3, 3)
fourlayer_conv2D_padding = 'same'
fourlayer_Activation = 'relu'
fourlayer_MaxPooling2D_pool_size = (2, 2)
fourlayer_Dropout = 0.25

# full-connection layer
full_connectionlayer_Dense = 1024
full_connectionlayer_Activation = 'relu'
full_connectionlayer_Dropout = 0.5

# output layer
ouputlayer_Activation = 'softmax'
# --------------------------------------------------------------------------------------------------

# -------------------------------variable argument declard-----------------------------------------
# img dataset array declard
images = []
labels = []
label_class = []

# train dataset
train_images = None  # x
train_labels = None  # y

# valid dataset
valid_images = None  # x
valid_labels = None  # y

# test dataset
test_images = None  # x
test_labels = None  # y

# model declard
net_model = Model()
history = None

datagen = ImageDataGenerator()

# train history declard
history = None

# --------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    start_time = time.time()

    print("Start FaceID Project.\n")

    # get img dataset
    images, labels, img_width, img_height, channels, n_class, label_class = loadImg.load_faceimg(
        FaceImgDataSet_Path, faceimg_type, img_resize)

    print("\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # show number of class
    print("Number Of Class: {} \n {} \n".format(n_class, label_class))
    # show img dataset length
    print("Images Length: {} \nLabels Length: {}".format(len(images), len(labels)))

    # show img size
    print("Image Size: {}x{}x{}".format(img_width, img_height, channels))

    images = np.array(images)
    labels = np.array(labels)

    # shuffle data
    perm_array = np.arange(len(images))
    np.random.shuffle(perm_array)
    images = images[perm_array]
    labels = labels[perm_array]

    # print(images.shape)
    # print(labels.shape)
    print(labels)

    label_class = enumerate(label_class)
    # char convert to int code
    labels = loadImg.char_to_code(labels, label_class, n_class)

    print(labels)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    # dataset process generate
    with tf.device(n_GPU):
        train_images, valid_images, test_images, train_labels, valid_labels, test_labels = dataset_process.process_data(
            images, labels, img_height, img_width, channels, n_class)
        # print(train_images.shape[0])

    # build Neural Network Model
    with tf.device(n_GPU):
        net_model = model.build_model(img_width, img_height, channels, n_class)
        modellogouput = model_logouput.model_logouput(str('face_'+face_class))
        str_log = '''

            This is  model each layer argument recoard.

            # input layer
            inputlayer_Activation = {}
            inputlayer_conv2D_hidden_unit = {}
            inputlayer_conv2D_kernel_size = {}
            inputlayer_conv2D_padding = {}

            # one layer
            onelayer_conv2D_hidden_unit = {}
            onelayer_conv2D_kernel_size = {}
            onelayer_conv2D_padding = {}
            onelayer_Activation = {}
            onelayer_MaxPooling2D_pool_size = {}
            onelayer_Dropout = {}

            # two layer
            twolayer_conv2D_hidden_unit = {}
            twolayer_conv2D_kernel_size = {}
            twolayer_conv2D_padding = {}
            twolayer_Activation = {}
            twolayer_MaxPooling2D_pool_size = {}
            twolayer_Dropout = {}

            # three layer
            threelayer_conv2D_hidden_unit = {}
            threelayer_conv2D_kernel_size = {}
            threelayer_conv2D_padding = {}
            threelayer_Activation = {}
            threelayer_MaxPooling2D_pool_size = {}
            threelayer_Dropout = {}

            # four layer
            fourlayer_conv2D_hidden_unit = {}
            fourlayer_conv2D_kernel_size = {}
            fourlayer_conv2D_padding = {}
            fourlayer_Activation = {}
            fourlayer_MaxPooling2D_pool_size = {}
            fourlayer_Dropout = {}

            # full-connection layer
            full_connectionlayer_Dense = {}
            full_connectionlayer_Activation = {}
            full_connectionlayer_Dropout = {}

            # output layer
            ouputlayer_Activation = {}
            optimizer = {}
            loss = {}

        '''.format(inputlayer_Activation, inputlayer_conv2D_hidden_unit, inputlayer_conv2D_kernel_size, inputlayer_conv2D_padding,
                   onelayer_conv2D_hidden_unit, onelayer_conv2D_kernel_size, onelayer_conv2D_padding, onelayer_Activation, onelayer_MaxPooling2D_pool_size, onelayer_Dropout,
                   twolayer_conv2D_hidden_unit, twolayer_conv2D_kernel_size, twolayer_conv2D_padding, twolayer_Activation, onelayer_MaxPooling2D_pool_size, twolayer_Dropout,
                   threelayer_conv2D_hidden_unit, threelayer_conv2D_kernel_size, threelayer_conv2D_padding, threelayer_Activation, threelayer_MaxPooling2D_pool_size, threelayer_Dropout,
                   fourlayer_conv2D_hidden_unit, fourlayer_conv2D_kernel_size, fourlayer_conv2D_padding, fourlayer_Activation, fourlayer_MaxPooling2D_pool_size, fourlayer_Dropout,
                   full_connectionlayer_Dense, full_connectionlayer_Activation, full_connectionlayer_Dropout,
                   ouputlayer_Activation, optimizer, loss)

        modellogouput.outputmodellog(str_log)

    # ImageDataGenerator declard and argument settings
    # https://chtseng.wordpress.com/2017/11/11/data-augmentation-%E8%B3%87%E6%96%99%E5%A2%9E%E5%BC%B7/
    datagen = ImageDataGenerator(featurewise_center=False,
                                 samplewise_center=False,
                                 featurewise_std_normalization=False,
                                 samplewise_std_normalization=False,
                                 zca_whitening=False,
                                 rotation_range=20,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 horizontal_flip=True,
                                 vertical_flip=False)

    print("\n\n")
    print(device_lib.list_local_devices())

    print("\n\nWait for configure operation sources...... \n\n")

    # process image computing of
    # "featurewise_center","featurewise_std_normalization","zca_whitening"
    datagen.fit(train_images)

    # start train
    with tf.device(n_GPU):

        steps_per_epoch = (len(train_images) / batch_size)
        # steps_per_epoch = len(train_images)

        history = net_model.fit_generator(datagen.flow(train_images, train_labels, batch_size=batch_size),
                                          steps_per_epoch=steps_per_epoch,
                                          epochs=epochs,
                                          verbose=verbose,
                                          validation_data=(valid_images, valid_labels))

    # print(history.history)

    # evaluate model
    score = net_model.evaluate(test_images, test_labels, verbose=1)
    print("\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("Keras CNN - accuracy: {:.2f}".format(score[1]))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    # create figure
    history_plot.plot_figure(history, figure_dir, figure_classname)

    print("\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("Model_Weigth Save Path: \n{}".format(model_weight_name))
    print("Model Save Path: \n{}".format(model_name))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    # check model save path is exist
    if not os.path.exists(model_path):

        os.mkdir(model_path)

    # save model and weight
    net_model.save_weights(os.path.join(model_path, model_weight_name))
    net_model.save(os.path.join(model_path, model_name))

    end_time = '{:.2f}'.format((time.time() - start_time))
    print("\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("Speed Time: {}s".format(end_time))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
