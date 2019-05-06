from keras.models import Sequential , Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from keras.models import load_model

import FaceID
import tensorflow as tf
import random
import keras
import numpy as np

def build_model(img_width,img_height,channels,n_class):

    input_shape = (img_width,img_height,channels)

    model = Sequential()
    
    # input layer
    model.add(Conv2D(FaceID.inputlayer_conv2D_hidden_unit,kernel_size=FaceID.inputlayer_conv2D_kernel_size,padding=FaceID.inputlayer_conv2D_padding,input_shape = input_shape))
    model.add(Activation(FaceID.inputlayer_Activation))

    # one hiddenlayer (Conv2D,Activation,MaxPooling2D,Dropout)
    model.add(Conv2D(FaceID.onelayer_conv2D_hidden_unit, FaceID.onelayer_conv2D_kernel_size,padding=FaceID.onelayer_conv2D_padding))
    model.add(Activation(FaceID.onelayer_Activation))
    model.add(MaxPooling2D(pool_size=FaceID.onelayer_MaxPooling2D_pool_size))
    model.add(Dropout(FaceID.onelayer_Dropout))

    # two hiddenlayer(Conv2D,Activation,MaxPooling2D,Dropout)
    model.add(Conv2D(FaceID.twolayer_conv2D_hidden_unit, FaceID.twolayer_conv2D_kernel_size,padding = FaceID.twolayer_conv2D_padding))
    model.add(Activation(FaceID.twolayer_Activation))
    model.add(MaxPooling2D(pool_size=FaceID.twolayer_MaxPooling2D_pool_size))
    model.add(Dropout(FaceID.twolayer_Dropout))

    # three hiddenlayer(Conv2D,Activation,MaxPooling2D,Dropout)
    model.add(Conv2D(FaceID.threelayer_conv2D_hidden_unit, FaceID.threelayer_conv2D_kernel_size, padding=FaceID.threelayer_conv2D_padding))
    model.add(Activation(FaceID.threelayer_Activation))
    model.add(MaxPooling2D(pool_size=FaceID.threelayer_MaxPooling2D_pool_size))
    model.add(Dropout(FaceID.threelayer_Dropout))

    # four hiddenlayer(Conv2D,Activation,MaxPooling2D,Dropout)
    model.add(Conv2D(FaceID.fourlayer_conv2D_hidden_unit, FaceID.fourlayer_conv2D_kernel_size,padding=FaceID.fourlayer_conv2D_padding))
    model.add(Activation(FaceID.fourlayer_Activation))
    model.add(MaxPooling2D(pool_size=FaceID.fourlayer_MaxPooling2D_pool_size))
    model.add(Dropout(FaceID.fourlayer_Dropout))

    # Flatten layer
    model.add(Flatten())

    # full-connection layer(Dense,Activation,Dropout)
    model.add(Dense(FaceID.full_connectionlayer_Dense))
    model.add(Activation(FaceID.full_connectionlayer_Activation))
    model.add(Dropout(FaceID.full_connectionlayer_Dropout)) 

    # output layer
    model.add(Dense(n_class))
    model.add(Activation(FaceID.ouputlayer_Activation))

    model.compile(loss=FaceID.loss,optimizer=FaceID.optimizer,metrics=['accuracy'])

    model.summary()

    return model