from __future__ import division, print_function

import os, json
from glob import glob
import numpy as np
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

from keras.utils.data_utils import get_file
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image
from keras.applications import InceptionV3
from keras.models import Model
from keras.callbacks import EarlyStopping, History


class IncepV3():

    def __init__(self):
        self.FILE_PATH = 'http://www.platform.ai/models/'

    def predict(self, imgs, details=False):
        all_preds = self.model.predict(imgs)
        idxs = np.argmax(all_preds, axis=1)
        preds = [all_preds[i, idxs[i]] for i in range(len(idxs))]
        classes = [self.classes[idx] for idx in idxs]
        return np.array(preds), idxs, classes

    def create(self, n_class):
        nf = 128
        p = 0
        self.base_model = base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(3, 360, 640)))
        x = base_model.output
        x = BatchNormalization(axis=1)(x)
        x = Convolution2D(nf,3,3, activation='relu', border_mode='same')(x)
        x = BatchNormalization(axis=1)(x)
        x = MaxPooling2D()(x)
        x = Convolution2D(nf,3,3, activation='relu', border_mode='same')(x)
        x = BatchNormalization(axis=1)(x)
        x = MaxPooling2D()(x)
        x = Convolution2D(nf,3,3, activation='relu', border_mode='same')(x)
        x = BatchNormalization(axis=1)(x)
        x = MaxPooling2D((1,2))(x)
        x = Convolution2D(8,3,3, border_mode='same')(x)
        x = Dropout(p)(x)
        x = GlobalAveragePooling2D()(x)
        predictions= Activation('softmax')(x)
        #x = GlobalAveragePooling2D()(x)
        #x = Dense(1024, activation='relu')(x)
        #x = BatchNormalization()(x)
        #x = Dropout(0.3)(x)
        #predictions = Dense(n_class, activation='softmax')(x)
        self.model = Model(input=base_model.input, output=predictions)


    def get_batches(self, path, gen=image.ImageDataGenerator(shear_range=0.2,
        zoom_range=0.1,
        rotation_range=10.,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True), shuffle=True, batch_size=8, class_mode='categorical'):
        return gen.flow_from_directory(path, target_size=(360,640),
                class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

    def finetune(self, batches, lr):
        for layer in self.model.layers[:172]:
           layer.trainable = False
        for layer in self.model.layers[172:]:
           layer.trainable = True
        self.compile(lr)

    def compile(self, lr=0.001):
        self.model.compile(optimizer=Adam(lr=lr),
                loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self, batches, val_batches, nb_epoch=1):
        callbacks = [EarlyStopping(monitor='val_loss', patience=1, verbose=0)]
        self.model.fit_generator(batches, samples_per_epoch=batches.nb_sample, nb_epoch=nb_epoch,
                validation_data=val_batches, nb_val_samples=val_batches.nb_sample, callbacks=callbacks)


    def test(self, path, batch_size=8):
        test_batches = self.get_batches(path, shuffle=False, batch_size=batch_size, class_mode=None)
        return test_batches, self.model.predict_generator(test_batches, test_batches.nb_sample)

