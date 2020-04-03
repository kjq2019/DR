#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import random as rn
import pandas as pd
import os

# Machine Learning
import tensorflow as tf
import keras
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import cohen_kappa_score

# Path specifications
KAGGLE_DIR = 'D:/Diabetic_Retinopathy/'
TRAIN_DF_PATH = KAGGLE_DIR + "train.csv"
TRAIN_IMG_PATH = KAGGLE_DIR + "train_images/"

# Specify title of our final model
SAVED_MODEL_NAME = 'D:/Diabetic_Retinopathy/finetune.h5'

# Set seed for reproducability
seed = 1234
rn.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

train_df = pd.read_csv(TRAIN_DF_PATH)
train_df['id_code'] = train_df['id_code'] + ".png"
print(f"Training images: {train_df.shape[0]}")

# train_df['diagnosis']

SIZE = 384
IMG_WIDTH = SIZE
IMG_HEIGHT = SIZE
CHANNELS = 3
BATCH_SIZE = 5

# y_labels = train_df['diagnosis'].values

train_datagen = ImageDataGenerator(rotation_range=360,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   validation_split=0.15,
                                   # preprocessing_function=preprocess_image,
                                   rescale=1 / 255.)

train_generator = train_datagen.flow_from_dataframe(train_df,
                                                    x_col='id_code',
                                                    y_col='diagnosis',
                                                    directory = TRAIN_IMG_PATH,
                                                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='other',
                                                    subset='training')

val_generator = train_datagen.flow_from_dataframe(train_df,
                                                  x_col='id_code',
                                                  y_col='diagnosis',
                                                  directory = TRAIN_IMG_PATH,
                                                  target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                  batch_size=BATCH_SIZE,
                                                  class_mode='other',
                                                  subset='validation')

def get_preds_and_labels(model, generator):
    """
    Get predictions and labels from the generator

    :param model: A keras model object
    :param generator: A keras ImageDataGenerator object

    :return: A tuple with two Numpy Arrays
    One containing the predictions, and one containing the labels
    """
    preds = []
    labels = []
    for _ in range(int(np.ceil(generator.samples / BATCH_SIZE))):
        x, y = next(generator)
        preds.append(model.predict(x))
        labels.append(y) # 估计y是int
    # Flatten list of numpy arrays
    return np.concatenate(preds).ravel(), np.concatenate(labels).ravel()


class Metrics(Callback):
    # Custom keras callback for saving the best model by QWK metric

    def on_train_begin(self, logs={}):
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
        """
        Gets QWK score on the validation data

        :param epoch: The current epoch number
        """
        # Get predictions and convert to integers
        y_pred, labels = get_preds_and_labels(model, val_generator)
        y_pred = np.rint(y_pred).astype(np.uint8).clip(0, 4)
        # We can use sklearns implementation of QWK straight out of the box
        # as long as we specify weights as 'quadratic'
        _val_kappa = cohen_kappa_score(labels, y_pred, weights='quadratic')
        self.val_kappas.append(_val_kappa)
        print(f"val_kappa: {round(_val_kappa, 4)}")
        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save(SAVED_MODEL_NAME)
        return

# type(base_model)
# model.summary()

from keras import layers, models, Input, Model

def _block(x, n_out, n, init_strides=(2, 2)):  # n_out是4*filters数
    h_out = n_out // 4
    out = _bottleneck(x, h_out, n_out, strides=init_strides)
    for i in range(1, n):
        out = _bottleneck(out, h_out, n_out)
    return out

def _bottleneck(x, h_out, n_out, strides=None):
    n_in = x.get_shape()[-1]
    #     print(n_in, n_out)
    if strides is None:
        strides = (1, 1) if n_in == h_out else (2, 2)
    h = layers.ZeroPadding2D(padding=(1, 1))(x)
    h = layers.Conv2D(h_out, (3, 3), strides=strides)(h)
    h = layers.BatchNormalization()(h)
    h = layers.Activation('relu')(h)
    h = layers.ZeroPadding2D(padding=(1, 1))(h)
    h = layers.Conv2D(h_out, (3, 3), strides=(1, 1))(h)
    h = layers.BatchNormalization()(h)
    #     print('h:', h.get_shape())
    if n_in != h_out:  # 判断支路output是否需要resize（主路？
        shortcut = layers.ZeroPadding2D(padding=(1, 1))(x)
        shortcut = layers.Conv2D(h_out, (3, 3), strides=strides)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    else:
        shortcut = x
    #     h = Lambda(lambda h: h + shortcut)(h)
    h = layers.Add()([h, shortcut])
    #     h = merge([h, shortcut], mode="sum")
    return layers.Activation('relu')(h)

#     x = tf.random_normal([32, 224, 224, 3]) # met a tensor for debug

def ResNet_self(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), classes=5):
    # resnet bottom
    img_input = Input(shape=input_shape)
    x = layers.BatchNormalization()(img_input)
    print(type(x))
    x = layers.ZeroPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(16, (7, 7), strides=(2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    # print(x.get_shape())

    x = _block(x, 64, 3, init_strides=(1, 1))  # -> [batch, 56, 56, 256]
    x = _block(x, 128, 4)  # -> [batch, 28, 28, 512]
    x = _block(x, 256, 6)  # -> [batch, 14, 14, 1024]
    x = _block(x, 512, 3)  # -> [batch, 14, 14, 1024]
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(5, activation='relu')(x)
    x = layers.Dense(1, activation="linear")(x)
    model = Model(img_input, x)
    return model

model = ResNet_self()

from keras import models
# model = models.load_model('D:/Diabetic_Retinopathy/effnet_B0_1.5.h5', custom_objects={'RAdam': RAdam})
# model = models.load_model('D:/Diabetic_Retinopathy/finetune_7_= =_64.h5')

model.compile(optimizer=keras.optimizers.Adam(lr=0.0005), loss='mse', metrics = ['accuracy'])

# For tracking Quadratic Weighted Kappa score
kappa_metrics = Metrics()
# Monitor MSE to avoid overfitting and save best model
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=12)
rlr = ReduceLROnPlateau(monitor='val_loss',
                        factor=0.5,
                        patience=4,
                        verbose=1,
                        mode='auto',
                        epsilon=0.0001)

from keras.callbacks import CSVLogger
csv_logger = CSVLogger(filename='D:/Diabetic_Retinopathy/finetune_log.csv', separator=',', append=True)

model.fit_generator(train_generator,
                    steps_per_epoch=train_generator.samples // BATCH_SIZE,
                    epochs=30,
                    validation_data=val_generator,
                    validation_steps = val_generator.samples // BATCH_SIZE,
                    callbacks=[kappa_metrics, es, rlr, csv_logger])




















