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
log_file = 'D:/Diabetic_Retinopathy/finetune_log.csv'

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

SIZE = 200
IMG_WIDTH = SIZE
IMG_HEIGHT = SIZE
CHANNELS = 3
BATCH_SIZE = 16

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

    :param model: A Keras model object
    :param generator: A Keras ImageDataGenerator object

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
    # Custom Keras callback for saving the best model by QWK metric

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

from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from classification_models.keras import Classifiers
import efficientnet.keras as efn

# resnet, preprocess_input = Classifiers.get('resnet18')
# base_model = resnet((IMG_WIDTH, IMG_HEIGHT, 3), weights='imagenet', include_top=False)
# base_model = ResNet50(include_top=False, weights='imagenet')
base_model = efn.EfficientNetB0(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), weights='noisy-student', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(5, activation='relu')(x)
predictions = Dense(1, activation='linear')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# type(base_model)
# model.summary()

from keras import models
# model = models.load_model('D:/Diabetic_Retinopathy/effnet_B0_1.5.h5', custom_objects={'RAdam': RAdam})
# model = models.load_model('D:/Diabetic_Retinopathy/finetune_5.5.h5')

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
csv_logger = CSVLogger(filename=log_file, separator=',', append=True)

model.fit_generator(train_generator,
                    steps_per_epoch=train_generator.samples // BATCH_SIZE,
                    epochs=30,
                    validation_data=val_generator,
                    validation_steps = val_generator.samples // BATCH_SIZE,
                    callbacks=[kappa_metrics, es, rlr, csv_logger])