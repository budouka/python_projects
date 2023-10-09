#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 10:53:35 2023

@author: pthompson
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# data comes from the fashion MNIST public dataset
# it can be downloaded e.g. here:
# https://www.kaggle.com/datasets/zalando-research/fashionmnist

# read in CSV data
filepath_train = '/Users/pthompson/Downloads/Fashion_MNIST/fashion-mnist_train.csv'
filepath_test = '/Users/pthompson/Downloads/Fashion_MNIST/fashion-mnist_test.csv'
data_df = pd.read_csv(filepath_train, header=0)
train_labels = data_df.iloc[:,0]
train_labels = train_labels.to_numpy()
train_images = data_df.iloc[:,1:]
train_images = train_images.values.reshape(-1, 28, 28, 1)


#modeling

data_augmentation = keras.Sequential(
  [
   layers.RandomFlip("horizontal",input_shape=(28,28,1)), 
   layers.RandomTranslation(height_factor = 0.2,
                             width_factor = 0.2),
   layers.RandomRotation(0.2),
   layers.RandomZoom(0.2)
  ]
)



model = Sequential([
    data_augmentation,
    layers.Rescaling(scale=1./255),
    layers.Conv2D(16,3,activation='relu',padding='same'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3,activation='relu',padding='same'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(512,activation='relu'),
    layers.Dense(10,activation="softmax")
    ])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.summary()

history = model.fit(train_images,train_labels,validation_split=0.2,epochs=10)

# Plot the model results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()



# data augmentation using a data generator
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# read in CSV data
filepath_train = '/Users/pthompson/Downloads/Fashion_MNIST/fashion-mnist_train.csv'
filepath_test = '/Users/pthompson/Downloads/Fashion_MNIST/fashion-mnist_test.csv'
data_df = pd.read_csv(filepath_train, header=0)
test_df = pd.read_csv(filepath_test, header=0)
df_sample = pd.DataFrame(np.random.randn(len(data_df),2))
mask = np.random.rand(len(data_df)) < 0.2
train_df = data_df[~mask]
val_df = data_df[mask]

train_labels = train_df.iloc[:,0]
train_labels = train_labels.to_numpy()
train_images = train_df.iloc[:,1:]
train_images = train_images.values.reshape(-1, 28, 28, 1)

val_labels = val_df.iloc[:,0]
val_labels = val_labels.to_numpy()
val_images = val_df.iloc[:,1:]
val_images = val_images.values.reshape(-1, 28, 28, 1)

test_labels = test_df.iloc[:,0]
test_labels = test_labels.to_numpy()
test_images = test_df.iloc[:,1:]
test_images = test_images.values.reshape(-1, 28, 28, 1)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)


# Optional: add image augmentations (for better generalizability at the cost of slower learning)
# Apply data augmentation
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow(
        train_images,  # This is the source directory for training images
        y=train_labels,
        #target_size=(28, 28),  # All images will be resized to 150x150
        batch_size=128,
        # Since you used binary_crossentropy loss, you need binary labels
        )

# Flow training images in batches of 128 using train_datagen generator
validation_generator = validation_datagen.flow(
        val_images,  # This is the source directory for training images
        y=val_labels,
        batch_size=32,
        # Since you used binary_crossentropy loss, you need binary labels
        )


model = Sequential([
    layers.Conv2D(16,3,activation='relu',padding='same',input_shape=(28,28,1)),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3,activation='relu',padding='same'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(512,activation='relu'),
    layers.Dense(10,activation="softmax")
    ])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.summary()

history = model.fit(train_generator,
                    validation_data = validation_generator,
                    batch_size=128,
                    epochs=10)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()