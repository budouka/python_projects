#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of image classification with convolutional neural networks

Using flower photos public dataset
"""


# SETUP

import matplotlib.pyplot as plt
import tensorflow as tf





# PREPROCESSING

# We download the public flower photos dataset
import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

# we set some parameters for handling the images
batch_size = 32
img_height = 180
img_width = 180

"""
The data contains images of different flowers. 
There are 5 classes: daisies, dandelions, roses, sunflowers, and tulips
The folder structure of the downloaded directory reflects these classes
i.e. there are 5 folders, one for each class. The label is inferred from the folder
"""


"""
We use TensorFlow's image data generator
it allows for augmenting the training data to achieve better generalizability
by creating additional, artificial training images.

We need to set up a separate validation data generator, because the image 
augmentation should only happen on the training data.

This is somewhat complicated by the fact that TensorFlow to date doesn't feature
image augmentation on a subset of data within an image generator.
The workaround implemented here is to use an identical seed that ensures
identical train-test-split results when invoking the two separate generators.
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen_train = ImageDataGenerator(
rescale=1./255,
validation_split = 0.2,
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest')


datagen_val = ImageDataGenerator( rescale=1./255, validation_split=0.2)    


train_generator = datagen_train.flow_from_directory(
        data_dir, 
        seed = 204, #this ensures that train and test sets are split identically
        target_size=(180, 180),
        batch_size=128,
        subset = 'training',
        class_mode='categorical')


val_generator = datagen_val.flow_from_directory(
        data_dir, 
        seed = 204, #this ensures that train and test sets are split identically
        target_size=(180, 180),
        batch_size=128,
        subset = 'validation',
        class_mode='categorical')




# MODEL BUILDING

# Model1: Simple Convolutional Neural Network
model1 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(180, 180, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])


model1.summary()


model1.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(), #categorical crossentropy as loss because we have 
              metrics=['accuracy'])

history = model1.fit(
      train_generator,
      validation_data=val_generator,
      epochs=10,
      verbose=1)

# Plot the performance by epoch
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


# Model2: Deeper Convolutional Neural Network

model2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(180, 180, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
#     # The fourth convolution (You can uncomment the 4th and 5th conv layers later to see the effect)
#     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
#     # The fifth convolution
#     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(5, activation='softmax')
])

model2.summary()


model2.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(), #categorical crossentropy as loss because we have 
              metrics=['accuracy'])

history = model2.fit(
      train_generator,
      validation_data=val_generator,
      epochs=10,
      verbose=1)

# Plot the performance by epoch
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
