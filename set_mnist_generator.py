import os
import set_epoch_num
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import h5py
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import create_model
import set_epoch_num


base_dir = "data/"
test_dir = os.path.join(base_dir, "mnist_test")
train_dir = os.path.join(base_dir, "mnist_train")
train_datagen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip = True,
                                   vertical_flip = True,
                                   validation_split=0.2)
val_datagen = ImageDataGenerator(rescale = 1./255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

def set_generator(train_dir):
    train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(32, 32),
            batch_size=64,
            shuffle=True,
            subset = 'training',
            color_mode = 'rgb',
            class_mode='categorical')
    validation_generator = val_datagen.flow_from_directory(
            train_dir,
            target_size=(32, 32),
            batch_size=64,
            shuffle=True,
            subset = 'validation',
            color_mode = 'rgb',
            class_mode='categorical')
    return train_generator, validation_generator


test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(32, 32),
        batch_size=64,
        color_mode = 'rgb',
        class_mode='categorical')