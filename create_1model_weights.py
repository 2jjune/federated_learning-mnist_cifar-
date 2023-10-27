import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import h5py
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random

tmp = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu',
                               input_shape=(32, 32, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        # tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.21),
        tf.keras.layers.Conv2D(256, kernel_size=(2, 2), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(256, kernel_size=(2, 2), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Conv2D(256, kernel_size=(2, 2), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.38),
        tf.keras.layers.Conv2D(512, kernel_size=(2, 2), activation='relu', padding='same'),
        # tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dense(20, activation='softmax')
    ])

# tmp = tf.keras.models.Sequential([
#         tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu',
#                                input_shape=(32, 32, 3)),
#         tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
#         tf.keras.layers.Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same'),
#         tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#         tf.keras.layers.Dropout(0.25),
#
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(1000, activation='relu'),
#         tf.keras.layers.Dense(20, activation='softmax')
#     ])

# tmp = tf.keras.models.Sequential([
#         tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu',
#                                input_shape=(28, 28, 1), activity_regularizer = tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)),
#         tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
#         tf.keras.layers.Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same',activity_regularizer = tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)),
#         tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#         tf.keras.layers.Dropout(0.25),
#
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(1000, activation='relu'),
#         tf.keras.layers.Dense(20, activation='softmax')
#     ])

#
# tmp = tf.keras.models.Sequential([
#         tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu',
#                                input_shape=(28, 28, 1), activity_regularizer = tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
#                               kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
#                               bias_regularizer=tf.keras.regularizers.l2(1e-4)),
#         tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
#         tf.keras.layers.Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same',activity_regularizer = tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
#                               kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
#                               bias_regularizer=tf.keras.regularizers.l2(1e-4)),
#         tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#         tf.keras.layers.Dropout(0.25),
#
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(1000, activation='relu'),
#         tf.keras.layers.Dense(20, activation='softmax')
#     ])

#cifar 88% model
# tmp = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     tf.keras.layers.Dropout(0.3),
#
#     tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     tf.keras.layers.Dropout(0.3),
#
#     tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     tf.keras.layers.Dropout(0.3),
#
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(20, activation='softmax')
# ])


tmp.save_weights('first_rgb_avgpool.h5')