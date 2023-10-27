import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import cv2
from keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import create_model
import random
import set_epoch_num
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py
import numpy as np
from tensorflow.keras import layers, models
import random
from keras import optimizers
import keras
import math
import matplotlib.pyplot as plt
import time
# categories = ["airplane", "Angle boot", "automobile", "Bag", "bird", "cat", "Coat", "deer", "dog", "Dress",
#               "frog", "horse", "Pullover", "Sandal", "ship", "Shirt", "Sneaker", "Trouser", "truck", "T-shirt"]
#
#
# def Dataization(img_path):
#     image_w = 28
#     image_h = 28
#     img = cv2.imread(img_path)
#     img = cv2.resize(img, None, fx=image_w / img.shape[1], fy=image_h / img.shape[0])
#     return (img / 256)
#
#
# src = []
# name = []
# test = []
# image_dir = "0001"
# for file in os.listdir(image_dir):
#     if (file.find('.png') is not -1):
#         src.append(image_dir + file)
#         name.append(file)
#         test.append(Dataization(image_dir + file))
#
# test = np.array(test)
# model = load_model('Gersang.h5')
# predict = model.predict_classes(test)
#
# for i in range(len(test)):
#     print(name[i] + " : , Predict : " + str(categories[predict[i]]))

#
# y = [0.1,0.2,0.5,0.8,0.01]
# x = 3
# p = np.argmax(y)
# if np.argmax(y) == x and y[p]>=0.9:
#     print(1)

#########################################################
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(20, activation='softmax')
    ])
prev_time=time.time()
model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images,train_labels, batch_size=64, epochs=50, validation_split=0.2)
loss,acc = model.evaluate(test_images,test_labels)
print(loss, acc)
print('time:',time.time()-prev_time)
###########################################################



# base_dir = "data/"
# test_dir = os.path.join(base_dir, "test")
# train_dir = os.path.join(base_dir, "all_train")
# train_datagen = ImageDataGenerator(rescale=1./255,
#                                    horizontal_flip = True,
#                                    vertical_flip = True,
#                                    validation_split=0.2)
# val_datagen = ImageDataGenerator(rescale = 1./255, validation_split=0.2)
# test_datagen = ImageDataGenerator(rescale=1./255)
#
#
#
#
# train_generator = train_datagen.flow_from_directory(
#         train_dir,
#         target_size=(32, 32),
#         batch_size=64,
#         shuffle=True,
#         subset = 'training',
#         color_mode = 'rgb',
#         class_mode='categorical')
# validation_generator = val_datagen.flow_from_directory(
#         train_dir,
#         target_size=(32, 32),
#         batch_size=64,
#         shuffle=True,
#         subset = 'validation',
#         color_mode = 'rgb',
#         class_mode='categorical')
#
# test_generator = test_datagen.flow_from_directory(
#         test_dir,
#         target_size=(32, 32),
#         batch_size=64,
#         color_mode = 'rgb',
#         class_mode='categorical')
#
#
# model = create_model.create_model()
# # model.compile(optimizer=tf.keras.optimizers.SGD(lr=1e-2, decay=set_epoch_num.decay), loss='categorical_crossentropy',
# #               metrics=['accuracy'])
# model.compile(optimizer=tf.keras.optimizers.SGD(lr=1e-1, decay = 8e-8), loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# # train_step = random.randrange(2, int(len(mnist_train_generator) / 2))
# train_step = 10
# # print('학습 데이터 수 :', train_step * 64)
# # model.load_weights('first_rgb.h5')
#
# # model[0].fit(mnist_train_generator, steps_per_epoch=len(train_generator), epochs=set_epoch, validation_data=set_generator.test_generator, validation_steps=len(set_generator.test_generator))
# # hist = model.fit(train_generator, steps_per_epoch=train_step, epochs=1000,
# #                     validation_data=test_generator, validation_steps=len(test_generator))
# hist = model.fit(train_generator, steps_per_epoch=len(train_generator), epochs=1000,
#                     validation_data=test_generator, validation_steps=len(test_generator))
# f = open('test_result.txt', 'a')
# data = "%f\n" % (hist.history['val_accuracy'][-1])
# f.write(data)
# f.close()
#
# test_loss, test_acc = model.evaluate(test_generator)
# print()
# print('---------------테스트 정확도: {}, loss: {}-----------------'.format(test_acc, test_loss))
