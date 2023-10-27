import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import h5py
import numpy as np
from tensorflow.keras import layers, models
import random
from keras import optimizers
import keras
import math
from tensorflow.keras import optimizers
import set_epoch_num
import create_model
import gc
from skimage import color
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# 1. MNIST 데이터셋 임포트
# npad = ((0,0),(2,2),(2,2),(0,0))
# mnist = tf.keras.datasets.fashion_mnist
# (mnist_training_images, mnist_training_labels), (mnist_test_images, mnist_test_labels) = mnist.load_data()
# mnist_training_images = mnist_training_images.reshape(60000, 28, 28, 1)
# mnist_training_images = np.concatenate((mnist_training_images,)*3, axis=-1) # 28,28,3
# mnist_training_images = np.pad(mnist_training_images, pad_width = npad, mode = 'constant', constant_values=0)
# mnist_training_images = mnist_training_images / 255.0
# mnist_test_images = mnist_test_images.reshape(10000, 28, 28, 1)
# mnist_test_images = np.concatenate((mnist_test_images,)*3, axis=-1) # 28,28,3
# mnist_test_images = np.pad(mnist_test_images, pad_width = npad, mode = 'constant', constant_values=0)
# mnist_test_images = mnist_test_images / 255.0

#cifar dataset
(cifar_training_images, cifar_training_labels), (cifar_test_images, cifar_test_labels) = keras.datasets.cifar10.load_data()
cifar_training_images = cifar_training_images.reshape(50000, 32, 32, 3)
cifar_training_images = cifar_training_images / 255.
cifar_test_images = cifar_test_images.reshape(10000, 32, 32, 3)
cifar_test_images = cifar_test_images / 255.

for i in range(10):
    globals()['training_{}'.format(i)] = []
    globals()['labels_{}'.format(i)] = []

label = [0,1,2,3,4,5,6,7,8,9]

for i in range(len(cifar_training_images)):
    for j in range(len(label)):
        if cifar_training_labels[i] == j:
            globals()[f'training_{j}'].append(cifar_training_images[i])
            globals()[f'labels_{j}'].append(j)

x_training = [training_0, training_1, training_2, training_3, training_4,
              training_5, training_6, training_7, training_8, training_9]
y_training = [labels_0, labels_1, labels_2, labels_3, labels_4,
              labels_5, labels_6, labels_7, labels_8, labels_9]


data_size = 5000
set_num = 8
set_epoch = 10000
#---------------------------------------------------------------------------------------------------------------------------

model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu',
                               input_shape=(32, 32, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Dropout(0.21),
        tf.keras.layers.Conv2D(256, kernel_size=(2, 2), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(256, kernel_size=(2, 2), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Conv2D(256, kernel_size=(2, 2), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.38),
        tf.keras.layers.Conv2D(256, kernel_size=(2, 2), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
model.compile(optimizer=tf.keras.optimizers.SGD(lr=set_epoch_num.lr, decay=set_epoch_num.decay),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
for i in range(set_num):
    # model_i = model[i]
    print()

    print(i,"*******************************")
    # train_generator, validation_generator = set_mnist_generator.set_generator(train_dir+str(x+1))

    print('학습 데이터 수 : 600')
    model.load_weights('first_cifar.h5')
    r = random.randrange(0, 10)
    # np.squeeze(cifar_test_labels, axis=1)
    # model[0].fit(mnist_train_generator, steps_per_epoch=len(train_generator), epochs=set_epoch, validation_data=set_generator.test_generator, validation_steps=len(set_generator.test_generator))
    # hist = model.fit(np.concatenate((np.array(random.sample(x_training[r], data_size)), np.array(cifar_training_images[:int(data_size*0.05)])), axis=0),
    #               np.concatenate((np.array(y_training[r][:data_size]), np.array(np.squeeze(cifar_training_labels[:int(data_size*0.05)]))), axis=0),
    #               epochs=set_epoch_num.set_epoch, batch_size=64,
    #               validation_data=(cifar_test_images, cifar_test_labels))
    hist = model.fit(np.array(random.sample(x_training[r], data_size)),
                  np.array(y_training[r][:data_size]),
                  epochs=set_epoch_num.set_epoch, batch_size=64,
                  validation_data=(cifar_test_images, cifar_test_labels))
    model.save_weights('{}_epoch {}.h5'.format(set_epoch,i+1))
    # print(hist.history['val_accuracy'])
    # print(np.array(hist.history['val_accuracy']).mean(axis=0))
    # tmp.append(hist.history['val_accuracy'][-1])
    del r
    del hist

for x in range(10000):
    tmp_model = []
    weights = []
    new_weights_median = list()
    new_weights_avg = list()

    for i in range(set_num):
        tmp_model.append(create_model.create_model())

    for i in range(set_num):
        if i < 10:
            tmp_model[i].load_weights("{}_epoch {}.h5".format(set_epoch, i + 1))
            print("{}_epoch {}.h5".format(set_epoch, i + 1))
        # else:
        #     tmp_model[i].load_weights("{}_epoch {}.h5".format(set_epoch, i + 1))
        #     print("{}_epoch {}.h5".format(set_epoch, i + 1))

    for i in range(set_num):
        weights.append(tmp_model[i].get_weights())
    print('weights 갯수 : ', len(weights))

    for weights_list_tuple in zip(*weights):
        # new_weights_median.append(
        #     np.array([np.median(np.array(w), axis=0) for w in zip(*weights_list_tuple)])
        # )
        new_weights_avg.append(
            np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
        )

    # model[0].set_weights(new_weights_median)
    model = create_model.create_model()
    model.set_weights(new_weights_avg)
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=set_epoch_num.lr, decay=set_epoch_num.decay),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    test_loss, test_acc = model.evaluate(cifar_test_images, cifar_test_labels)
    print()
    # print('---------------{}번째 median 테스트 정확도: {}, loss: {}-----------------'.format(x, test_acc, test_loss))
    print('---------------{}번째 avg 테스트 정확도: {}, loss: {}-----------------'.format(x, test_acc, test_loss))

    del model
    print()
    print('dataset retraining...')
    for i in range(set_num):
        print()
        print(i, "*******************************")

        print('학습 데이터 수 : 600')
        model=create_model.create_model()
        model.set_weights(new_weights_avg)
        model.compile(optimizer=tf.keras.optimizers.SGD(lr=set_epoch_num.lr, decay=set_epoch_num.decay),
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
        # model[0].fit(mnist_train_generator, steps_per_epoch=len(train_generator), epochs=set_epoch, validation_data=set_generator.test_generator, validation_steps=len(set_generator.test_generator))
        r = random.randrange(0, 10)
        # hist= model.fit(np.concatenate((np.array(random.sample(x_training[r], data_size)), np.array(cifar_training_images[:int(data_size*0.05)])), axis=0),
        #           np.concatenate((np.array(y_training[r][:data_size]), np.array(np.squeeze(cifar_training_labels[:int(data_size*0.05)]))), axis=0),
        #           epochs=set_epoch_num.set_epoch, batch_size=64,
        #           validation_data=(cifar_test_images, cifar_test_labels))
        hist = model.fit(np.array(random.sample(x_training[r], data_size)),
                         np.array(y_training[r][:data_size]),
                         epochs=set_epoch_num.set_epoch, batch_size=64,
                         validation_data=(cifar_test_images, cifar_test_labels))
        model.save_weights('{}_epoch {}.h5'.format(set_epoch, i + 1))
    del r
    del model
    del tmp_model
    del weights
    del new_weights_median
    del new_weights_avg
    del hist
    # cuda.select_device(0)
    # cuda.close()
    gc.collect()
    tf.keras.backend.clear_session()
