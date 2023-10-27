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
npad = ((0,0),(2,2),(2,2),(0,0))
mnist = tf.keras.datasets.fashion_mnist
(mnist_training_images, mnist_training_labels), (mnist_test_images, mnist_test_labels) = mnist.load_data()
mnist_training_images = mnist_training_images.reshape(60000, 28, 28, 1)
mnist_training_images = np.concatenate((mnist_training_images,)*3, axis=-1) # 28,28,3
mnist_training_images = np.pad(mnist_training_images, pad_width = npad, mode = 'constant', constant_values=0)
mnist_training_images = mnist_training_images / 255.0
mnist_test_images = mnist_test_images.reshape(10000, 28, 28, 1)
mnist_test_images = np.concatenate((mnist_test_images,)*3, axis=-1) # 28,28,3
mnist_test_images = np.pad(mnist_test_images, pad_width = npad, mode = 'constant', constant_values=0)
mnist_test_images = mnist_test_images / 255.0

#cifar dataset
(cifar_training_images, cifar_training_labels), (cifar_test_images, cifar_test_labels) = keras.datasets.cifar10.load_data()
cifar_training_images = cifar_training_images.reshape(50000, 32, 32, 3)
cifar_training_images = cifar_training_images / 255.
cifar_test_images = cifar_test_images.reshape(10000, 32, 32, 3)
cifar_test_images = cifar_test_images / 255.
cifar_training_labels = cifar_training_labels+10
cifar_test_labels = cifar_test_labels+10


test_images = np.append(mnist_test_images, np.array(cifar_test_images), axis=0)
cifar_test_labels = np.squeeze(cifar_test_labels,axis=1)
test_labels = np.append(mnist_test_labels, np.array(cifar_test_labels), axis=0)

for i in range(20):
    # globals()['mnist_training_{}'.format(i)] = []
    # globals()['mnist_labels_{}'.format(i)] = []
    # globals()['cifar_training_{}'.format(i)] = []
    # globals()['cifar_labels_{}'.format(i)] = []
    globals()['training_{}'.format(i)] = []
    globals()['labels_{}'.format(i)] = []

label = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]

for i in range(len(mnist_training_images)):
    for j in range(len(label)):
        if mnist_training_labels[i] == j:
            globals()[f'training_{j}'].append(mnist_training_images[i])
            globals()[f'labels_{j}'].append(j)

for i in range(len(cifar_training_images)):
    for j in range(len(label)):
        if cifar_training_labels[i] == j:
            globals()[f'training_{j}'].append(cifar_training_images[i])
            globals()[f'labels_{j}'].append(j)

x_training = [training_0, training_1, training_2, training_3, training_4,
              training_5, training_6, training_7, training_8, training_9,
              training_10, training_11, training_12, training_13, training_14,
              training_15, training_16, training_17, training_18, training_19]
y_training = [labels_0, labels_1, labels_2, labels_3, labels_4,
              labels_5, labels_6, labels_7, labels_8, labels_9,
              labels_10, labels_11, labels_12, labels_13, labels_14,
              labels_15, labels_16, labels_17, labels_18, labels_19]
data_size = set_epoch_num.data_size
set_num = set_epoch_num.set_num
set_epoch = set_epoch_num.set_epoch
#---------------------------------------------------------------------------------------------------------------------------

x=0
model = create_model.create_model()
model.compile(optimizer=tf.keras.optimizers.SGD(lr=set_epoch_num.lr, decay=set_epoch_num.decay),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
tmp=[]
for i in range(set_num):
    # model_i = model[i]
    print()

    if x+1>10:
        x = int(x/10)-1
    print(i,"*******************************")
    # train_generator, validation_generator = set_mnist_generator.set_generator(train_dir+str(x+1))
    x+=1

    print('학습 데이터 수 : 600')
    model.load_weights('first_rgb_avgpool.h5')
    r = random.randrange(0, 20)
    # np.squeeze(cifar_test_labels, axis=1)
    # model[0].fit(mnist_train_generator, steps_per_epoch=len(train_generator), epochs=set_epoch, validation_data=set_generator.test_generator, validation_steps=len(set_generator.test_generator))
    hist = model.fit(np.concatenate((np.array(random.sample(x_training[r], data_size)), np.array(cifar_training_images[:int(data_size*0.05)])), axis=0),
                  np.concatenate((np.array(y_training[r][:data_size]), np.array(np.squeeze(cifar_training_labels[:int(data_size*0.05)]))), axis=0),
                  epochs=set_epoch_num.set_epoch, batch_size=64,
                  validation_data=(test_images, test_labels))
    model.save_weights('{}_epoch {}.h5'.format(set_epoch,i+1))
    # print(hist.history['val_accuracy'])
    # print(np.array(hist.history['val_accuracy']).mean(axis=0))
    tmp.append(hist.history['val_accuracy'][-1])
    del r
    del hist
f = open('result.txt', 'w')
data = "accuracy : %f\n" % np.array(tmp).mean(axis=0)
f.write(data)
f.close()


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

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print()
    # print('---------------{}번째 median 테스트 정확도: {}, loss: {}-----------------'.format(x, test_acc, test_loss))
    print('---------------{}번째 avg 테스트 정확도: {}, loss: {}-----------------'.format(x, test_acc, test_loss))
    f = open('result.txt', 'a')
    f_acc = open('result_acc.txt', 'a')
    f_loss = open('result_loss.txt', 'a')
    # data = "%d번 median result accuracy : %f, loss : %f\n\n" % (x,test_acc, test_loss)
    data = "%d번 avg result accuracy : %f, loss : %f\n\n" % (x,test_acc, test_loss)
    data_acc = "%f\n"%(test_acc)
    data_loss = "%f\n"%(test_loss)
    f.write(data)
    f_acc.write(data_acc)
    f_loss.write(data_loss)
    f.close()
    f_acc.close()
    f_loss.close()
    del model
    print()
    print('dataset retraining...')
    tmp=[]
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
        r = random.randrange(0, 20)
        hist= model.fit(np.concatenate((np.array(random.sample(x_training[r], data_size)), np.array(cifar_training_images[:int(data_size*0.05)])), axis=0),
                  np.concatenate((np.array(y_training[r][:data_size]), np.array(np.squeeze(cifar_training_labels[:int(data_size*0.05)]))), axis=0),
                  epochs=set_epoch_num.set_epoch, batch_size=64,
                  validation_data=(test_images, test_labels))
        model.save_weights('{}_epoch {}.h5'.format(set_epoch, i + 1))
        tmp.append(hist.history['val_accuracy'][-1])
    f = open('result.txt', 'a')
    data = "%d번 accuracy : %f\n" % (x+1, np.array(tmp).mean(axis=0))
    f.write(data)
    f.close()
    del r
    del model
    del tmp_model
    del weights
    del new_weights_median
    del new_weights_avg
    del tmp
    del data
    del hist
    # cuda.select_device(0)
    # cuda.close()
    gc.collect()
    tf.keras.backend.clear_session()






#-------------------------------------------------------------------------------------------------
'''
for x in range(1000):
    weights = []
    for j in range(set_epoch_num.set_num):
        model = create_model.create_model()
        model.load_weights('first_rgb.h5')
        model.compile(optimizer=tf.keras.optimizers.SGD(lr=set_epoch_num.lr, decay=set_epoch_num.decay),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        r = random.randrange(0, 20)
        model.fit(np.array(random.sample(x_training[r], data_size)),
                  np.array(y_training[r][:data_size]),
                  epochs=set_epoch_num.set_epoch, batch_size=64,
                  validation_data=(test_images, test_labels))
        weights.append(model.get_weights())

    new_model = create_model.create_model()
    new_weights = []
    for weights_list_tuple in zip(*weights):
        new_weights.append(
            np.array([np.mean(np.array(w), axis=0) for w in zip(*weights_list_tuple)])
        )

    new_model.set_weights(new_weights)
    new_model.compile(optimizer=tf.keras.optimizers.SGD(lr=set_epoch_num.lr, decay=set_epoch_num.decay),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    test_loss, test_acc = new_model.evaluate(test_images, test_labels)
    print()
    print('---------------{}번째 avg 테스트 정확도: {}, loss: {}-----------------'.format(x, test_acc, test_loss))

    f = open('result.txt', 'a')
    data = "%d번 avg result accuracy : %f, loss : %f\n\n" % (x, test_acc, test_loss)
    f.write(data)
    f.close()

    model.save_weights('cnnfirst{}.h5'.format(j + 1))

'''
'''
for i in range(60000):
    if training_labels[i] == 0:
        mnist_training_0.append(training_images[i])
        mnist_labels_0.append(0)
    elif training_labels[i] == 1:
        mnist_training_1.append(training_images[i])
        mnist_labels_1.append(1)
    elif training_labels[i] == 2:
        mnist_training_2.append(training_images[i])
        mnist_labels_2.append(2)
    elif training_labels[i] == 3:
        mnist_training_3.append(training_images[i])
        mnist_labels_3.append(3)
    elif training_labels[i] == 4:
        mnist_training_4.append(training_images[i])
        mnist_labels_4.append(4)
    elif training_labels[i] == 5:
        mnist_training_5.append(training_images[i])
        mnist_labels_5.append(5)
    elif training_labels[i] == 6:
        mnist_training_6.append(training_images[i])
        mnist_labels_6.append(6)
    elif training_labels[i] == 7:
        mnist_training_7.append(training_images[i])
        mnist_labels_7.append(7)
    elif training_labels[i] == 8:
        mnist_training_8.append(training_images[i])
        mnist_labels_8.append(8)
    elif training_labels[i] == 9:
        mnist_training_9.append(training_images[i])
        mnist_labels_9.append(9)

x_training = [x0_training, x1_training, x2_training, x3_training, x4_training, x5_training, x6_training, x7_training,
              x8_training, x9_training]

y_training = [y0_labels, y1_labels, y2_labels, y3_labels, y4_labels, y5_labels, y6_labels, y7_labels, y8_labels,
              y9_labels]

data_size = 6000
x = []

# 3. 모델 구성
sgd = optimizers.SGD(lr=0.001, decay=1e-4)
new_weights = list()

for j in range(10000):
    weights = []
    for i in range(10):
        model = models.Sequential([
            layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                          padding='same', activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            layers.Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(1000, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        model.load_weights('cnnfirst{}.h5'.format(j))
        model.compile(optimizer=sgd,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        r = random.randrange(0, 10)
        model.fit(np.concatenate((np.array(random.sample(x_training[r], data_size)), np.array(training_images[:300])),
                                 axis=0),
                  np.concatenate((np.array(y_training[r][:data_size]), np.array(training_labels[:300])), axis=0),
                  epochs=1, batch_size=16,
                  validation_data=(test_images, test_labels))
        weights.append(model.get_weights())
    new_model = models.Sequential([
        layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                      padding='same', activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        layers.Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(1000, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    new_weights = []
    for weights_list_tuple in zip(*weights):
        new_weights.append(
            np.array([np.mean(np.array(w), axis=0) for w in zip(*weights_list_tuple)])
        )

    new_model.set_weights(new_weights)
    new_model.compile(optimizer=sgd,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy']
                      )

    test_loss, test_acc = new_model.evaluate(test_images, test_labels)
    print('테스트 정확도:', test_acc)
    File = open("시나리오3 랜덤10 6000 " + str(data_size) + ".txt", "a")
    File.write(str(test_acc) + "\n")
    File.close()

    model.save_weights('cnnfirst{}.h5'.format(j + 1))'''